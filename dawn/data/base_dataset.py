from typing import List
import logging
import glob
import os
import torch
import datetime
import json
import imageio
import random
from fvcore.common.timer import Timer
import albumentations as A
from albumentations.pytorch import ToTensorV2
from omegaconf  import OmegaConf
import numpy as np
from tqdm.auto import tqdm
import cv2
import time
from functools import lru_cache
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

# from torchvision.models import optical_flow
# from torchvision.utils import flow_to_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)

class BaseDataset(Dataset):
    def __init__(self, 
        data_path, 
        split="training", 
        image_size=256,
        num_frames=2,
        num_actions=10,
        min_skip=10,
        max_skip=30,
        cache_metadata=True,
        observation_type: List[str] = ["rgb_static", "rgb_gripper"],  # Default observation types
        observation_from: List[str] = None,
        action_type: str = "actions",
        **kwargs
    ):
        timer = Timer()

        self.observation_type = observation_type
        self.observation_from = observation_from if observation_from else observation_type
        self.image_size = image_size
        self.num_frames = num_frames
        self.num_actions = num_actions
        self.min_skip = min_skip
        self.max_skip = max_skip
        self.cache_metadata = cache_metadata
        self.action_type = action_type
        self.split = split

        if not os.path.exists(self.data_path):
            raise ValueError(f"Data path {self.data_path} does not exist.")

        self.episodes = self._load_episodes()
        self.transform = self._get_transform()


        # self.flow_model = optical_flow.raft_large(weights=optical_flow.Raft_Large_Weights.DEFAULT, progress=False).eval()
        # self.flow_model.requires_grad_(False)
        # self.flow_transform = optical_flow.Raft_Large_Weights.DEFAULT.transforms()
        
        logger.info(f"Loading from {data_path} takes {timer.seconds():.2f} seconds.")

    def _get_transform(self):
        if self.split == "training":
            return A.Compose([
                # A.Affine(scale=(0.8, 1.2), rotate=(-15, 15),
                #     translate_percent=(-0.1, 0.1), shear=(-10, 10), p=0.5),
                A.Resize(self.image_size, self.image_size),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.5),
                ToTensorV2(),
            ])
        return A.Compose([
            A.Resize(self.image_size, self.image_size),
            ToTensorV2(),
        ])

    def _load_metadata(self, episode):
        if "metadata" not in episode:
            metadata = json.load(open(os.path.join(episode["path"], "metadata.json"), "r"))
        else:
            metadata = episode["metadata"]

        if "frames" not in metadata:
            metadata["frames"] = sorted(os.listdir(os.path.join(episode["path"], self.observation_from[0])))
            metadata["frames"] = metadata["frames"][:metadata["length"]]
            # metadata["length"] = len(metadata["frames"])
        
        # if "last_idx_same_gripper" not in metadata:
        #     metadata["last_idx_same_gripper"] = [metadata["length"] - 1] * metadata["length"]
        #     if "actions" in metadata:
        #         for i in range(metadata["length"] - 2, -1, -1):
        #             if metadata["actions"][i][6] == metadata["actions"][i + 1][6]:
        #                 metadata["last_idx_same_gripper"][i] = metadata["last_idx_same_gripper"][i + 1]
        #             else:
        #                 metadata["last_idx_same_gripper"][i] = i
                # for i in range(metadata["length"]):
                #     print(i, metadata["last_idx_same_gripper"][i], metadata["actions"][i][6], torch.tensor(metadata["rel_actions"][i])[:6].abs().mean(), metadata["rel_actions"][i])
                # print("----------")
        if self.cache_metadata:
            episode["metadata"] = metadata
        
        return metadata

    def _load_single_episode(self, episode_file):
        """
        Processes a single episode file. This function will be run in parallel.
        """
        episode_path = os.path.join(self.data_path, episode_file)
        episode = {
            "idx": episode_file,
            "path": episode_path,
        }
        
        # metadata_path = os.path.join(episode_path, "metadata.json")
        # metadata = json.load(open(metadata_path, "r"))

        # obs_files = sorted(os.listdir(os.path.join(episode_path, self.observation_from[0])))
        # metadata["frames"] = obs_files
        # metadata["length"] = len(obs_files)
        # episode["metadata"] = metadata
        
        return episode
        # for obs_type, obs_from in zip(self.observation_type, self.observation_from):
        #     frames = [os.path.join(episode_path, obs_from, x) for x in metadata["frames"]]
        #     # This is the slow part that now runs in parallel for each episode
        #     images = np.stack([self.read_image(obs_file) for obs_file in frames], axis=0)
        #     episode[obs_type] = images


    def _load_episodes(self, num_workers=8): # Adjust num_workers as needed
        """
        Loads all episodes in parallel using a thread pool.
        """
        episode_files = sorted(os.listdir(self.data_path))
        episodes = []
        
        # # Use ThreadPoolExecutor to parallelize the loading
        # with ThreadPoolExecutor(max_workers=num_workers) as executor:
        #     # Create a future for each episode file to be loaded
        #     futures = [executor.submit(self._load_single_episode, f) for f in episode_files]
            
        #     # Use tqdm to show progress as episodes complete
        #     for future in tqdm(as_completed(futures), total=len(episode_files), desc="Loading episodes"):
        #         try:
        #             episode = future.result()
        #             episodes.append(episode)
        #         except Exception as e:
        #             logger.error(f"Failed to load an episode: {e}")
        episodes = [self._load_single_episode(f) for f in tqdm(episode_files, desc="Loading episodes")]
        logger.info(f"Loaded {len(episodes)} episodes from {self.data_path}")
        # The list of episodes may not be in the original sorted order, so sort it now if needed.
        # episodes.sort(key=lambda x: x['idx'])
        return episodes

        data = []
        for episode in tqdm(episodes):
            for i in range(self._load_metadata(episode)["length"]):
                data.append((i, episode))
        logger.info(f"Loaded {len(data)} frames from {self.data_path}")
        
        return data

        # return episodes

    # def _load_episodes(self):
        
    #     # Load episodes
    #     episodes = []
    #     for episode_file in tqdm(sorted(os.listdir(self.data_path))):
    #         episode = {
    #             "idx": episode_file,
    #             "path": os.path.join(self.data_path, episode_file),
    #         }
    #         metadata = json.load(open(os.path.join(episode["path"], "metadata.json"), "r"))

    #         obs_files = sorted(os.listdir(os.path.join(episode["path"], self.observation_from[0])))
    #         metadata["frames"] = obs_files
    #         metadata["length"] = len(obs_files)

    #         for obs_type, obs_from in zip(self.observation_type, self.observation_from):
    #             obs_path = os.path.join(episode["path"], obs_from)
    #             frames = [os.path.join(episode["path"], obs_from, x) for x in metadata["frames"]]
    #             images = np.stack([self.read_image(obs_file) for obs_file in frames], axis=0)
    #             episode[obs_type] = images

    #         episode["metadata"] = metadata
    #         episodes.append(episode)
            
    #     logger.info(f"Loaded {len(episodes)} episodes from {self.data_path}") 
    #     return episodes

    def __len__(self):
        return len(self.episodes)
    
    def get_frame_indices(self, episode_metadata, first_frame=None):
        metadata = episode_metadata
        if first_frame is not None:
            frames = [first_frame]
        else:
            frames = [random.choice(range(metadata["length"]))]
        skips = []
        while len(frames) < self.num_frames:
            skip = random.randint(self.min_skip, self.max_skip)
            next_idx = min(metadata["length"] - 1, frames[-1] + skip)
            # next_idx = min(next_idx, metadata["last_idx_same_gripper"][frames[-1]] + 1)
            frames.append(next_idx)
            skips.append(skip)
        return frames, skips

    def get_action(self, episode_metadata, frame_idx):
        metadata = episode_metadata
        # action = torch.tensor(metadata["rel_actions"][frame_idx: frame_idx + self.num_actions])
        action = torch.tensor(metadata[self.action_type][frame_idx: frame_idx + self.num_actions])
        
        return action

    # @lru_cache(maxsize=32768)
    def read_image(self, path):
        # return np.random.randint(0, 255, (self.image_size, self.image_size, 3)).astype(np.uint8)
        return np.array(Image.open(path))
        # return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx):
        # logger.info(f"Getting item {idx} from {self.split} split")
        episode = self.episodes[idx]
        # first_frame, episode = self.episodes[idx]
        first_frame = None
        
        metadata = self._load_metadata(episode)

        # Augment the language in the same task
        try:
            if "task" not in metadata:
                metadata["task"] = self.r_map[metadata["language"]]
            language = random.choice(self.annos[metadata["task"]])
        except:
            if isinstance(metadata["language"], list):
                language = random.choice(metadata["language"])
            else:
                language = metadata["language"]

        data = {
            "idx": episode["idx"],
            "language": language 
        }
        
        frames, skips = self.get_frame_indices(metadata, first_frame=first_frame)

        frame_idx = frames[-2]
        data["action"] = self.get_action(metadata, frame_idx)

        # Pad actions if they are less than num_actions 
        if len(data["action"]) < self.num_actions:
            pad_length = self.num_actions - len(data["action"])
            last_action = data["action"][-1:]
            data["action"] = torch.cat([data["action"], last_action.repeat(pad_length, 1)], dim=0)
            # data["action"] = torch.cat([data["action"], torch.zeros((pad_length, data["action"].shape[1]), dtype=torch.float32)], dim=0)
        
        # if abs(data["action"][:, :6]).mean() < 0.05 and data["action"][:, 6].min() == data["action"][:, 6].max():
        #     return self.__getitem__(random.randint(0, self.__len__() - 1))
        # print(episode["path"], data["action"].shape)
        skip = skips[-1]
        data["skip_frame"] = torch.tensor(skip, dtype=torch.int64)
        data["frame_idx"] = torch.tensor(frames)

        for obs_type, obs_from in zip(self.observation_type, self.observation_from):
            stime = time.time()
            if obs_type not in episode:
                if "frames" in metadata:
                    obs_files = [os.path.join(episode["path"], obs_from, x) for x in metadata["frames"]]
                else:
                    obs_files = sorted(glob.glob(os.path.join(episode["path"], obs_type, '*')))
                images = np.stack([self.read_image(obs_files[i]) for i in frames], axis=0)
            else:
                images = episode[obs_type][frames]
            etime = time.time()
            # logger.info(f"Loading {obs_type} took {etime - stime:.2f} seconds")
            data[obs_type] = self.transform(images=images)["images"] / 255.
            # logger.info(f"{episode["path"]}, {frames}")

        return data
