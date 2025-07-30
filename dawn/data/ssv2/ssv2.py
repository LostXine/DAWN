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
from tqdm import tqdm
from torch.utils.data import Dataset
from dawn.data.base_dataset import BaseDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)

class SSv2Dataset(BaseDataset):
    def __init__(self, 
        data_path, 
        split="training", 
        image_size=256,
        num_frames=2,
        num_actions=10,
        min_skip=10,
        max_skip=30,
        cache_metadata=True,
        observation_type: List[str] = ["rgb_static", "rgb_gripper"],  # Default observation types,
        observation_from: List[str] = ["", ""],  # Default observation types,
        **kwargs
    ):
        self.data_path = os.path.join(data_path)
        super().__init__(data_path, split, image_size, num_frames, num_actions, min_skip, max_skip, cache_metadata, observation_type, observation_from)

    def _get_transform(self):
        if self.split == "training":
            return A.Compose([
                # A.PadIfNeeded(min_height=320, min_width=320, border_mode=0, value=0, mask_value=0),
                # A.Affine(scale=(0.8, 1.2), rotate=(-15, 15),
                #     translate_percent=(-0.1, 0.1), shear=(-10, 10), p=0.5),
                A.Resize(self.image_size, self.image_size),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                ToTensorV2(),
            ])
        return A.Compose([
            # A.PadIfNeeded(min_height=320, min_width=320, border_mode=0, value=0, mask_value=0),
            A.Resize(self.image_size, self.image_size),
            ToTensorV2(),
        ])

    def __len__(self):
        # if self.split == "training":
        #     return len(self.episodes) * 1000
        
        return len(self.episodes)
    
    def __getitem__(self, idx):
        idx = idx % len(self.episodes)
        return super().__getitem__(idx)
        
    def get_action(self, metadata, frame_idx):
        # cartesian = metadata["action_dict"]["cartesian_velocity"][frame_idx: frame_idx + self.num_actions]
        # gripper = metadata["action_dict"]["gripper_velocity"][frame_idx: frame_idx + self.num_actions]
        # action = np.concatenate([cartesian, gripper], axis=-1)
        # action = torch.tensor(action, dtype=torch.float32)
        action = torch.zeros((self.num_actions, 7), dtype=torch.float32)
        return action
    
    def _load_episodes(self, num_workers=8): # Adjust num_workers as needed
        """
        Loads all episodes in parallel using a thread pool.
        """
        json_file = json.load(open(os.path.join(self.data_path, "labels", self.split + ".json"), "r"))
        
        episodes = []
        for idx, item in enumerate(tqdm(json_file)):
            ep_path = os.path.join(self.data_path, "rawframes", item["id"])
            episode = {
                "idx": item["id"],#idx,
                "path": ep_path
            }

            metadata = {
                "language": item["label"],
            }

            episode["metadata"] = metadata
            episodes.append(episode)

        logger.info(f"Loaded {len(episodes)} episodes from {self.data_path}")
        # The list of episodes may not be in the original sorted order, so sort it now if needed.
        episodes.sort(key=lambda x: x['idx'])
        return episodes

        data = []
        for episode in tqdm(episodes):
            for i in range(episode["metadata"]["length"]):
                data.append((i, episode))
        logger.info(f"Loaded {len(data)} frames from {self.data_path}")
        
        return data


    # def __getitem__(self, idx):
    #     # idx=0
    #     # idx = 0
    #     episode = self.episodes[idx]
    #     metadata = episode["metadata"]

    #     # Augment the language in the same task
    #     language = random.choice(self.annos[metadata["task"]])
    #     data = {
    #         "idx": episode["idx"],
    #         "language": language # metadata["language"],
    #         # "language_embedding": torch.tensor(metadata["language_embedding"], dtype=torch.float32),
    #     }
    #     frames = [random.choice(range(metadata["length"]))]
    #     skips = []
    #     while len(frames) < self.num_frames:
    #         skip = random.randint(self.min_skip, self.max_skip)
    #         next_idx = min(metadata["length"] - 1, frames[-1] + skip)
    #         frames.append(next_idx)
    #         skips.append(skip)
    #     frame_idx = frames[-2]
    #     # frame_idx = random.choice(range(metadata["length"]))
    #     # frame_idx = 0
    #     data["action"] = torch.tensor(metadata["rel_actions"][frame_idx: frame_idx + self.num_actions])
    #     # Pad actions if they are less than num_actions
    #     if len(data["action"]) < self.num_actions:
    #         pad_length = self.num_actions - len(data["action"])
    #         last_action = data["action"][-1:]
    #         data["action"] = torch.cat([data["action"], last_action.repeat(pad_length, 1)], dim=0)
    #         # data["action"] = torch.cat([data["action"], torch.zeros((pad_length, data["action"].shape[1]), dtype=torch.float32)], dim=0)
        
    #     # next_idx = min(metadata["length"] - 1, frame_idx + self.num_actions)
    #     # skip = random.randint(self.min_skip, self.max_skip)
    #     # next_idx = min(metadata["length"] - 1, frame_idx + skip)

    #     skip = skips[-1]
    #     data["skip_frame"] = torch.tensor(skip, dtype=torch.int64)

    #     for obs_type in self.observation_type:
    #         obs_files = sorted(glob.glob(os.path.join(episode["path"], obs_type, '*.jpg')))
    #         images = np.stack([imageio.imread(obs_files[idx]) for idx in frames], axis=0)
    #         data[obs_type] = self.transform(images=images)["images"] / 255.
    #         # data[obs_type] = torch.stack([self.transform(image=)["image"] ]) / 255. 
    #     return data
