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

from torch.utils.data import Dataset
from dawn.data.base_dataset import BaseDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)

class RealworldDataset(BaseDataset):
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
        action_type="actions",
        **kwargs
    ):
        self.data_path = os.path.join(data_path, "episodes")
        # Load task + language annotations
        try:
            self.annos = OmegaConf.load(os.path.join(data_path, "annotations.yaml"))
            self.r_map = {}
            for k, v in self.annos.items():
                v.append(k.replace("_", " "))
                for x in v:
                    self.r_map[x] = k
        except:
            logger.warning("No annotations found, using default language annotations.")

        super().__init__(data_path, split, image_size, num_frames, num_actions, min_skip, max_skip, cache_metadata, observation_type, action_type=action_type)


    def __len__(self):
        return len(self.episodes)
    
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
        # data["robot_obs"] = self.get_robot_state(metadata, frame_idx)

        # Pad actions if they are less than num_actions 
        if len(data["action"]) < self.num_actions:
            pad_length = self.num_actions - len(data["action"])
            last_action = data["action"][-1:]
            data["action"] = torch.cat([data["action"], last_action.repeat(pad_length, 1)], dim=0)
        
        skip = skips[-1]
        data["skip_frame"] = torch.tensor(skip, dtype=torch.int64)
        data["frame_idx"] = torch.tensor(frames)

        for obs_type, obs_from in zip(self.observation_type, self.observation_from):
            if obs_type not in episode:
                if "frames" in metadata:
                    obs_files = [os.path.join(episode["path"], obs_from, x) for x in metadata["frames"]]
                else:
                    obs_files = sorted(glob.glob(os.path.join(episode["path"], obs_type, '*')))
                images = np.stack([self.read_image(obs_files[i]) for i in frames], axis=0)
            else:
                images = episode[obs_type][frames]
            data[obs_type] = self.transform(images=images)["images"] / 255.
            
            if obs_type == "rgb_gripper":
                data[obs_type] = torch.flip(data[obs_type], dims=[2, 3])  # Flip the gripper camera
        return data
    
    def _load_metadata(self, episode):
        if "metadata" in episode:
            return episode["metadata"]
        
        metadata = json.load(open(os.path.join(episode["path"], "metadata.json"), "r"))
        if "frames" not in metadata:
            metadata["frames"] = sorted(os.listdir(os.path.join(episode["path"], self.observation_from[0])))
            metadata["length"] = len(metadata["frames"])
        if self.cache_metadata:
            episode["metadata"] = metadata
        
        return metadata
