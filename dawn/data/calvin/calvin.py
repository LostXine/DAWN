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

import sys
sys.path.append("/home/nero/Robotics/DAWN/")

from torch.utils.data import Dataset
from dawn.data.base_dataset import BaseDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)

class CalvinDataset(BaseDataset):
    def __init__(self, 
        data_path, 
        split="training", 
        image_size=256,
        num_frames=2,
        num_actions=10,
        min_skip=10, #5,
        max_skip=30, #30,
        observation_type: List[str] = ["rgb_static", "rgb_gripper"],  # Default observation types
        action_type="rel_actions",
        **kwargs
    ):
        # self.data_path = os.path.join(data_path, "episodes")
        self.data_path = os.path.join(data_path, split, "episodes")
        
        # Load task + language annotations
        try:
            self.annos = OmegaConf.load(os.path.join(data_path, "annotations.yaml"))
            self.r_map = {}
            for k, v in self.annos.items():
                for x in v:
                    self.r_map[x] = k
                self.annos[k].append(k.replace("_", " "))
        except:
            logger.warning("No annotations found, using default language annotations.")

        super().__init__(data_path, split, image_size, num_frames, num_actions, min_skip, max_skip, observation_type, action_type=action_type)    
        self.robot_obs_min, self.robot_obs_max, self.robot_obs_range = self._normalize_robot_state()

    def _normalize_robot_state(self):
        #normalize robot state by the global min and max
        robot_obs = []
        for episode in self.episodes:
            epi_robot_state = self._load_metadata(episode)["robot_obs"]
            robot_obs.extend(epi_robot_state)
        robot_obs = torch.tensor(robot_obs, dtype=torch.float32)
        #compute min-max for each dim in robot state (N, 15)
        robot_obs_min = robot_obs.min(dim=0)[0] # Shape (15,)
        robot_obs_max = robot_obs.max(dim=0)[0] # Shape (15,)
        robot_obs_range = robot_obs_max - robot_obs_min + 1e-8 # Shape (15,)
        logger.info(f"Robot state min: {robot_obs_min}, max: {robot_obs_max}, range: {robot_obs_range}")
        return robot_obs_min, robot_obs_max, robot_obs_range


    def get_action(self, episode_metadata, frame_idx):
        metadata = episode_metadata
        # action = torch.tensor(metadata["rel_actions"][frame_idx: frame_idx + self.num_actions])
        action = torch.tensor(metadata[self.action_type][frame_idx: frame_idx + self.num_actions])
        return action
    
    def get_robot_state(self, episode_metadata, frame_idx):
        metadata = episode_metadata
        robot_obs = torch.tensor(metadata["robot_obs"][frame_idx: frame_idx + self.num_actions], dtype=torch.float32)
        robot_obs = (robot_obs[:, 6:-1] - self.robot_obs_min[6:-1]) / self.robot_obs_range[6:-1] #Shape: (T, 8)
        if robot_obs.ndim == 1:
            robot_obs = robot_obs.unsqueeze(0)
        
        return robot_obs
        
    
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
        data["robot_obs"] = self.get_robot_state(metadata, frame_idx)

        # Pad actions if they are less than num_actions 
        if len(data["action"]) < self.num_actions:
            pad_length = self.num_actions - len(data["action"])
            last_action = data["action"][-1:]
            data["action"] = torch.cat([data["action"], last_action.repeat(pad_length, 1)], dim=0)
        
        # Pad actions if they are less than num_actions 
        if len(data["robot_obs"]) < self.num_actions:
            pad_length = self.num_actions - len(data["robot_obs"])
            last_action = data["robot_obs"][-1:]
            data["robot_obs"] = torch.cat([data["robot_obs"], last_action.repeat(pad_length, 1)], dim=0)
        

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
            
        return data
