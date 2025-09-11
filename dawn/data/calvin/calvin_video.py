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

# import sys
# sys.path.append("/home/nero/Robotics/DAWN/")

from torch.utils.data import Dataset
from dawn.data.base_dataset import BaseDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)

class CalvinVideoDataset(BaseDataset):
    def __init__(self, 
        data_path, 
        split="training", 
        image_size=256,
        num_frames=2,
        num_actions=10,
        min_skip=5, #5,
        max_skip=10, #30,
        observation_type: List[str] = ["rgb_static", "rgb_gripper"],  # Default observation types
        **kwargs
    ):
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

        super().__init__(data_path, split, image_size, num_frames, num_actions, min_skip, max_skip, observation_type)
    

    def get_action(self, episode_metadata, frame_idx):
        metadata = episode_metadata
        # action = torch.tensor(metadata["rel_actions"][frame_idx: frame_idx + self.num_actions])
        action = torch.tensor(metadata["rel_actions"][frame_idx: frame_idx + self.num_actions])
        
        return action

    
    def __getitem__(self, idx):
        # first_frame, episode = self.episodes[idx]
        episode = self.episodes[idx]
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

        frame_idx = frames[0]
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
if __name__ == "__main__":
    dataset = CalvinDataset(
        data_path="/home/nero/Robotics/DAWN/data/realsense", 
        split="training"
    )
    # x = dataset[0]
    from tqdm import tqdm
    # for i in tqdm(range(len(dataset))):
    #     x = dataset[i]
        # print(x["rgb_static"].shape, x["rgb_gripper"].shape)

    from accelerate import Accelerator
    accelerator = Accelerator()
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    from torchvision.models import optical_flow
    import torch.nn.functional as F
    import einops
    # flow_model = optical_flow.raft_large(weights=optical_flow.Raft_Large_Weights.DEFAULT, progress=False).eval()
    # flow_transform = optical_flow.Raft_Large_Weights.DEFAULT.transforms()

    # flow_model, flow_transform, dataloader = accelerator.prepare(flow_model, flow_transform, dataloader)
    dataloader = accelerator.prepare(dataloader)
    for batch in tqdm(dataloader):
        x = batch
        print(x['action'].shape)
        # print(x['action'])
        # continue 
        # print(batch["rgb_static"].shape, batch["rgb_gripper"].shape)
        # image_condition = batch["rgb_static"]
        # flow_input, _ = flow_transform(image_condition, image_condition)
        # print(flow_input.shape)
        # start_im = einops.rearrange(flow_input[:, :-1], "b t c h w -> (b t) c h w")
        # end_im = einops.rearrange(flow_input[:, 1:], "b t c h w -> (b t) c h w")
        # with torch.no_grad():
        #     flow_tensor = flow_model(start_im, end_im, num_flow_updates=6)[-1]
        # print(flow_tensor.shape)
        # image_size = 200
        # scale = 256 / image_size  # Hard Coded.
        # flow_tensor = F.interpolate(flow_tensor, size=(128, 128), mode="bilinear", align_corners=True)
        # flow_tensor = flow_tensor / scale
        # flow_tensor = einops.rearrange(flow_tensor, "(b t) c h w -> b t c h w", b=image_condition.shape[0])
        # print(flow_tensor.shape)

        # flow_dim = flow_tensor.shape[-1]  # Image size is always square.
        # normalized_flow_tensor = (flow_tensor + flow_dim) / (flow_dim * 2)
        # print(normalized_flow_tensor.shape)
        # # if get_all_flow:
        # #     return normalized_flow_tensor, None
        # # clean_flow, prev_flow = normalized_flow_tensor[:, 0], normalized_flow_tensor[:, 0]
        # clean_flow, prev_flow = normalized_flow_tensor[:, 0], normalized_flow_tensor[:, 0]

        # print(f"clean_flow shape: {clean_flow.shape}, prev_flow shape: {prev_flow.shape}")
        # break
        
