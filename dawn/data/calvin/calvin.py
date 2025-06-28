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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)

class CalvinDataset(Dataset):
    def __init__(self, 
        data_path, 
        split="training", 
        image_size=256,
        num_frames=2,
        num_actions=10,
        min_skip=5,
        max_skip=30,
        observation_type: List[str] = ["rgb_static", "rgb_gripper"]  # Default observation types
    ):
        timer = Timer()

        self.data_path = os.path.join(data_path, split, "episodes")
        self.observation_type = observation_type
        self.image_size = image_size
        self.num_frames = num_frames
        self.num_actions = num_actions
        self.min_skip = min_skip
        self.max_skip = max_skip

        self.split = split

        if not os.path.exists(self.data_path):
            raise ValueError(f"Data path {self.data_path} does not exist.")

        
        # Load task + language annotations
        self.annos = OmegaConf.load(os.path.join(data_path, "annotations.yaml"))
        self.r_map = {}
        for k, v in self.annos.items():
            for x in v:
                self.r_map[x] = k
            self.annos[k].append(k.replace("_", " "))

        self.episodes = self._load_episodes()
        self.transform = self._get_transform()

        logger.info(f"Loading from {data_path} takes {timer.seconds():.2f} seconds.")

    def _get_transform(self):
        if self.split == "training":
            return A.Compose([
                A.Affine(scale=(0.8, 1.2), rotate=(-15, 15),
                    translate_percent=(-0.1, 0.1), shear=(-10, 10), p=0.5),
                A.Resize(self.image_size, self.image_size),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                # A.OneOf([
                #     A.GaussNoise(std_limit=(0.1, 0.2), p=0.5),
                #     A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5),
                #     A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True, p=0.5),
                #     A.SaltAndPepper(p=0.5)
                # ], p=0.3),

                ToTensorV2(),
            ])
        return A.Compose([
            A.Resize(self.image_size, self.image_size),
            ToTensorV2(),
        ])

    def _load_episodes(self):
        
        # Load episodes
        episodes = []
        total_frames = 0
        for episode_file in sorted(os.listdir(self.data_path)):
            episode = {
                "idx": episode_file,
                "path": os.path.join(self.data_path, episode_file),
            }
            metadata = json.load(open(os.path.join(episode["path"], "metadata.json"), "r"))
            episode["metadata"] = metadata
            episode["metadata"]["task"] = self.r_map[metadata["language"]]
            episodes.append(episode)

        logger.info(f"Loaded {len(episodes)} episodes from {self.data_path}") 
        logger.info(f"Loaded {total_frames} frames in total from {self.data_path}")      
        return episodes

    def __len__(self):
        return len(self.episodes)
    
    def __getitem__(self, idx):
        # idx=0
        # idx = 0
        episode = self.episodes[idx]
        metadata = episode["metadata"]

        # Augment the language in the same task
        language = random.choice(self.annos[metadata["task"]])
        data = {
            "idx": episode["idx"],
            "language": language # metadata["language"],
            # "language_embedding": torch.tensor(metadata["language_embedding"], dtype=torch.float32),
        }
        frame_idx = random.choice(range(metadata["length"]))
        # frame_idx = 0
        data["action"] = torch.tensor(metadata["rel_actions"][frame_idx: frame_idx + self.num_actions])
        # Pad actions if they are less than num_actions
        if len(data["action"]) < self.num_actions:
            pad_length = self.num_actions - len(data["action"])
            last_action = data["action"][-1:]
            data["action"] = torch.cat([data["action"], last_action.repeat(pad_length, 1)], dim=0)
            # data["action"] = torch.cat([data["action"], torch.zeros((pad_length, data["action"].shape[1]), dtype=torch.float32)], dim=0)
        
        # next_idx = min(metadata["length"] - 1, frame_idx + self.num_actions)
        skip = random.randint(self.min_skip, self.max_skip)
        next_idx = min(metadata["length"] - 1, frame_idx + skip)
        skip = next_idx - frame_idx
        frame_idx = [frame_idx, next_idx]
        data["skip_frame"] = torch.tensor(skip, dtype=torch.int64)

        for obs_type in self.observation_type:
            obs_files = sorted(glob.glob(os.path.join(episode["path"], obs_type, '*.jpg')))
            images = np.stack([imageio.imread(obs_files[idx]) for idx in frame_idx], axis=0)
            data[obs_type] = self.transform(images=images)["images"] / 255.
            # data[obs_type] = torch.stack([self.transform(image=)["image"] ]) / 255. 
        return data

if __name__ == "__main__":
    dataset = CalvinDataset(
        data_path="/home/nero/Robotics/DAWN/data/calvin/dataset_opt/task_ABC_D", 
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
        print(x['action'])
        continue 
        print(batch["rgb_static"].shape, batch["rgb_gripper"].shape)
        image_condition = batch["rgb_static"]
        flow_input, _ = flow_transform(image_condition, image_condition)
        print(flow_input.shape)
        start_im = einops.rearrange(flow_input[:, :-1], "b t c h w -> (b t) c h w")
        end_im = einops.rearrange(flow_input[:, 1:], "b t c h w -> (b t) c h w")
        with torch.no_grad():
            flow_tensor = flow_model(start_im, end_im, num_flow_updates=6)[-1]
        print(flow_tensor.shape)
        image_size = 200
        scale = 256 / image_size  # Hard Coded.
        flow_tensor = F.interpolate(flow_tensor, size=(128, 128), mode="bilinear", align_corners=True)
        flow_tensor = flow_tensor / scale
        flow_tensor = einops.rearrange(flow_tensor, "(b t) c h w -> b t c h w", b=image_condition.shape[0])
        print(flow_tensor.shape)

        flow_dim = flow_tensor.shape[-1]  # Image size is always square.
        normalized_flow_tensor = (flow_tensor + flow_dim) / (flow_dim * 2)
        print(normalized_flow_tensor.shape)
        # if get_all_flow:
        #     return normalized_flow_tensor, None
        # clean_flow, prev_flow = normalized_flow_tensor[:, 0], normalized_flow_tensor[:, 0]
        clean_flow, prev_flow = normalized_flow_tensor[:, 0], normalized_flow_tensor[:, 0]

        print(f"clean_flow shape: {clean_flow.shape}, prev_flow shape: {prev_flow.shape}")
        break
        
