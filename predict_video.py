import sys
sys.path.append('../')  # Adjust path to import from the parent directory if needed

import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import os
import tempfile

import hydra 

import logging
import accelerate
import torch
import numpy as np
import json
from dawn.models.system2.flow_utils import visualize_flow_vectors_as_PIL, FlowNormalizer
        
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


config = {
    "_target_" : "dawn.models.system2.ldm_support.LatentMotionEstimation",
    "flow_to_rgb" : False,
    "use_cfg" : False,
    "use_interval" : True,
    "support_types": ["rgb_gripper"],
    "input_type": "rgb_static",
    "use_text_sentence": False,
}

accelerator = accelerate.Accelerator()
accelerate.utils.set_seed(0)

weights = "outputs/DAWN_stage_1/2025-07-24_07-34/checkpoints/model_0042000.pth"
print(f"Loading model configuration from {config}")
model = hydra.utils.instantiate(config)

if os.path.exists(weights):
    logger.info(f"Loading model weights from {weights}.")
    logger.info(model.load_state_dict(torch.load(weights, map_location="cpu"), strict=False))
else:
    logger.warning(f"Model weights file not found: {weights}. Skipping loading weights.")

model = accelerator.prepare(model)
model.eval()

normalizer = FlowNormalizer(256, 256)

import glob
from tqdm.auto import tqdm
data_path = "data/realworld/val"
output_path = "outputs/flow_visualization/realworld_zero_shot_skip5"

eps = sorted(os.listdir(data_path))
for ep in eps:
    ep_path = os.path.join(data_path, ep)
    if not os.path.isdir(ep_path):
        continue
    
    os.makedirs(os.path.join(output_path, ep), exist_ok=True)
    length = len(glob.glob(os.path.join(ep_path, "rgb_static", "*")))
    language = json.load(open(os.path.join(ep_path, "episode_metadata.json"), "r"))["language"]
    print(f"Processing episode {ep} with length {length} and language: {language}")
    for i in tqdm(range(0, length, 5)):
        rgb = Image.open(os.path.join(ep_path, "rgb_static", f"{i:06d}.png")).convert("RGB").resize((256, 256))
        gripper = Image.open(os.path.join(ep_path, "rgb_gripper", f"{i:06d}.png")).convert("RGB").resize((256, 256))

        rgb_ = torch.from_numpy(np.array(rgb)).permute(2, 0, 1).float() / 255.0
        gripper_ = torch.from_numpy(np.array(gripper)).permute(2, 0, 1).float() / 255.0
        skip = torch.tensor(20)
        c, h, w = rgb_.shape

        input_data = {
            "rgb_static": rgb_.view(1, 1, c, h, w).repeat(1, 3, 1, 1, 1).to(model.device),
            "rgb_gripper": gripper_.view(1, 1, c, h, w).repeat(1, 3, 1, 1, 1).to(model.device),
            "skip_frame": skip.view(-1).to(model.device),
            "language": [language],
        }
        with torch.no_grad():
            output = model(input_data)
            # print(output['generated_flow'].shape)
            pd = output['generated_flow']
            pd_flow = normalizer.unnormalize(pd[0].permute(1, 2, 0).cpu().numpy())  # Convert to (H, W, C) format and unnormalize
            # print(pd_flow.min(), pd_flow.max())
            # pd_flow[abs(pd_flow) < 1] = 0
            pd = visualize_flow_vectors_as_PIL(rgb, pd_flow, step=8, title=language)
            pd.save(os.path.join(output_path, ep, f"{i:06d}.png"))
        # break

