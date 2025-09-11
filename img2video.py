import os
from functools import partial

from tqdm.auto import tqdm
import numpy as np
from PIL import Image

def as_gif(images, path="temp.gif", duration=100):
    # Render the images as the gif:
    images[0].save(path, save_all=True, append_images=images[1:], duration=duration, loop=0)
    gif_bytes = open(path, "rb").read()
    return gif_bytes

data_path = 'outputs/flow_visualization/realsense_ft33_skip5'
output = 'outputs/flow_visualization/realsense_ft33_skip5_gif'
os.makedirs(output, exist_ok=True)

eps = sorted(os.listdir(data_path))
duration = 1000
for ep in tqdm(eps):
    ep_path = os.path.join(data_path, ep)
    if not os.path.isdir(ep_path):
        continue
    images = []
    for img_name in sorted(os.listdir(ep_path)):
        img_path = os.path.join(ep_path, img_name)
        if not img_name.endswith('.png'):
            continue
        img = Image.open(img_path).convert("RGB")
        images.append(img)
    
    path = os.path.join(output, f'{ep}.gif')
    images[0].save(path, save_all=True, append_images=images[1:], duration=duration, loop=0)
    