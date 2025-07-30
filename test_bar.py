# simulate_progress.py
import torch
import accelerate
import time
import os

if __name__ == "__main__":
    accelerator = accelerate.Accelerator()
    device = accelerator.device


    total = 0
    for i in range(10):
        accelerate.utils.set_seed(0)
        r = torch.rand(1000, 1000, device=device)
        total = (r ** 2).sum().item()
        print(f"{i}: Device: {device}, Total: {total}")