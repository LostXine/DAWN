import os
import sys
import torch
import accelerate 
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from diffusers.optimization import get_cosine_schedule_with_warmup

from utils.logging import setup_logging
from torch import optim

from dawn.trainer import Trainer

def main(cfg: DictConfig = None):
    # Initialize the accelerator
    accelerator = accelerate.Accelerator(**cfg.accelerator)

    accelerator.init_trackers(
        project_name=cfg.project,
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    accelerate.utils.set_seed(cfg.seed)

    setup_logging(accelerator.is_main_process, log_dir=cfg.trainer.save_dir)

    logger = logging.getLogger(__name__)
    logger.info("Configuration:\n" + OmegaConf.to_yaml(cfg))

    # # Init dataset
    train_dataset = hydra.utils.instantiate(cfg.dataset.train)
    val_dataset = hydra.utils.instantiate(cfg.dataset.val)

    # Init dataloader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.loader.train_batch_size, 
        shuffle=True, 
        num_workers=cfg.loader.num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.loader.val_batch_size,
        shuffle=False,
        num_workers=cfg.loader.num_workers
    )

    model = hydra.utils.instantiate(cfg.model)
    
    if cfg.weights is not None:
        logger.info(f"Loading model weights from {cfg.weights}.")
        # Load the model weights
        if os.path.exists(cfg.weights):
            logger.info(model.load_state_dict(torch.load(cfg.weights, map_location="cpu"), strict=False))
        else:
            logger.warning(f"Weights file {cfg.weights} does not exist. Skipping loading weights.")
        
    # Optimizer
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    # Scheduler

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=(cfg.trainer.lr_warmup_steps * accelerator.num_processes),
        num_training_steps=(cfg.trainer.total_steps * accelerator.num_processes),
    )

    trainer = Trainer(
        cfg=cfg.trainer,
        accelerator=accelerator,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=lr_scheduler,
    )

    # Start training
    trainer.train()

if __name__ == "__main__":
    with hydra.initialize(config_path="configs"):
        argv = sys.argv[1:]
        print(argv)
        if len(argv) > 0 and argv[0].startswith("config="):
            config_name = argv[0].split("=")[1]
            argv = argv[1:]
        else:
            config_name = "default"
        cfg = hydra.compose(config_name=config_name, overrides=argv)
        OmegaConf.resolve(cfg)
        main(cfg)
