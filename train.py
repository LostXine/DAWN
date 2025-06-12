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
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.syntax import Syntax

from dawn.trainer import Trainer

# @hydra.main(config_path="configs", config_name="default")
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
        batch_size=cfg.loader.batch_size, 
        shuffle=True, 
        num_workers=cfg.loader.num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.loader.batch_size,
        shuffle=False,
        num_workers=cfg.loader.num_workers
    )

    model = hydra.utils.instantiate(cfg.model)

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
    # cfg = OmegaConf.load("configs/stage1.yaml")  # Load your configuration file

    with hydra.initialize(config_path="configs"):
        cfg = hydra.compose(config_name="stage1", overrides=sys.argv[1:])
        OmegaConf.resolve(cfg)
        main(cfg)

    # try:
    #     x = 1 / 0
    # except Exception:
    #     logger.exception("An exception occurred! Rich will format this traceback.")

    # logger.info("Logging complete. Check 'app_ansi.log'.")
    # from tqdm import tqdm
    # from tqdm.contrib.logging import logging_redirect_tqdm # Ensure this import
    # from accelerate import Accelerator
    # accelerator = Accelerator()
    # setup_logging(accelerator.is_main_process)  

    # logger = logging.getLogger(__name__)

    # logger.info("Starting training...")
    # logger.info("This is an informational message.")
    # logger.warning("This is a warning. Something might be wrong.")
    # logger.error("This is an error. Something is definitely wrong.")
    # # progress = tqdm(range(10000), disable=not accelerator.is_main_process, desc=f"Epoch")
    # import time
    # from rich.progress import Progress

    # with Progress(disable=not accelerator.is_main_process) as progress:
    #     task1 = progress.add_task("[cyan]Task 1", total=100)
    #     task2 = progress.add_task("[magenta]Task 2", total=100)
    #     for i in range(100):
    #         progress.update(task1, advance=1)
    #         progress.update(task2, advance=2)
    #         # if i % 10 == 0:
    #             # logger.info(f"Completed {i} iterations.")
    #         time.sleep(0.1)

    # for i in range(10):
    #     logger.info("Training complete.")    


    # accelerator.wait_for_everyone()
    # logger.info("End!")