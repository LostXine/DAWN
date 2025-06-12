import accelerate
import logging 
import os
import torch
from rich.progress import Progress, SpinnerColumn, BarColumn, MofNCompleteColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, 
        cfg,
        accelerator, 
        model, 
        optimizer, 
        train_loader, 
        val_loader,
        scheduler,
    ):
        self.cfg = cfg
        self.accelerator = accelerator
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_dir = os.path.join(cfg.save_dir, "checkpoints")
        os.makedirs(self.save_dir, exist_ok=True)

        # Prepare the model and optimizer with the accelerator
        logger.info(f"Preparing model and optimizer with {self.accelerator.__class__.__name__}.")
        self.model, self.optimizer, self.scheduler, self.train_loader, self.val_loader = self.accelerator.prepare(
            self.model, 
            self.optimizer, 
            self.scheduler,
            self.train_loader, 
            self.val_loader
        )
        logger.info(f"Model and optimizer prepared. Model: {self.model.__class__.__name__}, Optimizer: {self.optimizer.__class__.__name__}")

        self.progress = Progress(
            TextColumn("{task.description}"),
            SpinnerColumn(),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            disable=not self.accelerator.is_main_process,
        ) 
        self.progress.start()
    def load_checkpoint(self, checkpoint_path):
        """
        Load a model checkpoint.
        Args:
            checkpoint_path (str): Path to the checkpoint file.
        """
        self.accelerator.load_state(checkpoint_path, self.model, self.optimizer)

    def save_checkpoint(self, k=5):
        """
        Save the model and optimizer state to a checkpoint.
        Args:
            checkpoint_path (str): Path to save the checkpoint file.
            k: Keep last k checkpoints
        """
        # ckpt_lst = sorted()
        
        ckpt_lst = sorted(os.listdir(self.save_dir))
        while len(ckpt_lst) >= k:
            logger.info(f"Removing old checkpoint: {ckpt_lst[0]}")
            os.remove(os.path.join(self.save_dir, ckpt_lst[0]))
            ckpt_lst = ckpt_lst[1:]
        
        ckpt = os.path.join(self.save_dir, f"model_{self.cur_step:07d}.pth")
        torch.save(self.model.module.state_dict(), ckpt)
        logger.info(f"Saved checkpoint to {ckpt}")
        
    def train(self):
        self.model.train()
        self.cur_step = 0        
        logger.info(f"Starting training at step {self.cur_step} for {self.cfg.total_steps} steps.")
        train_task = self.progress.add_task(
            "[bold blue]Training...", 
            total=self.cfg.total_steps, 
            completed=self.cur_step,
        )

        while self.cur_step < self.cfg.total_steps:
            for batch in self.train_loader:
                # Forward pass
                with self.accelerator.accumulate(self.model):
                    outputs = self.model(batch)

                    # Compute loss
                    loss = outputs["total_loss"]
                    
                    # Backward pass
                    self.accelerator.backward(loss)

                    # Update parameters
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()  # Update learning rate

                self.cur_step += 1
                self.progress.update(train_task, advance=1)
                
                # Log 
                if self.accelerator.is_main_process and self.cur_step % self.cfg.log_interval == 0:
                    current_lr = self.scheduler.get_last_lr()[0]

                    self.accelerator.log({
                        "loss": loss.item(),
                        "step": self.cur_step,
                        "lr": current_lr,
                    })
                    logger.info(f"Step {self.cur_step}, Loss: {loss.item():.7f}, LR: {current_lr:.7f}")
                
                # Save checkpoint
                if self.accelerator.is_main_process and self.cur_step % self.cfg.save_interval == 0:
                    self.save_checkpoint()
                    
                # Validation step
                if self.cur_step % self.cfg.val_interval == 0:
                    self.validate()

                if self.cur_step >= self.cfg.total_steps:
                    break

    
    def validate(self):
        self.model.eval()

        val_task = self.progress.add_task(
            "[bold green]Validating...", 
            total=len(self.val_loader)
        )
        
        total_loss = 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                outputs = self.model(batch)
                loss = outputs["total_loss"]
                total_loss += loss.item()

                self.progress.update(val_task, advance=1)
                # break 
        avg_val_loss = total_loss / len(self.val_loader)
        if self.accelerator.is_main_process:
            images=None
            images = self.model.module.visualize(batch, outputs)
            logger.info(f"Validation Loss: {avg_val_loss:.7f}")
            self.accelerator.log({
                "val_loss": avg_val_loss,
                "step": self.cur_step,
                "images": images if images is not None else None,
            })
        self.model.train()
        self.progress.remove_task(val_task)
        