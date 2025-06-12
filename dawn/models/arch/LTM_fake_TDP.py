import torch
import logging
from humanfriendly import format_size
logger = logging.getLogger(__name__)

class ImagineToAct(torch.nn.Module):
    def __init__(self, imagine_model=None, action_model=None):
        super().__init__()

        logger.info(f"Initializing {__class__.__name__}.")
        self.imagine_model = imagine_model
        self.action_model = action_model
        
        total_params = sum(p.numel() for p in self.parameters())
        total_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {format_size(total_params)}, Trainable parameters: {format_size(total_trainable_params)}")

    def forward(self, batch_data):
        # Imagine  
        imagined_output = self.imagine_model(batch_data)
        
        # Action 
        loss = self.action_model(
            x=imagined_output,
            labels=batch_data.get("action", None)
        )

        return {
            "total_loss": loss["loss"],
        }
