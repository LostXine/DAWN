import torch
from torch import nn
import torch.nn.funtional as F

from transformers import CLIPVisualEncoder 
from diffusers import DDPMScheduler, DDIMScheduler

class LatentDiffusionPolicy(nn.Module):
    def __init__(self, num_action_steps=10, action_dim=7, in_channels=6, latent_dim=128, num_heads=8, num_layers=3):
        super()().__init__()
        logger.info(f"Initializing {__class__.__name__} with num_action_steps {num_action_steps}, action_dim {action_dim}, in_channels {in_channels}, latent_dim {latent_dim}, num_heads {num_heads}, num_layers {num_layers}.")
        self.num_action_steps = num_action_steps
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.scheduler = DDIMScheduler(num_train_timesteps=1000,)
        
        model_config = transformers.ViTConfig(
            hidden_size=hidden_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=mlp_dim,
            image_size=image_size,
            patch_size=patch_size,
            num_channels=in_channels,
        )
        self.visual_model = transformers.ViTModel(model_config)
        # self.action_enc = nn.Linear(action_dim, latent_dim)
        # self.action_dec = nn.Linear(latent_dim, action_dim)

        # self.denoising_model = LatentTransformerDenoisingModel(
        #     num_queries=num_action_steps,
        #     latent_dim=latent_dim,
        #     num_heads=num_heads, num_layers=num_layers, cond_dim=cond_dim
        # )

        self.criterion = torch.nn.functional.mse_loss  # Assuming MSE loss for action classification
    
    def forward(self, x, labels=None):
        # Forward pass through the ViT model
        visual_condition = self.visual_model(x).last_hidden_state

        bs = visual_condition.shape[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=visual_condition.device, dtype=torch.int64
        )

        if labels is not None:
            # If labels are provided, compute the loss
            loss = self.criterion(outputs, labels.flatten(start_dim=1))

            return_dict["loss"] = loss

        return return_dict

if __name__ == "__main__":
    