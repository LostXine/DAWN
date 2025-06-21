from transformers import AutoModel
import torch
from torch import nn
import einops 

from diffusers import DDPMScheduler, DDIMScheduler
import torchvision 

class DiffusionPolicy(nn.Module):
    def __init__(self, 
        visual_model_name='resnet50', 
        in_channels=8, 
        num_action_steps=10, 
        action_dims=7,
    ):
        super().__init__()
        self.visual_encoder = AutoModel.from_pretrained(visual_model_name)
        if 'resnet' in visual_model_name:
            self.visual_encoder.embedder.embedder.convolution = nn.Conv2d(
                in_channels, 
                self.visual_encoder.embedder.embedder.convolution.out_channels, 
                kernel_size=7, 
                stride=2, 
                padding=3, 
                bias=False
            )
        elif 'convnext' in visual_model_name:
            self.visual_encoder.embeddings.patch_embeddings = nn.Conv2d(
                in_channels, 
                self.visual_encoder.embeddings.patch_embeddings.out_channels, 
                kernel_size=7, 
                stride=4 
            )
            self.visual_encoder.embeddings.num_channels = in_channels
        else:
            raise ValueError(f"Unsupported model type: {model_name}")
        self.in_channels = in_channels
        self.num_action_steps = num_action_steps
        self.action_dims = action_dims

        # Initialize the scheduler
        self.scheduler = DDIMScheduler(num_train_timesteps=1000)

    def forward(self, x, labels=None):
        # Forward pass through the visual encoder
        
        if not self.training:
            return self.forward_eval(x, labels)

        image_condition = self.visual_encoder(x).last_hidden_state
        image_condition = einops.rearrange(image_condition, 'b c h w -> b (h w) c')  # Rearrange to (batch_size, channels, time)

        bs = image_condition.shape[0]

        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps, (bs,), device=image_condition.device, dtype=torch.int64
        )

        noise = torch.randn((bs, self.num_action_steps, self.action_dims), device=timesteps.device)
        noise_action = self.scheduler.add_noise(labels, noise, timesteps)

        model_inputs = noise_action

        noise_pred = self.model(model_inputs, timesteps, image_condition, return_dict=False)[0]
        
        loss = F.mse_loss(noise_pred, noise)

        outputs = {
            "total_loss": loss,
        }

        return outputs