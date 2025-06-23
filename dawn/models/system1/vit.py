import torch
from torch import nn
import transformers
from torchvision.models import vision_transformer
import logging

logger = logging.getLogger(__name__)

class ViTAction(nn.Module):
    def __init__(self, image_size=128, patch_size=16, in_channels=5, action_space=7, num_actions=10, hidden_dim=192, num_layers=12, num_heads=3, mlp_dim=768):
        super().__init__()
        logger.info(f"Initializing {__class__.__name__}.")
        
        model_config = transformers.ViTConfig(
            hidden_size=hidden_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=mlp_dim,
            image_size=image_size,
            patch_size=patch_size,
            num_channels=in_channels,
            num_labels=action_space * num_actions,  # Assuming action space is a vector of size action_space * num_actions
        )
        self.num_actions = num_actions
        self.model = transformers.ViTForImageClassification(model_config)
        self.criterion = torch.nn.functional.mse_loss  # Assuming MSE loss for action classification
    
    def forward(self, x, labels=None):
        # Forward pass through the ViT model
        x = x["visual_input"]
        outputs = self.model(x).logits
        return_dict = { "logits": outputs.view(outputs.size(0), self.num_actions, -1) }

        if labels is not None:
            # If labels are provided, compute the loss
            loss = self.criterion(outputs, labels.flatten(start_dim=1))
            return_dict["loss"] = loss

        return return_dict

if __name__ == "__main__":
    vit_tiny = ViTAction()
    vit_tiny.eval()

    # Test with a variable image size (e.g., 128x128)
    img_size = 128
    input_channels = 5
    dummy_input = torch.randn(16, 5, img_size, img_size)  # 1 image, 5 channels, 128x128
    # output = vit_tiny(dummy_input)

    action_gt = torch.randn(16, 7)  # Assuming 7 action dimensions
    output = vit_tiny(dummy_input, labels=action_gt)

    print(output["logits"].shape)  # Default output [1, 7] (for action vectors)
    if "loss" in output:
        print("Loss:", output["loss"].item())
    else:
        print("No loss computed.")