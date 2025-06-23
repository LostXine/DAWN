import torch
from torch import nn
import transformers
from transformers import AutoModel, AutoProcessor
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class TransformerAction(nn.Module):
    def __init__(
        self, 
        image_size=128, 
        in_channels=5, 
        action_space=7, 
        num_actions=10, 
        hidden_dim=768, 
        num_layers=6, 
        num_heads=8,
        mlp_dim=2048,
    ):
        super().__init__()
        logger.info(f"Initializing {__class__.__name__}.")
        
        # Initialize the ViT feature extractor
        self.feature_extractor = AutoModel.from_pretrained("google/vit-base-patch16-224-in21k", add_pooling_layer=False)
        patch_embed = self.feature_extractor.embeddings.patch_embeddings.projection 
        self.feature_extractor.embeddings.patch_embeddings.projection = torch.nn.Conv2d(in_channels, patch_embed.out_channels, kernel_size=patch_embed.kernel_size, stride=patch_embed.stride, padding=patch_embed.padding)
        self.feature_extractor.embeddings.patch_embeddings.num_channels = in_channels

        # Initialize the action decoder
        self.query_embed = nn.Embedding(num_actions, hidden_dim)
        # self.query_pos = nn.Embedding(num_actions, hidden_dim)
        self.action_decoder = MLP(hidden_dim, mlp_dim, action_space, 3)

        trans_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=0.1,
            activation='relu',
            batch_first=True,
        )
        self.model = nn.TransformerDecoder(trans_layer, 3)

        self.num_actions = num_actions
        self.criterion = torch.nn.functional.mse_loss  # Assuming MSE loss for action classification
    
    def forward(self, x, labels=None):
        # Forward pass through the ViT model
        x = x["visual_input"]
        b, c, h, w = x.shape
        inp = F.interpolate(x, size=(self.feature_extractor.config.image_size, self.feature_extractor.config.image_size), mode='bilinear', align_corners=False)
        visual_feat = self.feature_extractor(inp).last_hidden_state  # [B, C, H, W] -> [B, N, C]
        query_embed = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)  # [B, num_actions, hidden_dim]
        # query_pos = self.query_pos.weight.unsqueeze(0).expand(b, -1, -1)  # [B, num_actions, hidden_dim]
        
        action_embedding = self.model(tgt=query_embed, memory=visual_feat)
        outputs = self.action_decoder(action_embedding)  # [B, num_actions, action_space * 3]
        # print(outputs.shape)
        return_dict = { "logits": outputs}

        if labels is not None:
            # If labels are provided, compute the loss
            loss = self.criterion(outputs.flatten(start_dim=1), labels.flatten(start_dim=1))
            return_dict["loss"] = loss

        return return_dict

if __name__ == "__main__":
    vit_tiny = TransformerAction()
    vit_tiny.eval()

    # Test with a variable image size (e.g., 128x128)
    img_size = 256
    input_channels = 5
    dummy_input = torch.randn(16, 5, img_size, img_size)  # 1 image, 5 channels, 128x128
    # output = vit_tiny(dummy_input)

    action_gt = torch.randn(16, 10, 7)  # Assuming 7 action dimensions
    output = vit_tiny(dummy_input, labels=action_gt)

    print(output["logits"].shape)  # Default output [1, 7] (for action vectors)
    if "loss" in output:
        print("Loss:", output["loss"].item())
    else:
        print("No loss computed.")