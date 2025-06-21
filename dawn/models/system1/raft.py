import torch
from torch import nn
from torchvision.models import optical_flow
import torch.nn.functional as F

from torchvision.utils import flow_to_image
import logging 
import einops

logger = logging.getLogger(__name__)

class RAFT(nn.Module):
    def __init__(self, flow_to_rgb=True):
        super().__init__()
        logger.info(f"Initializing {__class__.__name__}.")
        # Initialize the RAFT model
        self.flow_model = optical_flow.raft_large(weights=optical_flow.Raft_Large_Weights.DEFAULT, progress=False).eval()
        self.flow_transform = optical_flow.Raft_Large_Weights.DEFAULT.transforms()

        self.flow_to_rgb = flow_to_rgb

        mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)

        self.register_buffer('mean', mean, False)
        self.register_buffer('std', std, False)

        for param in self.flow_model.parameters():
            param.requires_grad = False

    def gen_flow(self, images):
        print(images.min(), images.max(), images.shape)
        if images.shape[1] < 2:
            images = torch.cat([images, images], dim=1)  # Duplicate the first frame if only one frame is provided.
        flow_input = images
        # flow_input, _ = self.flow_transform(images, images)
        start_im = einops.rearrange(flow_input[:, :-1], "b t c h w -> (b t) c h w")
        end_im = einops.rearrange(flow_input[:, 1:], "b t c h w -> (b t) c h w")
        print(start_im.min(), start_im.max(), start_im.shape)
        with torch.no_grad():
            flow_tensor = self.flow_model(start_im, end_im, num_flow_updates=6)[-1]
            
        if self.flow_to_rgb:
            flow_tensor = flow_to_image(flow_tensor)
            # flow_tensor = (flow_tensor - self.mean) / self.std
        else:
            flow_dim = flow_tensor.shape[-1]  # Image size is always square.
            flow_tensor = (flow_tensor + flow_dim) / (flow_dim * 2)

        flow_tensor = einops.rearrange(flow_tensor, "(b t) c h w -> b t c h w", b=flow_input.shape[0])
        # normalize the flow
        return flow_tensor[:, 0]
    
        
        # flow_dim = flow_tensor.shape[-1]  # Image size is always square.
        # normalized_flow_tensor = (flow_tensor + flow_dim) / (flow_dim * 2)
        # return normalized_flow_tensor[:, 0]  # Return only the first frame's flow.
    
    @torch.no_grad()
    def forward(self, batch_data):
        rgb_flow = self.gen_flow(batch_data["rgb_static"])
        norm_rgb = (batch_data["rgb_static"][:, 0] - self.mean) / self.std
        # output = torch.cat([norm_rgb, rgb_flow], dim=1)
        output = rgb_flow
        return output


if __name__ == "__main__":
    raft_model = RAFT().cuda()
    raft_model.eval()

    # Test with a variable image size (e.g., 128x128)
    img_size = 128
    input_channels = 3
    dummy_input = torch.randn(8, 2, 3, img_size, img_size).cuda()  # Batch of 8 images, 3 channels, 128x128

    # Generate optical flow
    output = raft_model({"rgb_static": dummy_input})

    print(output.shape)  # Expected output shape: [8, 5, img_size, img_size]