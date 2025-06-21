import logging
import torch
from torch import nn
import torch.nn.functional as F
import diffusers
from diffusers import DDPMScheduler, DDIMScheduler
import einops
from torchvision.models import optical_flow
from torchvision.utils import flow_to_image
# from transformers import CLIPTextModel, AutoTokenizer
# from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class MotionEstimation(nn.Module):
    def __init__(self, 
            pretrained: str = None,
            image_size=128, 
            in_channels=6, 
            out_channels=3, 
            condition_dim=768, 
            size="B", 
            flow_to_rgb=True, 
            lang_model='use'
        ):
        super().__init__()
        logger.info(f"Initializing {__class__.__name__} with image size {image_size}, in_channels {in_channels}, out_channels {out_channels}, condition_dim {condition_dim}, size {size}, flow_to_rgb {flow_to_rgb}.")
        self.flow_model = optical_flow.raft_large(weights=optical_flow.Raft_Large_Weights.DEFAULT, progress=False).eval()
        self.flow_transform = optical_flow.Raft_Large_Weights.DEFAULT.transforms()
        self.flow_to_rgb = flow_to_rgb
        self.image_size = image_size

        mean = torch.tensor([0, 0, 0]).view(1, 3, 1, 1)
        std = torch.tensor([255, 255, 255]).view(1, 3, 1, 1)

        self.register_buffer('mean', mean, False)
        self.register_buffer('std', std, False)

        for param in self.flow_model.parameters():
            param.requires_grad = False

        if size == "B":
            self.model = diffusers.UNet2DConditionModel(
            sample_size=image_size,  # Target image resolution
            in_channels=in_channels,  # Number of input channels, e.g., RGB image + flow
            out_channels=out_channels,  # Number of output channels, e.g., flow image
            layers_per_block=2,  # Number of ResNet layers per UNet block
            block_out_channels=(128, 128, 256, 256, 512, 512),  # Channels for each UNet block
            down_block_types=(
                "DownBlock2D",  # Regular ResNet downsampling block
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",  # Downsampling block with attention
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",  # Regular ResNet upsampling block
                "AttnUpBlock2D",  # Upsampling block with attention
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
            cross_attention_dim=condition_dim,  # Dimension of the conditional embedding (e.g., text embeddings)
        )
        else:
            self.model = diffusers.UNet2DConditionModel(
                sample_size=image_size,  # Target image resolution
                in_channels=in_channels,  # Number of input channels, e.g., RGB image + flow
                out_channels=out_channels,  # Number of output channels, e.g., flow image
                layers_per_block=2,  # Number of ResNet layers per UNet block
                block_out_channels=(64, 128, 256, 512),
                down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
                up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
                cross_attention_dim=condition_dim,
            )
        
        if pretrained is not None:
            from safetensors import safe_open
            tensors = {}
            with safe_open(pretrained, framework="pt") as f:
                for k in f.keys():
                    tensors[k] = f.get_tensor(k)
            logger.info(self.model.load_state_dict(tensors, strict=True))

        # Noise scheduler, optimizer and LR scheduler.
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
        # self.model.set_attn_processor(diffusers.models.attention_processor.AttnProcessor())

        self.lang_model = lang_model
        # if lang_model == 'use':
        #     self.text_encoder = SentenceTransformer("sentence-transformers/sentence-t5-base")
        # else:
        #     self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        #     self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        # self.text_encoder.eval()

        # for param in self.text_encoder.parameters():
        #     param.requires_grad = False
        for param in self.flow_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def gen_flow(self, images):
        if images.shape[1] < 2:
            images = torch.cat([images, images], dim=1)  # Duplicate the first frame if only one frame is provided.
        flow_input, _ = self.flow_transform(images, images)
        start_im = einops.rearrange(flow_input[:, :-1], "b t c h w -> (b t) c h w")
        end_im = einops.rearrange(flow_input[:, 1:], "b t c h w -> (b t) c h w")
        with torch.no_grad():
            flow_tensor = self.flow_model(start_im, end_im, num_flow_updates=6)[-1]
                
        if self.flow_to_rgb:
            flow_tensor = flow_to_image(flow_tensor)
            flow_tensor = (flow_tensor - self.mean) / self.std
        else:
            flow_dim = flow_tensor.shape[-1]  # Image size is always square.
            flow_tensor = (flow_tensor + flow_dim) / (flow_dim * 2)

        flow_tensor = einops.rearrange(flow_tensor, "(b t) c h w -> b t c h w", b=flow_input.shape[0])
        # normalize the flow
        return flow_tensor[:, 0]
    
    # @torch.no_grad()
    # def encode_text(self, text):
    #     if self.lang_model == "use":
    #         text_condition = self.text_encoder.encode(text, show_progress_bar = False)[:, None]
    #         text_condition = torch.tensor(text_condition, dtype=torch.float32).to(next(self.model.parameters()).device)
    #         # print(f"Text condition shape: {text_condition.shape}")
    #     else:
    #         text_condition = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(gt_rgb_flow.device)
    #         text_condition = self.text_encoder(text_condition.input_ids, return_dict=False)[0]

    #     return text_condition

    def forward(self, batch_data):
        """
        """
        if not self.training:
            return self.forward_eval(batch_data)

        gt_rgb_flow = self.gen_flow(batch_data["rgb_static"])
        norm_rgb = (batch_data["rgb_static"] - self.mean) / self.std

        text = batch_data["language"]
        text_condition = batch_data["language_embedding"].unsqueeze(1)  # Assuming text_condition is already precomputed and passed in the batch_data
        # text_condition = self.encode_text(text)

        bs = gt_rgb_flow.shape[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=gt_rgb_flow.device, dtype=torch.int64
        )

        image_condition = norm_rgb[:, 0]
        noise = torch.randn(gt_rgb_flow.shape, device=gt_rgb_flow.device)
        noisy_flow = self.noise_scheduler.add_noise(gt_rgb_flow, noise, timesteps)
        model_inputs = [noisy_flow, image_condition]

        # TO-DO prev_flow
        
        prev_flow = gt_rgb_flow
        prev_flow = torch.ones_like(gt_rgb_flow) * 0.5
        model_inputs.append(prev_flow)
        model_inputs = torch.concat(model_inputs, dim=1)

        # logger.info(f"Model inputs shape: {model_inputs.shape}, timesteps shape: {timesteps.shape}, text_condition shape: {text_condition.shape}")
        # exit(0)
        noise_pred = self.model(model_inputs, timesteps, text_condition, return_dict=False)[0]

        loss = F.mse_loss(noise_pred, noise)

        outputs = {
            "total_loss": loss,
        }

        return outputs

    def forward_eval(self, batch_data, num_inference_steps=25):
        scheduler = DDIMScheduler(num_train_timesteps=1000)  # You can adjust timesteps based on your model's training
        scheduler.set_timesteps(num_inference_steps)

        gt_rgb_flow = self.gen_flow(batch_data["rgb_static"])
        
        norm_rgb = (batch_data["rgb_static"] - self.mean) / self.std

        norm_rgb = norm_rgb[:, 0]  # Use the first frame for conditioning
        text = batch_data["language"]
        text_condition = batch_data["language_embedding"].unsqueeze(1)  # Assuming text_condition is already precomputed and passed in the batch_data
        # text_condition = self.encode_text(text)

        bs = norm_rgb.shape[0]
        start_flow = torch.randn(gt_rgb_flow.shape, device=gt_rgb_flow.device)
        latents = start_flow.clone()

        # Iterate through DDIM timesteps
        for t in scheduler.timesteps:
            # Prepare the model inputs
            with torch.no_grad():
                # Predict the noise (epsilon) using the model
                prev_flow = gt_rgb_flow
                prev_flow  = torch.ones_like(prev_flow) * 0.5
                model_input = torch.concat([latents, norm_rgb, prev_flow], dim=1)
                time_step = torch.ones(latents.shape[0], dtype=torch.int64, device=latents.device) * t
                predicted_noise = self.model(model_input, time_step, text_condition, return_dict=False)[0]

            # Update the latent based on DDIM step
            latents = scheduler.step(predicted_noise, t, latents).prev_sample

        # The final latent is the denoised output
        generated_flow = latents
        outputs = {
            "generated_flow": generated_flow,
            "gt_flow": gt_rgb_flow,
            "visual_input": norm_rgb,
            # "feats": torch.cat([norm_rgb, generated_flow], dim=1),  # Concatenate the RGB and generated flow for visualization
        }
        
        # if batch_data["rgb_static"].shape[1] > 1:
        #     gt_rgb_flow = self.gen_flow(batch_data["rgb_static"])
        
        with torch.no_grad():
            loss = torch.mean((generated_flow - gt_rgb_flow) ** 2)

        outputs["total_loss"] = loss

        return outputs

    def visualize(self, batch_data, outputs):
        import wandb
        from diffusers.utils import make_image_grid
        from PIL import Image
        from .flow_utils import visualize_flow_vectors_as_PIL, FlowNormalizer
        gt_rgb_flow = self.gen_flow(batch_data["rgb_static"])
        images = batch_data["rgb_static"][:, 0]
        text = batch_data["language"]

        generated_flow = outputs["generated_flow"]
        if self.flow_to_rgb:
            generated_flow = ((generated_flow * self.std) + self.mean).clamp(0, 255).to(torch.uint8)
            gt_rgb_flow = ((gt_rgb_flow * self.std) + self.mean).clamp(0, 255).to(torch.uint8)        
    
        # Convert to numpy for visualization
        images_np = images.permute(0, 2, 3, 1).cpu().numpy()
        generated_flow_np = generated_flow.permute(0, 2, 3, 1).cpu().numpy()
        gt_rgb_flow_np = gt_rgb_flow.permute(0, 2, 3, 1).cpu().numpy()
        
        images = []
        for i in range(min(16, images_np.shape[0])):
            if not self.flow_to_rgb:
                img = visualize_flow_vectors_as_PIL(images_np[i], None, title="Image")
                normalizer = FlowNormalizer(self.image_size, self.image_size)
                gt_flow = normalizer.unnormalize(gt_rgb_flow_np[i])
                gt = visualize_flow_vectors_as_PIL(images_np[i], gt_flow, step=4, title="Ground Truth Optical Flow")
                pd_flow = normalizer.unnormalize(generated_flow_np[i])
                generated = visualize_flow_vectors_as_PIL(images_np[i], pd_flow, step=4, title="Generated Optical Flow")
            else:
                img = Image.fromarray(images_np[i])
                gt = Image.fromarray(gt_rgb_flow_np[i])
                generated = Image.fromarray(generated_flow_np[i])
            
            grid = make_image_grid([img, gt, generated],
                rows = 1,
                cols = 3,
            )
                
            images.append(
                wandb.Image(grid, caption=text[i])
            )
        return images
        
if __name__ == "__main__":

    model = MotionEstimation(
        image_size=128, 
        in_channels=6, 
        out_channels=3, 
        condition_dim=512, 
        size="B", 
        flow_to_rgb=True
    ).cuda()

    batch_data = {
        "rgb_static": torch.randn(8, 2, 3, 128, 128).cuda(),  # Batch of 8 images, 3 channels, 128x128
        "text_condition": torch.randn(8, 1, 512).cuda(),  # Batch of 8 text embeddings
        "language": "move this part to the left"
    }

    outputs = model(batch_data)
    print(f"Loss: {outputs['loss'].item()}")
