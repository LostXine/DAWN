import logging
import torch
from torch import nn
import torch.nn.functional as F
import diffusers
from diffusers import DDPMScheduler, DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
import einops
from torchvision.models import optical_flow
from torchvision.utils import flow_to_image
import numpy as np
import random
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionInstructPix2PixPipeline,
    UNet2DConditionModel,
)

from transformers import AutoModel
from humanfriendly import format_size
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, PeftModel

logger = logging.getLogger(__name__)


class LatentMotionEstimationV2(nn.Module):
    def __init__(self, 
            pretrained: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
            image_size=256, 
            in_channels=8, 
            out_channels=3, 
            condition_dim=768, 
            flow_to_rgb=True, 
            use_cfg: bool = False,
            guidance_scale: float = 7.5, # Guidance scale for classifier-free guidance
            num_inference_steps=25,
            use_interval: bool = False, # Whether to use interval embeddings

        ):
        super().__init__()
        logger.info(f"Initializing {__class__.__name__} with image size {image_size}, in_channels {in_channels}, out_channels {out_channels}, condition_dim {condition_dim}, flow_to_rgb {flow_to_rgb}.")
        self.flow_model = optical_flow.raft_large(weights=optical_flow.Raft_Large_Weights.DEFAULT, progress=False).eval()
        self.flow_transform = optical_flow.Raft_Large_Weights.DEFAULT.transforms()
        self.flow_to_rgb = flow_to_rgb
        self.image_size = image_size
        self.num_inference_steps = num_inference_steps
        self.use_cfg = use_cfg
        self.guidance_scale = guidance_scale
        self.use_interval = use_interval

        self.pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            pretrained,
            safety_checker=None,
            requires_safety_checker=False,
            # attn_implementation="flash_attention_2",
            # torch_dtype=torch.float16,
        )

        # Load scheduler, tokenizer and models.
        self.unet = self.pipeline.unet
        self.tokenizer = self.pipeline.tokenizer
        self.text_encoder = self.pipeline.text_encoder
        self.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix")

        # self.noise_scheduler = DDPMScheduler.from_pretrained(pretrained, subfolder="scheduler")
        # self.tokenizer = CLIPTokenizer.from_pretrained(pretrained, subfolder="tokenizer")
        # self.text_encoder = CLIPTextModel.from_pretrained(pretrained, subfolder="text_encoder")
        # self.unet = UNet2DConditionModel.from_pretrained(pretrained, subfolder="unet")
        # self.vae = AutoencoderKL.from_pretrained(pretrained, subfolder="vae")

        self.unet.conv_in = nn.Conv2d(
            in_channels, self.unet.conv_in.out_channels, kernel_size=self.unet.conv_in.kernel_size, stride=self.unet.conv_in.stride, padding=self.unet.conv_in.padding
        )
        self.unet.register_to_config(in_channels=in_channels)

        # History 
        # self.image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
        self.feature_extractor = AutoModel.from_pretrained("google/vit-base-patch16-224-in21k", add_pooling_layer=False)
        patch_embed = self.feature_extractor.embeddings.patch_embeddings.projection 
        self.feature_extractor.embeddings.patch_embeddings.projection = torch.nn.Conv2d(5, patch_embed.out_channels, kernel_size=patch_embed.kernel_size, stride=patch_embed.stride, padding=patch_embed.padding)
        self.feature_extractor.embeddings.patch_embeddings.num_channels = 5


        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.flow_model.requires_grad_(False)

        
        # Noise scheduler, optimizer and LR scheduler.
        self.noise_scheduler = DDIMScheduler(num_train_timesteps=1000)
        self.noise_scheduler.set_timesteps(self.num_inference_steps)  # Set the number of inference steps
        self.generator = torch.Generator(device=self.device).manual_seed(0)

        if self.use_interval:
            self.interval_embed = nn.Embedding(31, condition_dim)


        total_params = sum(p.numel() for p in self.parameters())
        total_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {format_size(total_params)}, Trainable parameters: {format_size(total_trainable_params)}")

    def load_weights(self):
        pass
    
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
            flow_tensor = flow_to_image(flow_tensor) / 255.
        else:
            flow_dim = flow_tensor.shape[-1]  # Image size is always square.
            flow_tensor = (flow_tensor + flow_dim) / (flow_dim * 2)

        flow_tensor = einops.rearrange(flow_tensor, "(b t) c h w -> b t c h w", b=flow_input.shape[0])
        # normalize the flow
        return flow_tensor
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    def uncond_text(self):
        try:
            return self.uncond_text_embed
        except:
            self.uncond_text_embed = self.encode_text([""])
            return self.uncond_text_embed

    def encode_image(self, image, flow):
        visual_input = torch.cat([image, flow], dim=1)  # Concatenate image and flow along the channel dimension
        visual_input = F.interpolate(visual_input, size=(self.feature_extractor.config.image_size, self.feature_extractor.config.image_size), mode='bilinear', align_corners=False)
        visual_feat = self.feature_extractor(visual_input).last_hidden_state  # [B, C, H, W] -> [B, N, C]
        return visual_feat

    @torch.no_grad()
    def encode_text(self, text):
        text_condition = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=20).to(self.device)
        text_condition = self.text_encoder(**text_condition, return_dict=False)[0]
        # text_condition.unsqueeze_(1)  # Add a sequence length dimension
        # print(f"Text condition shape: {text_condition.shape}")
        # exit(0)
        return text_condition

    def forward(self, batch_data, **kwargs):
        """
        """
        if not self.training:
            return self.forward_eval(batch_data, **kwargs)

        bsz = batch_data["rgb_static"].shape[0]

        # Prepare ground truth flow
        gt_rgb_flow = self.gen_flow(batch_data["rgb_static"])
        if not self.flow_to_rgb:
            add = gt_rgb_flow.mean(dim=2, keepdim=True)
            gt_rgb_flow = torch.cat([gt_rgb_flow, add], dim=2)

        norm_gt_rgb_flow = gt_rgb_flow * 2 - 1 # Normalize the flow to [-1, 1]
        # Prepare target latents
        # with torch.amp.autocast(enabled=False, device_type=self.device.type):

        # norm_gt_rgb_flow = norm_gt_rgb_flow[:, -1]
        latents = self.vae.encode(norm_gt_rgb_flow[:, -1]).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor  # Scale the latents

        # Prepare condition embeddings
        # image = self.image_processor.preprocess(batch_data["rgb_static"][:, -2])
        image = batch_data["rgb_static"][:, -2] * 2 - 1  # Normalize the image to [-1, 1]
        # with torch.amp.autocast(enabled=False):
        image_latents = self.vae.encode(image).latent_dist.sample()
        image_latents = image_latents * self.vae.config.scaling_factor  # Scale the latents
        
        text = batch_data["language"]
        text_condition = self.encode_text(text)
        if random.random() < 0.5 or kwargs.get("use_history", False):
            norm_previous_flow = norm_gt_rgb_flow[:, :-1, :2]
            previous_rgb = batch_data["rgb_static"][:, :-2]
        
            # return {}
            b, t, c, h, w = previous_rgb.shape
            previous_rgb = einops.rearrange(previous_rgb, "b t c h w -> (b t) c h w")
            norm_previous_flow = einops.rearrange(norm_previous_flow, "b t c h w -> (b t) c h w")
            history_feat = self.encode_image(previous_rgb, norm_previous_flow)
            history_feat = einops.rearrange(history_feat, "(b t) c n -> b (t c) n", b=bsz)

            text_condition = torch.cat([text_condition, history_feat], dim=1)  # Concatenate along the sequence length dimension
        if self.use_interval:
            interval_embed = self.interval_embed(batch_data["skip_frame"]).unsqueeze(1)  # Shape: (bsz, 1, condition_dim)
            text_condition = torch.cat([interval_embed, text_condition], dim=1)  # Concatenate along the sequence length dimension
        
        # Prepare noise
        bs = latents.shape[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=latents.device, dtype=torch.int64
        )
        noise = torch.randn(latents.shape, device=latents.device)

        noisy_latent = self.noise_scheduler.add_noise(latents, noise, timesteps)
        model_inputs = [noisy_latent, image_latents]

        model_inputs = torch.concat(model_inputs, dim=1)  # Concatenate the noisy latents and image latents

        noise_pred = self.unet(model_inputs, timesteps, text_condition, return_dict=False)[0]
        
        loss = F.mse_loss(noise_pred, noise)
        outputs = { "diffu_loss": loss }
        outputs["total_loss"] = sum([v for k, v in outputs.items() if "loss" in k])

        return outputs
   
    def forward_eval(self, batch_data, inference_step=None, **kwargs):
        if inference_step is not None:
            self.noise_scheduler.set_timesteps(inference_step)
        
        bsz = batch_data["rgb_static"].shape[0]
        # Prepare ground truth flow
        gt_rgb_flow = self.gen_flow(batch_data["rgb_static"])
        norm_gt_rgb_flow = gt_rgb_flow * 2 - 1 # Normalize the flow to [-1, 1]

        # if not self.flow_to_rgb:
        #     add = gt_rgb_flow.mean(dim=1, keepdim=True)
        #     gt_rgb_flow = torch.cat([gt_rgb_flow, add], dim=1)
        # # Prepare target latents
        # # with torch.amp.autocast(enabled=False, device_type=self.device.type):
        # latents = self.vae.encode(norm_gt_rgb_flow).latent_dist.sample()
        # latents = latents * self.vae.config.scaling_factor  # Scale the latents

        # image = batch_data["rgb_static"][:, 0]
        image = self.pipeline.image_processor.preprocess(batch_data["rgb_static"][:, -2])
        image_latents = self.vae.encode(image).latent_dist.sample()
        image_latents = image_latents * self.vae.config.scaling_factor  # Scale the latents

        #         
        text = batch_data["language"]
        text_embed = self.encode_text(text)
        text_condition = text_embed
        # if kwargs.get("use_history", False):
        norm_previous_flow = norm_gt_rgb_flow[:, :-1, :2]
        previous_rgb = batch_data["rgb_static"][:, :-2]
    
        b, t, c, h, w = previous_rgb.shape
        previous_rgb = einops.rearrange(previous_rgb, "b t c h w -> (b t) c h w")
        norm_previous_flow = einops.rearrange(norm_previous_flow, "b t c h w -> (b t) c h w")
        history_feat = self.encode_image(previous_rgb, norm_previous_flow)
        history_feat = einops.rearrange(history_feat, "(b t) c n -> b (t c) n", b=bsz)

        text_condition = torch.cat([text_condition, history_feat], dim=1)  # Concatenate along the sequence length dimension
        if self.use_interval:
            interval_embed = self.interval_embed(batch_data["skip_frame"]).unsqueeze(1)
            text_condition = torch.cat([interval_embed, text_condition], dim=1)  # Concatenate along the sequence length dimension


        latents = torch.randn(image_latents.shape, device=image_latents.device)
        # Iterate through DDIM timesteps
        for t in self.noise_scheduler.timesteps:
            # Prepare the model inputs
            model_input = torch.concat([latents, image_latents], dim=1)

            time_step = torch.ones(model_input.shape[0], dtype=torch.int64, device=latents.device) * t
            predicted_noise = self.unet(model_input, time_step, text_condition, return_dict=False)[0]

            # Update the latent based on DDIM step
            latents = self.noise_scheduler.step(predicted_noise, t, latents).prev_sample

        latents = latents / self.vae.config.scaling_factor  # Scale back the latents

        generated_flow = self.vae.decode(latents).sample
        generated_flow = (generated_flow / 2 + 0.5).clamp(0, 1)  # Scale back to [0, 1]

        gt_rgb_flow = gt_rgb_flow[:, -1]
        if not self.flow_to_rgb:
            generated_flow = generated_flow[:, :2]  # Keep only the first two channels for flow
            gt_rgb_flow = gt_rgb_flow[:, :2]  # Keep only the first two channels for ground truth flow
        

        outputs = {
            "generated_flow": generated_flow,
            "gt_flow": gt_rgb_flow,
            "visual_input": image,

        }
        
        with torch.no_grad():
            loss = torch.mean((generated_flow - gt_rgb_flow).abs())

        outputs["mse_loss"] = F.mse_loss(generated_flow, gt_rgb_flow)
        outputs["l1_loss"] = loss
        outputs["total_loss"] = outputs["mse_loss"] + outputs["l1_loss"]

        return outputs

    def visualize(self, batch_data, outputs, inference=False):
        import wandb
        from diffusers.utils import make_image_grid
        from PIL import Image
        from .flow_utils import visualize_flow_vectors_as_PIL, FlowNormalizer
        
        generated_flow = outputs["generated_flow"]
        images = batch_data["rgb_static"][:, -2]
        goals = batch_data["rgb_static"][:, -1]
        text = batch_data["language"]
        
        if not inference:
            gt_rgb_flow = self.gen_flow(batch_data["rgb_static"])[:, -1]
        else:
            gt_rgb_flow = torch.zeros_like(generated_flow) + 0.5  

        if self.flow_to_rgb:
            generated_flow = (generated_flow * 255).to(torch.uint8)
            gt_rgb_flow = (gt_rgb_flow * 255).to(torch.uint8)        
    
        loss = ((gt_rgb_flow - generated_flow) ** 2).mean(dim=[1, 2, 3]) * 1000
        # Convert to numpy for visualization
        images_np = (images.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        goals_np = (goals.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        generated_flow_np = generated_flow.permute(0, 2, 3, 1).cpu().numpy()
        gt_rgb_flow_np = gt_rgb_flow.permute(0, 2, 3, 1).cpu().numpy()
        
            
        images = []
        for i in range(min(16, images_np.shape[0])):
            if not self.flow_to_rgb:
                img = visualize_flow_vectors_as_PIL(images_np[i], None, title=text[i])
                goal = visualize_flow_vectors_as_PIL(goals_np[i], None, 
                    title=f"Interval = {batch_data['skip_frame'][i]}, Loss x 1K = {loss[i].item():.4f}"
                )
                normalizer = FlowNormalizer(self.image_size, self.image_size)
                gt_flow = normalizer.unnormalize(gt_rgb_flow_np[i])
                gt = visualize_flow_vectors_as_PIL(images_np[i], gt_flow, step=4, title="Ground Truth Optical Flow")
                pd_flow = normalizer.unnormalize(generated_flow_np[i])
                generated = visualize_flow_vectors_as_PIL(images_np[i], pd_flow, step=4, title="Generated Optical Flow")
            else:
                img = Image.fromarray(images_np[i])
                goal = Image.fromarray(goals_np[i])
                gt = Image.fromarray(gt_rgb_flow_np[i])
                generated = Image.fromarray(generated_flow_np[i])
            
            if inference:
                img = np.array(generated.convert("RGB"))
                images.append(img)
            else:
                grid = make_image_grid([img, goal, gt, generated],
                    rows = 2,
                    cols = 2,
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
