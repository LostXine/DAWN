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

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionInstructPix2PixPipeline,
    UNet2DConditionModel,
)

from transformers import AutoModel, CLIPTextModel, CLIPTokenizer, CLIPVisionModel

from humanfriendly import format_size
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, PeftModel

logger = logging.getLogger(__name__)

class MotionSDXL(nn.Module):
    def __init__(self, 
            pretrained: str = "stabilityai/stable-diffusion-xl-base-1.0",
            image_size=256, 
            out_channels=3, 
            condition_dim=768, 
            flow_to_rgb=True, 
            use_cfg: bool = False,
            conditioning_dropout_prob = 0.1, # Probability of conditioning dropout
            guidance_scale: float = 7.5, # Guidance scale for classifier-free guidance
            num_inference_steps=25,
            use_interval: bool = False, # Whether to use interval embeddings
            use_text_sentence: bool = False, # Whether to use text sentences for conditioning
            input_type: str = "rgb_static", # Input type for the model, can be "rgb_static" or "rgb_gripper"
            support_types: list = ["rgb_gripper"], # Supported input types
            **kwargs  # Additional arguments for the model
        ):
        super().__init__()
        logger.info(f"Initializing {__class__.__name__} with image size {image_size}, flow_to_rgb {flow_to_rgb}.")
        self.flow_model = optical_flow.raft_large(weights=optical_flow.Raft_Large_Weights.DEFAULT, progress=False).eval()
        self.flow_transform = optical_flow.Raft_Large_Weights.DEFAULT.transforms()
        self.flow_to_rgb = flow_to_rgb
        self.image_size = image_size
        self.num_inference_steps = num_inference_steps
        self.use_cfg = use_cfg
        self.guidance_scale = guidance_scale
        self.use_interval = use_interval
        self.conditioning_dropout_prob = conditioning_dropout_prob
        
        self.use_text_sentence = use_text_sentence
        
        self.input_type = input_type
        self.support_types = support_types
        if self.input_type not in ["rgb_static", "rgb_gripper"]:
            raise ValueError(f"Invalid input type: {self.input_type}. Supported types are 'rgb_static' and 'rgb_gripper'.")
        

        self.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix")

        self.pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            pretrained,
            safety_checker=None,
            requires_safety_checker=False,
            use_safetensors=True,
        )
        
        self.unet = self.pipeline.unet

        self.pipeline.vae = self.vae
        self.text_encoder = self.pipeline.text_encoder
        self.tokenizer = self.pipeline.tokenizer
        self.scheduler = self.pipeline.scheduler

        self.unet.enable_gradient_checkpointing()
        self.noise_scheduler = DDIMScheduler(num_train_timesteps=1000)
        self.noise_scheduler.set_timesteps(self.num_inference_steps)  # Set the number of inference steps
        self.generator = torch.Generator(device=self.device).manual_seed(0)


        in_channels = 8
        out_channels = self.unet.conv_in.out_channels
        self.unet.register_to_config(in_channels=in_channels, sample_size=256 // 8)

        with torch.no_grad():
            new_conv_in = nn.Conv2d(
                in_channels, out_channels, self.unet.conv_in.kernel_size, self.unet.conv_in.stride, self.unet.conv_in.padding
            )
            new_conv_in.weight.zero_()
            new_conv_in.weight[:, :4, :, :].copy_(self.unet.conv_in.weight)
            self.unet.conv_in = new_conv_in

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        # self.text_encoder_2.requires_grad_(False)
        self.flow_model.requires_grad_(False)
        
        # self.generator = torch.Generator(device=self.device).manual_seed(0)

        self.feature_extractor = AutoModel.from_pretrained("google/vit-base-patch16-224-in21k", add_pooling_layer=False)
        
        condition_dim = self.unet.config.cross_attention_dim
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

        if not self.flow_to_rgb:
            add = flow_tensor.mean(dim=2, keepdim=True)
            flow_tensor = torch.cat([flow_tensor, add], dim=2)

        return flow_tensor

    @property
    def device(self):
       return next(self.parameters()).device
    
    def encode_image(self, visual_input, flow = None):
        if flow is not None:
            visual_input = torch.cat([visual_input, flow], dim=1)  # Concatenate image and flow along the channel dimension
        
        visual_input = F.interpolate(visual_input, size=(self.feature_extractor.config.image_size, self.feature_extractor.config.image_size), mode='bilinear', align_corners=False)
        visual_feat = self.feature_extractor(visual_input).last_hidden_state  # [B, C, H, W] -> [B, N, C]
        # visual_feat = self.mlp(visual_feat)  # Project to condition_dim
        return visual_feat

    def encode_prompt(self, batch_data):
        text = batch_data["language"]
        batch_size = len(text)
        (
            prompt_embeds, negative_prompt_embeds, 
            pooled_prompt_embeds, negative_pooled_prompt_embeds
        ) = self.pipeline.encode_prompt(prompt=text, device=self.device)

        support_embed = self.get_support_embedding(batch_data)
        if support_embed is not None:
            prompt_embeds = torch.cat([prompt_embeds, support_embed], dim=1)  # Concatenate along the sequence length dimension
            # negative_prompt_embeds = torch.cat([negative_prompt_embeds, torch.zeros_like(support_embed)], dim=1)
        
        if self.use_interval:
            interval_embed = self.interval_embed(batch_data["skip_frame"]).unsqueeze(1)  # Shape: (bsz, 1, condition_dim)
            prompt_embeds = torch.cat([interval_embed, prompt_embeds], dim=1)  # Concatenate along the sequence length dimension
            # negative_prompt_embeds = torch.cat([torch.zeros_like(interval_embed), negative_prompt_embeds], dim=1)

        negative_prompt_embeds = torch.zeros_like(prompt_embeds, device=prompt_embeds.device)  # Use zero embeddings for negative prompts

        add_time_ids = self.pipeline._get_add_time_ids(
            (self.image_size, self.image_size),
            (0, 0),
            (self.image_size, self.image_size),
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=self.text_encoder_2.config.projection_dim,
        )
        add_time_ids = add_time_ids.to(self.device).repeat(batch_size, 1)
        print(add_time_ids.shape, pooled_prompt_embeds.shape)
        return (
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds, add_time_ids
        )

    def forward(self, batch_data, **kwargs):
        if not self.training:
            return self.forward_eval(batch_data, **kwargs)

        bsz = batch_data[self.input_type].shape[0]

        # Prepare ground truth flow
        gt_rgb_flow = self.gen_flow(batch_data[self.input_type])
        
        norm_gt_rgb_flow = gt_rgb_flow * 2 - 1 # Normalize the flow to [-1, 1]
        latents = self.vae.encode(norm_gt_rgb_flow[:, -1]).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor  # Scale the latents

        # Prepare condition embeddings
        image = batch_data[self.input_type][:, -2] * 2 - 1  # Normalize the image to [-1, 1]
        
        image_latents = self.vae.encode(image).latent_dist.sample()
        image_latents = image_latents * self.vae.config.scaling_factor  # Scale the latents
        (
            prompt_embeds, negative_prompt_embeds, 
            pooled_prompt_embeds, negative_pooled_prompt_embeds,
            add_time_ids,
        ) = self.encode_prompt(batch_data)
        prompt_mask = torch.rand(bsz, 1, 1, device=latents.device) < self.conditioning_dropout_prob
        prompt_embeds = torch.where(prompt_mask, negative_prompt_embeds, prompt_embeds)
        pooled_prompt_embeds = torch.where(prompt_mask.squeeze(-1), negative_pooled_prompt_embeds, pooled_prompt_embeds)

        # Prepare noise
        bs = latents.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=latents.device, dtype=torch.int64)
        noise = torch.randn(latents.shape, device=latents.device)

        noisy_latent = self.noise_scheduler.add_noise(latents, noise, timesteps)
        model_inputs = [noisy_latent, image_latents]
        model_inputs = torch.concat(model_inputs, dim=1)  # Concatenate the noisy latents and image latents

        noise_pred = self.unet(
            model_inputs, 
            timesteps, 
            encoder_hidden_states=prompt_embeds, 
            added_cond_kwargs={
                "text_embeds": pooled_prompt_embeds,
                "time_ids": add_time_ids,
            },  
            return_dict=False
        )[0]
        
        loss = F.mse_loss(noise_pred, noise)
        outputs = { "diffu_loss": loss }
        outputs["total_loss"] = sum([v for k, v in outputs.items() if "loss" in k])

        return outputs
    
    def get_support_embedding(self, batch_data):
        if len(self.support_types) == 0:
            return None
        
        support_embed = []
        for support_type in self.support_types:
            if support_type not in batch_data:
                continue
            support_image = batch_data[support_type][:, -2] * 2 - 1
            support_feat = self.encode_image(support_image)
            support_embed.append(support_feat)
        
        if len(support_embed) == 0:
            return None
        return torch.cat(support_embed, dim=1)

    
    def forward_eval(self, batch_data, inference_step=None, **kwargs):
        if inference_step is not None:
            self.noise_scheduler.set_timesteps(inference_step)
        
        bsz = batch_data[self.input_type].shape[0]
        # Prepare ground truth flow
        gt_rgb_flow = self.gen_flow(batch_data[self.input_type])
        norm_gt_rgb_flow = gt_rgb_flow * 2 - 1 # Normalize the flow to [-1, 1]

        image = batch_data[self.input_type][:, -2] * 2 - 1  # Normalize the image to [-1, 1]
        
        (
            prompt_embeds, negative_prompt_embeds, 
            pooled_prompt_embeds, negative_pooled_prompt_embeds
        ) = self.encode_prompt(batch_data)

        with torch.no_grad():
            generated_flow = pipeline(
                image=image,
                prompt_embeds = prompt_embeds,
                negative_prompt_embeds = negative_prompt_embeds,
                pooled_prompt_embeds = pooled_prompt_embeds,
                negative_pooled_prompt_embeds = negative_pooled_prompt_embeds,
                width=self.image_size,
                height=self.image_size,
                num_inference_steps=self.num_inference_steps,
                image_guidance_scale=1.5,
                guidance_scale=7,
                # generator=self.generator,
                output_type="pt",
            ).images

        gt_rgb_flow = gt_rgb_flow[:, -1]

        
        with torch.no_grad():
            loss = torch.mean((generated_flow - gt_rgb_flow).abs())
        
        outputs = {}
        outputs["use_prev"] = use_prev
        outputs["mse_loss"] = F.mse_loss(generated_flow, gt_rgb_flow)
        outputs["l1_loss"] = loss
        outputs["total_loss"] = outputs["mse_loss"] + outputs["l1_loss"]

        if not self.flow_to_rgb:
            generated_flow = generated_flow[:, :2]  # Keep only the first two channels for flow
            gt_rgb_flow = gt_rgb_flow[:, :2]  # Keep only the first two channels for ground truth flow
        
        outputs.update({
            "generated_flow": generated_flow,
            "gt_flow": gt_rgb_flow,
            "visual_input": image,
            "lang_feat": prompt_embeds
        })
        
        return outputs
    
    def visualize(self, batch_data, outputs, inference=False):
        import wandb
        from diffusers.utils import make_image_grid
        from PIL import Image
        from .flow_utils import visualize_flow_vectors_as_PIL, FlowNormalizer
        
        generated_flow = outputs["generated_flow"]
        images = batch_data[self.input_type][:, -2]
        goals = batch_data[self.input_type][:, -1]

        support_images = []
        if len(self.support_types):
            for support_type in self.support_types:
                if support_type in batch_data:
                    img = batch_data[support_type][:, -2]
                    img = (img.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
                    support_images.append(img)
            
        text = batch_data["language"]
        
        if not inference:
            gt_rgb_flow = outputs["gt_flow"]
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
                support = [visualize_flow_vectors_as_PIL(support_images[j][i], None, 
                    title=f"Support {j}: {self.support_types[j]}") 
                            for j in range(len(support_images))]
                normalizer = FlowNormalizer(self.image_size, self.image_size)
                gt_flow = normalizer.unnormalize(gt_rgb_flow_np[i])
                gt = visualize_flow_vectors_as_PIL(images_np[i], gt_flow, step=8, title="Ground Truth Optical Flow")
                pd_flow = normalizer.unnormalize(generated_flow_np[i])
                generated = visualize_flow_vectors_as_PIL(images_np[i], pd_flow, step=8, title="Generated Optical Flow")
            else:
                img = Image.fromarray(images_np[i])
                goal = Image.fromarray(goals_np[i])
                gt = Image.fromarray(gt_rgb_flow_np[i])
                generated = Image.fromarray(generated_flow_np[i])
            
            if inference:
                img = np.array(generated.convert("RGB"))
                images.append(img)
            else:
                grid = make_image_grid([img, goal, *support, gt, generated, *support],
                    rows = 2,
                    cols = 2 + len(support_images),
                )
                    
                images.append(
                    wandb.Image(grid, caption=text[i])
                )
        
        return images
