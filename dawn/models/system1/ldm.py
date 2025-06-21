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

from humanfriendly import format_size
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, PeftModel

logger = logging.getLogger(__name__)

class LatentMotionEstimation(nn.Module):
    def __init__(self, 
            pretrained: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
            image_size=256, 
            in_channels=8, 
            out_channels=3, 
            condition_dim=768, 
            flow_to_rgb=True, 
            num_inference_steps=25,

            # LoRA specific parameters
            enable_lora: bool = True,
            lora_rank: int = 4, # Common LoRA rank
            lora_alpha: int = 32, # Common LoRA alpha
            lora_dropout: float = 0.0, # Dropout for LoRA layers
            lora_weights_path: str = None, # Path to load pre-trained LoRA weights

        ):
        super().__init__()
        logger.info(f"Initializing {__class__.__name__} with image size {image_size}, in_channels {in_channels}, out_channels {out_channels}, condition_dim {condition_dim}, flow_to_rgb {flow_to_rgb}.")
        self.flow_model = optical_flow.raft_large(weights=optical_flow.Raft_Large_Weights.DEFAULT, progress=False).eval()
        self.flow_transform = optical_flow.Raft_Large_Weights.DEFAULT.transforms()
        self.flow_to_rgb = flow_to_rgb
        self.image_size = image_size
        self.num_inference_steps = num_inference_steps

        self.enable_lora = enable_lora
        self.lora_weights_path = lora_weights_path

        
        self.pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            pretrained,
            safety_checker=None,
            requires_safety_checker=False,
            # attn_implementation="flash_attention_2",
            # torch_dtype=torch.float16,
        )

        self.pipeline.set_progress_bar_config(disable=True)
        self.pipeline.enable_xformers_memory_efficient_attention()

        self.unet = self.pipeline.unet
        self.vae = self.pipeline.vae
        # self.pipeline.vae = self.vae

        self.tokenizer = self.pipeline.tokenizer
        self.text_encoder = self.pipeline.text_encoder
        self.unet.conv_in = nn.Conv2d(
            in_channels, self.unet.conv_in.out_channels, kernel_size=self.unet.conv_in.kernel_size, stride=self.unet.conv_in.stride, padding=self.unet.conv_in.padding
        )
        self.unet.register_to_config(in_channels=in_channels)

        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.flow_model.requires_grad_(False)


        # self.tokenizer = CLIPTokenizer.from_pretrained(
        #     pretrained, subfolder="tokenizer")
        # self.text_encoder = CLIPTextModel.from_pretrained(
        #     pretrained, subfolder="text_encoder")
        # self.vae = AutoencoderKL.from_pretrained(
        #     pretrained, subfolder="vae")
        # self.unet = UNet2DConditionModel.from_pretrained(
        #     pretrained, subfolder="unet")

        # self.pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        #     # pretrained,
        #     "stable-diffusion-v1-5/stable-diffusion-v1-5",
        #     unet = self.unet,
        #     text_encoder=self.text_encoder,
        #     vae=self.vae,
        #     tokenizer=self.tokenizer,
        #     safety_checker=None,
        #     requires_safety_checker=False,
        #     # attn_implementation="flash_attention_2",
        #     # torch_dtype=torch.float16,
        # )
        self.enable_lora=False
        if self.enable_lora:
            # Configure LoRA for the UNet
            # You can also add LoRA for the text_encoder if needed, but UNet is primary
            lora_config = LoraConfig(
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["to_q", "to_v", "query", "value", "ff.net.0.proj"],
                bias="none",
            )
            self.unet = get_peft_model(self.unet, lora_config)

            self.unet.conv_in.requires_grad_(True)

            # Apply LoRA to the UNet
            # get_peft_model will automatically wrap the target modules
            logger.info("LoRA applied to UNet.")

            # Load LoRA weights if a path is provided
            if self.lora_weights_path:
                logger.info(f"Loading LoRA weights from {self.lora_weights_path}")
                # For `load_lora_weights` on the pipeline:
                # This is for loading LoRA weights for inference directly into the pipeline
                self.pipeline.load_lora_weights(self.lora_weights_path)
                logger.info(f"LoRA weights loaded into pipeline for inference.")

                # If you need to explicitly load state_dict into the PEFT model (e.g., after training your own LoRA)
                # You'd typically load the adapter directly into the PEFT model
                # peft_model_state_dict = torch.load(self.lora_weights_path, map_location="cpu")
                # set_peft_model_state_dict(self.unet, peft_model_state_dict)
                # logger.info("LoRA weights loaded into PEFT UNet for further training/specific usage.")



        # Noise scheduler, optimizer and LR scheduler.
        self.noise_scheduler = DDIMScheduler(num_train_timesteps=1000)
        self.noise_scheduler.set_timesteps(self.num_inference_steps)  # Set the number of inference steps
        self.generator = torch.Generator(device=self.device).manual_seed(0)


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
        return flow_tensor[:, 0]
    
    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def encode_text(self, text):
        text_condition = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=20).to(self.device)
        text_condition = self.text_encoder(**text_condition, return_dict=False)[0]
        # text_condition.unsqueeze_(1)  # Add a sequence length dimension
        # print(f"Text condition shape: {text_condition.shape}")
        # exit(0)
        return text_condition

    def forward(self, batch_data):
        """
        """
        if not self.training:
            return self.forward_eval(batch_data)

        bsz = batch_data["rgb_static"].shape[0]

        # Prepare ground truth flow
        gt_rgb_flow = self.gen_flow(batch_data["rgb_static"])
        if not self.flow_to_rgb:
            add = gt_rgb_flow.mean(dim=1, keepdim=True)
            gt_rgb_flow = torch.cat([gt_rgb_flow, add], dim=1)
        norm_gt_rgb_flow = gt_rgb_flow * 2 - 1 # Normalize the flow to [-1, 1]
        # Prepare target latents
        # with torch.amp.autocast(enabled=False, device_type=self.device.type):
        latents = self.vae.encode(norm_gt_rgb_flow).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor  # Scale the latents

        # Prepare condition embeddings
        image = self.pipeline.image_processor.preprocess(batch_data["rgb_static"][:, 0])
        # with torch.amp.autocast(enabled=False):
        image_latents = self.vae.encode(image).latent_dist.sample()
        image_latents = image_latents * self.vae.config.scaling_factor  # Scale the latents
        
        text = batch_data["language"]
        # random_p = torch.rand(bsz, device=latents.device)
        # prompt_mask = (random_p < 0.1).reshape(bsz, 1, 1)
        # text = [text[i] if prompt_mask[i] else "" for i in range(bsz)]
        text_condition = self.encode_text(text)

        # logger.info(f"Text condition shape: {text_condition.shape}, Image latents shape: {image_latents.shape}, Latents shape: {latents.shape}")
        
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

        # Decode back to image
        # alpha_prod_t = self.noise_scheduler.alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        # sqrt_alpha_prod_t = alpha_prod_t.sqrt()
        # sqrt_one_minus_alpha_prod_t = (1.0 - alpha_prod_t).sqrt()
        # generated_latents = (noisy_latent - sqrt_one_minus_alpha_prod_t * noise_pred) / sqrt_alpha_prod_t
        
        # generated_latents = generated_latents / self.vae.config.scaling_factor  # Scale back the latents
        # generated_flow = self.vae.decode(generated_latents).sample
        # generated_flow = (generated_flow / 2 + 0.5).clamp(0, 1)  # Scale back to [0, 1]

        # recon_loss = F.mse_loss(generated_flow, gt_rgb_flow)

        outputs = {
            # "diffu_loss": loss,
            # "recon_loss": recon_loss,
            "total_loss": loss,
        }

        return outputs

    def forward_eval(self, batch_data):

        # Prepare ground truth flow
        gt_rgb_flow = self.gen_flow(batch_data["rgb_static"])

        # if not self.flow_to_rgb:
        #     add = gt_rgb_flow.mean(dim=1, keepdim=True)
        #     gt_rgb_flow = torch.cat([gt_rgb_flow, add], dim=1)
        # norm_gt_rgb_flow = gt_rgb_flow * 2 - 1 # Normalize the flow to [-1, 1]
        # # Prepare target latents
        # # with torch.amp.autocast(enabled=False, device_type=self.device.type):
        # latents = self.vae.encode(norm_gt_rgb_flow).latent_dist.sample()
        # latents = latents * self.vae.config.scaling_factor  # Scale the latents

        # image = batch_data["rgb_static"][:, 0]
        image = self.pipeline.image_processor.preprocess(batch_data["rgb_static"][:, 0])
        image_latents = self.vae.encode(image).latent_dist.sample()
        image_latents = image_latents * self.vae.config.scaling_factor  # Scale the latents

        #         
        text = batch_data["language"]
        text_condition = self.encode_text(text)

        # 
        latents = torch.randn(image_latents.shape, device=image_latents.device)
        # Iterate through DDIM timesteps
        for t in self.noise_scheduler.timesteps:
            # Prepare the model inputs
            with torch.no_grad():
                # Predict the noise (epsilon) using the model
                model_input = torch.concat([latents, image_latents], dim=1)
                time_step = torch.ones(latents.shape[0], dtype=torch.int64, device=latents.device) * t
                predicted_noise = self.unet(model_input, time_step, text_condition, return_dict=False)[0]

            # Update the latent based on DDIM step
            latents = self.noise_scheduler.step(predicted_noise, t, latents).prev_sample

        latents = latents / self.vae.config.scaling_factor  # Scale back the latents

        generated_flow = self.vae.decode(latents).sample
        generated_flow = (generated_flow / 2 + 0.5).clamp(0, 1)  # Scale back to [0, 1]

        if not self.flow_to_rgb:
            generated_flow = generated_flow[:, :2]  # Keep only the first two channels for flow
            gt_rgb_flow = gt_rgb_flow[:, :2]  # Keep only the first two channels for ground truth flow
        
        # add = gt_rgb_flow.mean(dim=1, keepdim=True)
        # flow = torch.cat([gt_rgb_flow, add], dim=1)
        # flow = gt_rgb_flow
        # flow = flow * 2 - 1 # Normalize the flow to [-1, 1]
        # Prepare target latents
        
        # latents = self.vae.encode(flow).latent_dist.sample()
        # latents = latents * self.vae.config.scaling_factor  # Scale the latents
        
        # generated_flow = self.vae.decode(latents).sample
        # generated_flow = (generated_flow / 2 + 0.5).clamp(0, 1)  # Scale back to [0, 1]


        # generated_flow = self.pipeline(
        #     image=image,
        #     prompt_embeds=text_condition,
        #     num_inference_steps=self.num_inference_steps,
        #     image_guidance_scale=1.5,
        #     guidance_scale=7,
        #     generator=self.generator,
        #     output_type="pt",
        # ).images

        # print(generated_flow.shape, generated_flow.min(), generated_flow.max())
        # generated_flow = generated_flow[:, :2]
        # exit(0)


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
            generated_flow = (generated_flow * 255).to(torch.uint8)
            gt_rgb_flow = (gt_rgb_flow * 255).to(torch.uint8)        
    
        # Convert to numpy for visualization
        images_np = (images.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
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
