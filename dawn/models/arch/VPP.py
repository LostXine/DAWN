import torch
import logging
from humanfriendly import format_size
from dawn.models.modules.diffusion.feature_extraction import Diffusion_feature_extractor
from dawn.models.modules.transformers.video_former import Video_Former_3D
import einops

import random
import os
logger = logging.getLogger(__name__)

class ImagineToAct(torch.nn.Module):
    def __init__(self, imagine_model=None, action_model=None, imagine_model_weights=None, use_flow=True, input_type="rgb_static"):
        super().__init__()

        logger.info(f"Initializing {__class__.__name__}.")
        self.imagine_model = imagine_model
        self.action_model = action_model

        self.use_all_layer = True
        model = imagine_model
        self.tvp_encoder = Diffusion_feature_extractor(pipeline=model.pipeline)
        
        self.extract_layer_idx = extract_layer_idx = 1
        condition_dim_list = [1280,1280,1280,640]
        sum_dim = 0
        for i in range(extract_layer_idx+1):
            sum_dim = sum_dim + condition_dim_list[i+1]
        condition_dim = 1024 # condition_dim_list[extract_layer_idx+1] if not self.use_all_layer else sum_dim
        # print(condition_dim)
        
        self.video_former = Video_Former_3D(
            dim=384,
            depth=6,
            # condition_dim=1024,
            num_frame=4,
            num_time_embeds=4,
            num_latents=224,
            condition_dim=condition_dim,

        )
        self.imagine_model_weights = imagine_model_weights
        self.use_flow = use_flow
        self.input_type = imagine_model.input_type
        
        for p in self.imagine_model.parameters():
            p.requires_grad = False
        
        total_params = sum(p.numel() for p in self.parameters())
        total_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {format_size(total_params)}, Trainable parameters: {format_size(total_trainable_params)}")

        # self.load_weights()
        self.reset()

    def reset(self):
        self.precalc_actions = None 

    def load_weights(self):
        if self.imagine_model_weights is not None:
            if not os.path.exists(self.imagine_model_weights):
                logger.warning(f"Imagine model weights file {self.imagine_model_weights} does not exist. Skipping loading weights.")
            else:
                logger.info(f"Loading imagine model weights from {self.imagine_model_weights}.")
                logger.info(self.imagine_model.load_state_dict(torch.load(self.imagine_model_weights, map_location="cpu"), strict=False))
        
    def forward(self, batch_data, gen_flow=False, split="train", **kwargs):
        # Imagine  
        imagined_output = None
        with torch.no_grad():
            self.imagine_model.eval()
            b, t, c, h, w = batch_data[self.input_type].shape
            norm_flow = torch.zeros((b, 2, h, w), device=batch_data[self.input_type].device)
            flow = norm_flow 

            goal = self.imagine_model.encode_text(batch_data["language"])
            imgs = norm_rgb = batch_data[self.input_type][:, :1]
            perceptual_features = self.tvp_encoder(imgs, goal, self.imagine_model.num_inference_steps, self.extract_layer_idx)
            perceptual_features = einops.rearrange(perceptual_features, 'b f c h w-> b f c (h w)')
            perceptual_features = einops.rearrange(perceptual_features, 'b f c l-> b f l c')

            print(goal.shape, perceptual_features.shape)
            visual_input = self.video_former(perceptual_features) 
            x = {
                "visual_input": visual_input,
                "lang_goal": goal,
            }
        
        # Action 
        return_dict = self.action_model(
            x=x,
            labels=batch_data.get("action", None)
        )

        # print(return_dict.keys())

        return_dict.update({
            "total_loss": return_dict.pop("loss", None),
            "generated_flow": flow,
            "action_logits": return_dict["logits"] if "logits" in return_dict else None,
        })

        return return_dict

    @torch.no_grad()
    def step(self, data, visualize=True):
        if self.precalc_actions is None or self.precalc_actions["action_logits"].shape[1] == 0:
            # logger.info("No precalculated actions available, generating new actions.")
            # Generate actions
            actions = self.forward(data, gen_flow=True)
            self.precalc_actions = actions
        
        flow_output = {
            "generated_flow": self.precalc_actions["generated_flow"],
        }
        if visualize:
            vis_image = self.visualize(data, flow_output, inference=True)[0]
        else:
            vis_image = None
        # Use precalculated actions if available
        this_action = self.precalc_actions["action_logits"][0, 0].detach().cpu()
        self.precalc_actions['action_logits'] = self.precalc_actions['action_logits'][:, 1:]
        # logger.info(f"Using precalculated action: {this_action}")
        return {
            "action": this_action,
            "viz_flow": vis_image,
        }
    
    def visualize(self, batch_data, outputs, inference=False):
        return self.imagine_model.visualize(batch_data, outputs, inference=inference)