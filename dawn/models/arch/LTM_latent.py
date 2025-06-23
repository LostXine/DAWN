import torch
import logging
from humanfriendly import format_size
import random

logger = logging.getLogger(__name__)

class ImagineToAct(torch.nn.Module):
    def __init__(self, imagine_model=None, action_model=None, imagine_model_weights=None, use_flow=True):
        super().__init__()

        logger.info(f"Initializing {__class__.__name__}.")
        self.imagine_model = imagine_model

        self.action_model = action_model
        
        self.imagine_model_weights = imagine_model_weights
        self.use_flow = use_flow

        for p in self.imagine_model.parameters():
            p.requires_grad = False
        
        total_params = sum(p.numel() for p in self.parameters())
        total_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {format_size(total_params)}, Trainable parameters: {format_size(total_trainable_params)}")

        self.load_weights()
        self.reset()

    def reset(self):
        self.precalc_actions = None 
        self.cnt = 10

    def load_weights(self):
        if self.imagine_model_weights is not None:
            logger.info(f"Loading imagine model weights from {self.imagine_model_weights}.")
            logger.info(self.imagine_model.load_state_dict(torch.load(self.imagine_model_weights, map_location="cpu"), strict=False))
        
    def forward(self, batch_data, gen_flow=False, split="train"):
        # Imagine  
        imagined_output = None
        with torch.no_grad():
            self.imagine_model.eval()
            if not self.use_flow:
                b, t, c, h, w = batch_data["rgb_static"].shape
                norm_flow = torch.zeros((b, 2, h, w), device=batch_data["rgb_static"].device)
                flow = norm_flow 
            else:
            # if not self.training or random.random() < 0.5:
                if split != "train" or gen_flow:
                    imagined_output = self.imagine_model(batch_data)
                    flow = imagined_output["generated_flow"]
                else:
                    flow = self.imagine_model.gen_flow(batch_data["rgb_static"])
                    # Add some noise to the flow to adapt with the generated flow 
                    noise = torch.randn_like(flow) / self.imagine_model.image_size / 2
                    p = torch.rand(flow.shape[0], device=flow.device) < 0.5
                    flow = flow + noise * p[:, None, None, None]  # Add noise only to some samples
                norm_flow = flow * 2 - 1 
            norm_rgb = batch_data["rgb_static"][:, 0]
            visual_input = torch.cat([
                norm_rgb,
                norm_flow
            ], dim=1)
            goal = self.imagine_model.encode_text(batch_data["language"])

            x = {
                "visual_input": visual_input,
                "lang_goal": goal,
            }
                
            # print(imagined_output["feats"].shape)
            # visual_input = imagined_output["feats"]
        
        # Action 
        loss = self.action_model(
            x=x,
            labels=batch_data.get("action", None)
        )

        # logger.info(f"GT action: {batch_data['action'][0, 0]}")
        # logger.info(f"PD action: {loss['logits'][0, 0]}")
        return {
            "total_loss": loss["loss"] if "loss" in loss else None,
            "generated_flow": flow,
            "action_logits": loss["logits"] if "logits" in loss else None,
        }

    @torch.no_grad()
    def step(self, data):
        if self.cnt == 0 or self.precalc_actions is None or self.precalc_actions["action_logits"].shape[1] == 0:
            # logger.info("No precalculated actions available, generating new actions.")
            # Generate actions
            actions = self.forward(data, gen_flow=True)
            self.precalc_actions = actions
        
        self.cnt -= 1 
        flow_output = {
            "generated_flow": self.precalc_actions["generated_flow"],
        }
        vis_image = self.visualize(data, flow_output, inference=True)[0]
        
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