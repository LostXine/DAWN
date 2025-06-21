import torch
import logging
from humanfriendly import format_size
import random

logger = logging.getLogger(__name__)

class ImagineToAct(torch.nn.Module):
    def __init__(self, imagine_model=None, action_model=None, imagine_model_weights=None):
        super().__init__()

        logger.info(f"Initializing {__class__.__name__}.")
        self.imagine_model = imagine_model

        self.action_model = action_model
        
        self.imagine_model_weights = imagine_model_weights
        
        for p in self.imagine_model.parameters():
            p.requires_grad = False
        
        total_params = sum(p.numel() for p in self.parameters())
        total_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {format_size(total_params)}, Trainable parameters: {format_size(total_trainable_params)}")

        self.reset()

    def reset(self):
        self.precalc_actions = None 

    def load_weights(self):
        if self.imagine_model_weights is not None:
            logger.info(f"Loading imagine model weights from {self.imagine_model_weights}.")
            logger.info(self.imagine_model.load_state_dict(torch.load(self.imagine_model_weights), strict=False))
        
    def forward(self, batch_data, gen_flow=False):
        # Imagine  
        with torch.no_grad():
            self.imagine_model.eval()
            # if not self.training or random.random() < 0.5:
            if True:
                imagined_output = self.imagine_model(batch_data)
                flow = imagined_output["generated_flow"]
            else:
                flow = self.imagine_model.gen_flow(batch_data["rgb_static"])
            norm_rgb = (batch_data["rgb_static"][:, 0] - self.imagine_model.mean) / self.imagine_model.std
            visual_input = torch.cat([
                norm_rgb,
                flow
            ], dim=1)
                
            # print(imagined_output["feats"].shape)
            # visual_input = imagined_output["feats"]
        
        # Action 
        loss = self.action_model(
            x=visual_input,
            labels=batch_data.get("action", None)
        )


        # logger.info(f"GT action: {batch_data['action'][0, 0]}")
        # logger.info(f"PD action: {loss['logits'][0, 0]}")
        return {
            "total_loss": loss["loss"] if "loss" in loss else None,
            "generated_flow": flow,
            "action_logits": loss["logits"],
        }

    @torch.no_grad()
    def step(self, data):
        if self.precalc_actions is None or len(self.precalc_actions) == 0:
            logger.info("No precalculated actions available, generating new actions.")
            # Generate actions
            actions = self.forward(data, gen_flow=True)
            self.precalc_actions = actions["action_logits"][0].detach().cpu()
    
        # Use precalculated actions if available
        this_action = self.precalc_actions[0]
        self.precalc_actions = self.precalc_actions[1:]
        logger.info(f"Using precalculated action: {this_action}")
        return this_action
    
    def visualize(self, batch_data, outputs):
        return self.imagine_model.visualize(batch_data, outputs)