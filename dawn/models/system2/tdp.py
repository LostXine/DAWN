import torch
from torch import nn
import logging
from functools import partial
from tqdm import trange

logger = logging.getLogger(__name__)

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
    return x[(...,) + (None,) * dims_to_append]

@torch.no_grad()
def sample_ddim(
    model, 
    state, 
    action, 
    goal, 
    sigmas, 
    scaler=None,
    extra_args=None, 
    callback=None, 
    disable=None, 
    eta=1., 
):
    """
    DPM-Solver 1( or DDIM sampler"""
    extra_args = {} if extra_args is None else extra_args
    s_in = action.new_ones([action.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    old_denoised = None

    for i in trange(len(sigmas) - 1, disable=disable):
        # predict the next action
        denoised = model(state, action, goal, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'action': action, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        action = (sigma_fn(t_next) / sigma_fn(t)) * action - (-h).expm1() * denoised
    return action

from torch import nn
from .utils import append_dims
from dawn.models.module.diffusion_decoder import DiffusionTransformer

class TransformerDiffusionPolicy(nn.Module):
    def __init__(self, action_dim, obs_dim, goal_dim, num_tokens, goal_window_size, obs_seq_len, act_seq_len, device, sigma_data=1., proprio_dim=8):
        super().__init__()
        self.inner_model = DiffusionTransformer(
            action_dim = action_dim,
            obs_dim = obs_dim,
            goal_dim = goal_dim,
            proprio_dim= proprio_dim,
            goal_conditioned = True,
            embed_dim = 384,
            n_dec_layers = 4,
            n_enc_layers = 4,
            n_obs_token = num_tokens,
            goal_seq_len = goal_window_size,
            obs_seq_len = obs_seq_len,
            action_seq_len =act_seq_len,
            embed_pdrob = 0,
            goal_drop = 0,
            attn_pdrop = 0.3,
            resid_pdrop = 0.1,
            mlp_pdrop = 0.05,
            n_heads= 8,
            device = device,
            use_mlp_goal = True,
        )
        self.sigma_data = sigma_data

    def forward(
        self, 
        embedding, 
        latent_goal, 
        actions
    ):
        sigmas = self.make_sample_density()(shape=(len(actions),), device=self.device).to(self.device)
        noise = torch.randn_like(actions).to(self.device)
        loss, _ = self.loss(perceptual_emb, actions, latent_goal, noise, sigmas)

        outputs = {
            "loss": loss,

        }
        return outputs
    
    def loss(self, state, action, goal, noise, sigma, **kwargs):
        c_skip, c_out, c_in = [append_dims(x, action.ndim) for x in self.get_scalings(sigma)]
        noised_input = action + noise * append_dims(sigma, action.ndim)
        model_output = self.inner_model(state, noised_input * c_in, goal, sigma, **kwargs)
        target = (action - c_skip * noised_input) / c_out
        return (model_output - target).pow(2).flatten(1).mean(), model_output 

    def make_sample_density(self):
        """
        Generate a sample density function based on the desired type for training the model
        We mostly use log-logistic as it has no additional hyperparameters to tune.
        """
        sd_config = []
        if self.sigma_sample_density_type == 'lognormal':
            loc = self.sigma_sample_density_mean  # if 'mean' in sd_config else sd_config['loc']
            scale = self.sigma_sample_density_std  # if 'std' in sd_config else sd_config['scale']
            return partial(utils.rand_log_normal, loc=loc, scale=scale)

        if self.sigma_sample_density_type == 'loglogistic':
            loc = sd_config['loc'] if 'loc' in sd_config else math.log(self.sigma_data)
            scale = sd_config['scale'] if 'scale' in sd_config else 0.5
            min_value = sd_config['min_value'] if 'min_value' in sd_config else self.sigma_min
            max_value = sd_config['max_value'] if 'max_value' in sd_config else self.sigma_max
            return partial(utils.rand_log_logistic, loc=loc, scale=scale, min_value=min_value, max_value=max_value)

        if self.sigma_sample_density_type == 'loguniform':
            min_value = sd_config['min_value'] if 'min_value' in sd_config else self.sigma_min
            max_value = sd_config['max_value'] if 'max_value' in sd_config else self.sigma_max
            return partial(utils.rand_log_uniform, min_value=min_value, max_value=max_value)

        if self.sigma_sample_density_type == 'uniform':
            return partial(utils.rand_uniform, min_value=self.sigma_min, max_value=self.sigma_max)

        if self.sigma_sample_density_type == 'v-diffusion':
            min_value = self.min_value if 'min_value' in sd_config else self.sigma_min
            max_value = sd_config['max_value'] if 'max_value' in sd_config else self.sigma_max
            return partial(utils.rand_v_diffusion, sigma_data=self.sigma_data, min_value=min_value, max_value=max_value)
        if self.sigma_sample_density_type == 'discrete':
            sigmas = self.get_noise_schedule(self.num_sampling_steps * 1e5, 'exponential')
            return partial(utils.rand_discrete, values=sigmas)
        if self.sigma_sample_density_type == 'split-lognormal':
            loc = sd_config['mean'] if 'mean' in sd_config else sd_config['loc']
            scale_1 = sd_config['std_1'] if 'std_1' in sd_config else sd_config['scale_1']
            scale_2 = sd_config['std_2'] if 'std_2' in sd_config else sd_config['scale_2']
            return partial(utils.rand_split_log_normal, loc=loc, scale_1=scale_1, scale_2=scale_2)
        else:
            raise ValueError('Unknown sample density type')

    def get_scalings(self, sigma):
        """
        Compute the scalings for the denoising process.

        Args:
            sigma: The input sigma.
        Returns:
            The computed scalings for skip connections, output, and input.
        """
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        return c_skip, c_out, c_in