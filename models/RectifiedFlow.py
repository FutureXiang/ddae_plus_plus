import torch
import torch.nn as nn
from torchdiffeq import odeint
from scipy.stats import norm
import numpy as np


def sample_logit_normal(mu, sigma, size=1):
    # Generate samples from the normal distribution
    samples = norm.rvs(loc=mu, scale=sigma, size=size)
    
    # Transform samples to be in the range (0, 1) using the logistic function
    samples = 1 / (1 + np.exp(-samples))

    # Numpy to Tensor
    samples = torch.tensor(samples, dtype=torch.float32)
    return samples


class RectifiedFlow(nn.Module):
    def __init__(self, device, use_lognorm=False):
        ''' RectifiedFlow (or OT-CFM) proposed by \
            "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow", and \
            "Flow Matching for Generative Modeling".

            Args:
                device: The CUDA device that tensors run on.
            Training parameters:
                use_lognorm
        '''
        super().__init__()
        self.device = device
        self.use_lognorm = use_lognorm

    def perturb(self, x, t=None):
        ''' Add noise to a clean image (diffusion process).

            Args:
                x: The image tensor.
                t: The specified timestep ranged in `[0, 1]`. Type: float / torch.Tensor / None.
            Returns:
                The perturbed image, the corresponding t, and the noise.
        '''
        if t is None:
            if self.use_lognorm:
                t = sample_logit_normal(mu=0, sigma=1, size=x.shape[0]).to(self.device)
            else:
                t = torch.rand((x.shape[0], )).to(self.device)
        else:
            if isinstance(t, float) or len(t.shape) == 0:
                t = torch.tensor([t]).to(self.device).repeat(x.shape[0])

        noise = torch.randn_like(x)
        x_noised = x * t[:, None, None, None] + noise * (1 - t)[:, None, None, None]
        return x_noised, t, noise

    def forward(self, model, x, other_kwargs={}):
        ''' Training with velocity prediction loss + cosine loss, proposed by \
            "FasterDiT: Towards Faster Diffusion Transformers Training without Architecture Modification", and \
            "Reconstruction vs. Generation: Taming Optimization Dilemma in Latent Diffusion Models".
        '''
        x_noised, t, noise = self.perturb(x, t=None)

        v_pred = model(x_noised, t, **other_kwargs)
        v = x - noise
        mse_loss = (v - v_pred) ** 2
        cos_loss = 1 - nn.functional.cosine_similarity(v, v_pred, dim=1)
        return {'loss': mse_loss, 'cos_loss': cos_loss}

    def get_feature(self, model, x, t, other_kwargs={}, clean=False, use_gap=True):
        ''' Get network's intermediate activation in a forward pass.

            Args:
                x: The clean image tensor ranged in `[0, 1]`.
                t: The specified timestep ranged in `[0, 1]`. Type: float / torch.Tensor.
                clean: to use clean image instead of noised one (for fine tuning).
            Returns:
                A {name: tensor} dict which contains global average pooled features.
        '''
        x_noised, t, noise = self.perturb(x, t)

        def gap(act):
            assert len(act.shape) == 3
            # vit (B, L, C)
            return torch.mean(act.float(), dim=1)

        _, acts = model(x if clean else x_noised, t, **other_kwargs)
        if not use_gap:
            return acts
        all_feats = {blockname: gap(acts[blockname]) for blockname in acts}
        return all_feats

    def sample(self, model, x, other_kwargs={},
               sampler_type="euler",
               num_steps=250,
               timestep_shift=0.0,
               return_nfe=False):

        t = torch.linspace(0.0, 1.0, num_steps)

        if timestep_shift > 0:
            # Eq. 23 in Stable Diffusion 3 paper
            # However, since our notation (1 = data, 0 = noise) is inverted,
            # alpha_ours should be 1 / alpha_original (e.g., 3.0 -> 0.333)
            def compute_tm(t_n, alpha):
                numerator = alpha * t_n
                denominator = 1 + (alpha - 1) * t_n
                return numerator / denominator

            t = torch.tensor([compute_tm(t_n, timestep_shift) for t_n in t])

        def func(t, x):
            func.nfe += 1
            v_pred = model(x, t.repeat(x.shape[0]), **other_kwargs)
            return v_pred

        func.nfe = 0
        x = odeint(
            func,
            x,
            t.to(self.device),
            method=sampler_type,
            atol=1e-6,
            rtol=1e-3,
        )
        if return_nfe:
            return x[-1], func.nfe
        else:
            return x[-1]
