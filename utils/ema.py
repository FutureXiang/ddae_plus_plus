# Modified from https://github.com/crowsonkb/k-diffusion

from copy import deepcopy
import torch


class EMA(torch.nn.Module):
    """Implements an EMA warmup using an inverse decay schedule.
    If inv_gamma=1 and power=1, implements a simple average. inv_gamma=1, power=2/3 are
    good values for models you plan to train for a million or more steps (reaches decay
    factor 0.999 at 31.6K steps, 0.9999 at 1M steps), inv_gamma=1, power=3/4 for models
    you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999 at
    215.4k steps).
    """

    def __init__(self, model, beta, inv_gamma=1.0, power=2/3, min_value=0.0):
        super().__init__()
        self.beta = beta
        self.model = model

        self.ema_model = deepcopy(model)
        self.ema_model.requires_grad_(False)

        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value

        self.register_buffer('step', torch.tensor(0))

    def get_current_decay(self):
        if self.step.item() <= 0:
            return 0.

        value = 1 - (1 + self.step / self.inv_gamma) ** - self.power
        return value.clamp(min = self.min_value, max = self.beta).item()

    @torch.no_grad()
    def update(self):
        """Incorporates updated model parameters into an exponential moving averaged
        version of a model. It should be called after each optimizer step.
        """
        decay = self.get_current_decay()

        model_params = dict(self.model.named_parameters())
        averaged_params = dict(self.ema_model.named_parameters())
        assert model_params.keys() == averaged_params.keys()

        for name, param in model_params.items():
            if torch.isinf(param).any() or torch.isnan(param).any():
                print("Warning: inf/nan detected in online model param, ignored in EMA update.")
                continue
            else:
                averaged_params[name].lerp_(param, 1 - decay)

        model_buffers = dict(self.model.named_buffers())
        averaged_buffers = dict(self.ema_model.named_buffers())
        assert model_buffers.keys() == averaged_buffers.keys()

        for name, buf in model_buffers.items():
            if torch.isinf(buf).any() or torch.isnan(buf).any():
                print("Warning: inf/nan detected in online model buf, ignored in EMA update.")
                continue
            else:
                averaged_buffers[name].copy_(buf)

        self.step += 1

    def __call__(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)
    
    def eval(self):
        return self.ema_model.eval()
