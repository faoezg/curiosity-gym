import torch
import numpy as np

# following Z. Guo et. al. 2022, Appendix A.3
class RewardNormaliser:
    def __init__(self, decay: float = 0.995, eps: float = 1e-6):
        self.decay = decay
        self.eps = eps
        self.mean = 0.0
        self.sq_mean = 0.0
        self.c = 1

    def __call__(self, raw_tensor: torch.Tensor) -> torch.Tensor:
        batch_mean = raw_tensor.mean().item()
        batch_squared_mean   = (raw_tensor ** 2).mean().item()

        self._ema_update(batch_mean, batch_squared_mean)

        mu, mu_squared = self._adjust_ema()
        

        std = np.sqrt(max(mu_squared - mu ** 2, 0.0) + self.eps)

        return raw_tensor / std

    def _ema_update(self, batch_mean: float, batch_squared_mean: float):
        self.mean = self.decay * self.mean + (1 - self.decay) * batch_mean
        self.sq_mean = self.decay * self.sq_mean + (1 - self.decay) * batch_squared_mean
        self.c += 1

    def _adjust_ema(self):
        mu = self.mean / (1 - self.decay ** self.c)
        mu_squared = self.sq_mean / (1 - self.decay ** self.c)
        return mu, mu_squared
