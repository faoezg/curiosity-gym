import numpy as np

class RewardNormalizer():
    def __init__(self) -> None:
        self.eps = 1e-8
        self.reset()

    def normalize_reward(self, reward: float):
        self._update_statistics(reward)
        return (reward - self.reward_mean) / np.sqrt(self.variance() + self.eps)
    
    def _update_statistics(self, reward: float):
        self.reward_count += 1
        delta = self._calc_delta(reward)
        self.reward_mean += delta / self.reward_count
        delta2 = self._calc_delta(reward)
        self.reward_var += delta * delta2
        
    def _calc_delta(self, reward: float):
        return reward - self.reward_mean
    
    def variance(self):
        if (self.reward_count == 0):
            return 0.0
        return self.reward_var / self.reward_count

    def reset(self):
        self.reward_mean = 0
        self.reward_var = 0.0
        self.reward_count = 0
