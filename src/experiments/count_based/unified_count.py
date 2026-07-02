from typing import Any, override
from collections import defaultdict
import numpy as np

from experiments.components import IntrinsicMotivationModel, RewardNormalizer

# Given that this uses the actuall count as the pseudo_count, most of this statistics is actually not needed
class UnifiedCountModel(IntrinsicMotivationModel):
    """
    Docstring for UnifiedCountModel
    This model does approximate any functions.
    It is table based given that in a gridworld such curosity gym
    the state-space is discrete.

    """
    def __init__(self,
                 possible_state_dim: int,
                 pseudo_prior: float = 0.01, # ensure non-zero prob, 
                 beta: float = 0.05,
                 eps: float = 0.01,
                 clip_range = 0.02
                 ) -> None:
        self.possible_state_dim = possible_state_dim
        self.prior = pseudo_prior
        self.visitation_dict = defaultdict(int) # ensures that unseen states return 0 when read 
        self.total_visited_states = 0

        self.beta = beta
        self.eps = eps
        self.clip_range = clip_range

        self.reward_normalizer = RewardNormalizer()

        super().__init__("count_based")
    
 
    @override
    def _train_network(self, state: np.ndarray) -> None:
        state_key = state.tobytes()
        self.visitation_dict[state_key] += 1
        self.total_visited_states += 1
    
    def calc_visitation_prob(self, state: np.ndarray) -> float:
        state_key = state.tobytes()
        prob = self._calc_cell_visitation_prob(state_key)
        return prob

    def calc_visitation_prob_after_observation(self, state: np.ndarray) -> float:
        state_key = state.tobytes()
        prob = self._calc_cell_visitation_prob(state_key, 1)
        return prob
    
    def _calc_cell_visitation_prob(self, state: bytes, observation_count: int = 0) -> float:
        pseudo_count = self.visitation_dict[state] + observation_count + self.prior # + prior if state is unvisited
        total_visited_states = self.total_visited_states + observation_count + self.prior

        return pseudo_count / total_visited_states

    @override
    def calc_intrinsic_reward(self, state: np.ndarray) -> float:
        rho = self.calc_visitation_prob(state)
        rho_after = self.calc_visitation_prob_after_observation(state)

        if (rho_after <= rho):
            pseudo_count = 0.0
        else:
            pseudo_count = (rho * (1.0 - rho_after) / (rho_after - rho))

        intrinsic_reward = self.beta / np.sqrt(pseudo_count + self.eps)
        if (intrinsic_reward <= self.clip_range):
            return 0.0
        else:
            #return self.reward_normalizer.normalize_reward(intrinsic_reward)
            return intrinsic_reward
    
    def reset(self):
        self.reward_normalizer.reset()
