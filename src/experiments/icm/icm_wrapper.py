from torch import device

from curiosity_gym.core.gridengine import GridEngine
from experiments.components import IntrinsicMotivationModelWrapper
from .icm_model import ICMModel

class ICMCuriosityWrapper(IntrinsicMotivationModelWrapper):
    def __init__(
        self,
        device: device | str,
        env: GridEngine,
        icm: ICMModel,
        intrinsic_reset_threshold: float = 0.5,
        allow_global_state_reset: bool = False,
        max_training_steps: int = 500,
        max_episodes: int = 1000
    ):
        super().__init__(env, icm, device, intrinsic_reset_threshold, allow_global_state_reset, max_training_steps, max_episodes)
        self.intrinsic_model: ICMModel = self.intrinsic_model

    def _get_intrinsic_reward_from_model(self, state, action):
        int_reward = self.intrinsic_model.calc_intrinsic_reward(self.prev_state, state, action)
        self.intrinsic_model._train_network(self.prev_state, state, action)
        return int_reward

    def _get_intrinsic_reward_from_model_no_training(self, state, action):
        return self.intrinsic_model.calc_intrinsic_reward(self.prev_state, state, action)
