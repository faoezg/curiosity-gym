from experiments.components import IntrinsicMotivationModelWrapper
from curiosity_gym.core.gridengine import GridEngine
from torch import device
from .coin_flip_model import CoinFlipModel

class CoinFlipWrapper(IntrinsicMotivationModelWrapper):
    def __init__(
        self,
        device: device,
        env: GridEngine,
        cfm: CoinFlipModel,
        intrinsic_reset_threshold: float = 0.5,
        allow_global_state_reset: bool = False,
        max_training_steps: int = 500,
        max_episodes: int = 1000,
        buffer_size: int = 1,
        ):
        super().__init__(env, cfm, device, intrinsic_reset_threshold, allow_global_state_reset, max_training_steps, max_episodes, buffer_size=buffer_size)
        self.intrinsic_model: CoinFlipModel = self.intrinsic_model
    
    def reset(self, **kwargs):
        self.intrinsic_model.reset()
        return super().reset(**kwargs)

    def _get_intrinsic_reward_from_model(self, state, action):
        return self.intrinsic_model.calc_intrinsic_reward(state, action)

    def _get_intrinsic_reward_from_model_no_training(self, state, action):
        return self.intrinsic_model.calc_intrinsic_reward(state, action, with_training=False)
