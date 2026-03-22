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
        intrinsic_reset_threshold: float = 0.5
    ):
        super().__init__(env, cfm, device, intrinsic_reset_threshold)
        self.intrinsic_model: CoinFlipModel = self.intrinsic_model
    
    def reset(self, **kwargs):
        self.intrinsic_model.reset()
        return super().reset(**kwargs)

    def _get_intrinsic_reward_from_model(self, state, action):
        return self.intrinsic_model.calc_intrinsic_reward(state, action)
