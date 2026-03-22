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
        intrinsic_reset_threshold: float = -1
    ):
        super().__init__(env, icm, device, intrinsic_reset_threshold)
        self.intrinsic_model: ICMModel = self.intrinsic_model

    def _get_intrinsic_reward_from_model(self, state, action):
        return self.intrinsic_model.calc_intrinsic_reward(self.prev_state, state, action)
