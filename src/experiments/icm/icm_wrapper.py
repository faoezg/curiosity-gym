from torch import device

from curiosity_gym.core.gridengine import GridEngine
from experiments.components import IntrinsicMotivationModelWrapper, RewardNormalizer
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
        max_episodes: int = 1000,
        use_rgb_step: bool = False,
        rank: int | None = None
    ):
        super().__init__(env, icm, device, intrinsic_reset_threshold, allow_global_state_reset, max_training_steps, max_episodes, use_rgb_step)
        self.intrinsic_model: ICMModel = self.intrinsic_model
        self.batch_size = 128 
        self.reward_normalizer = RewardNormalizer()
    
    def reset(self, **kwargs):
        self.reward_normalizer.reset()
        return super().reset(**kwargs)

    def _get_intrinsic_reward_from_model(self, state, action):
        int_reward = self.intrinsic_model.calc_intrinsic_reward(self.prev_state, state, action)
        inv_loss = 0
        forward_loss = 0
        if (self.total_training_step % 12 == 0 and len(self.replay_buffer) >= self.batch_size):
            batch = self._sample_batch(self.batch_size)
            inv_loss, forward_loss = self.intrinsic_model._train_network_with_batch(batch)
            self.writer.add_scalar("Loss/Inv_Modell", inv_loss, self.total_training_step)
            self.writer.add_scalar("Loss/Forw_Modell", forward_loss, self.total_training_step)

        #if (len(self.rollout_buffer) >= 20):
        #    inv_loss, forward_loss = self.intrinsic_model._train_network_with_batch(self._get_rollout())
        #    self.writer.add_scalar("Loss/Inv_Modell", inv_loss, self.total_training_step)
        #    self.writer.add_scalar("Loss/Forw_Modell", forward_loss, self.total_training_step)

        #inv_loss, forward_loss = self.intrinsic_model._train_network(self.prev_state, state, action)

        #return self.reward_normalizer.normalize_reward(int_reward)
        return int_reward

    def _get_intrinsic_reward_from_model_no_training(self, state, action):
        return self.intrinsic_model.calc_intrinsic_reward(self.prev_state, state, action)
