import gym
import numpy as np
import torch

from genrl.deep.common.rollout_storage import RolloutBuffer


class MarioRollout(RolloutBuffer):
    def __init__(
        self, buffer_size, env, device="cpu", gae_lambda=1, gamma=0.99,
    ):
        super(MarioRollout, self).__init__(
            buffer_size, env, device, gae_lambda, gamma
        )

    def reset(self):
        self.observations = np.zeros(
            (self.buffer_size,) + self.env.obs_shape, dtype=np.float32
        )
        self.actions = np.zeros(
            (self.buffer_size,) + self.env.action_shape, dtype=np.float32,
        )
        self.rewards = np.zeros((self.buffer_size,), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size,), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size,), dtype=np.float32)
        self.values = np.zeros((self.buffer_size,), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size,), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size,), dtype=np.float32)
        self.generator_ready = False

        self.pos = 0
        self.full = False

    def get(self, batch_size=None):
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size)
        # Prepare the data
        if not self.generator_ready:
            # for tensor in [
            #     "observations",
            #     "advantages",
            #     "returns",
            # ]:
            #     print(tensor)
            #     self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size

        start_idx = 0
        while start_idx < self.buffer_size:
            yield self._get_samples(
                indices[start_idx : start_idx + batch_size]
            )
            start_idx += batch_size
