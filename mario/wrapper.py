import cv2
import numpy as np
from gym.core import Wrapper
from gym.spaces import Box, Discrete

from genrl.environments import FrameStack, GymWrapper


class MarioPreprocessing(Wrapper):
    def __init__(
        self, env, frameskip, grayscale, screen_size,
    ):
        super(MarioPreprocessing, self).__init__(env)

        self.frameskip = frameskip
        self.grayscale = grayscale
        self.screen_size = screen_size

        self.episode_reward = 0

        if isinstance(frameskip, int):
            self.frameskip = (frameskip, frameskip + 1)

        if grayscale:
            self.observation_space = Box(
                low=0, high=255, shape=(screen_size, screen_size), dtype=np.uint8,
            )
        else:
            self.observation_space = Box(
                low=0, high=255, shape=(screen_size, screen_size, 3), dtype=np.uint8,
            )

        self._obs_buffer = [
            np.empty(self.env.observation_space.shape[:2], dtype=np.uint8),
            np.empty(self.env.observation_space.shape[:2], dtype=np.uint8),
        ]

    def step(self, action):
        frameskip = np.random.choice(range(*self.frameskip))
        index = 0

        reward = 0
        for timestep in range(frameskip):
            _, step_reward, done, info = self.env.step(action)
            reward += step_reward

            if done:
                break

            if timestep >= frameskip - 2:
                self._get_screen(index)
                index += 1

            self.episode_reward += reward

        return self._get_obs(), reward, done, info

    def reset(self):
        self.env.reset()
        self._get_screen(0)
        self._obs_buffer[1].fill(0)

        return self._get_obs()

    def _get_screen(self, index):
        self._obs_buffer[index] = np.dot(
            self.env.screen[..., :3], [0.299, 0.587, 0.114]
        )

    def _get_obs(self):
        np.maximum(self._obs_buffer[0], self._obs_buffer[1], out=self._obs_buffer[0])

        obs = cv2.resize(
            self._obs_buffer[0],
            (self.screen_size, self.screen_size),
            interpolation=cv2.INTER_AREA,
        )

        return np.array(obs, dtype=np.uint8)


class MarioWrapper(Wrapper):
    def __init__(self, env):
        super(MarioWrapper, self).__init__(env)

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        self.episode_reward += reward
        return next_state, reward, done, info

    def reset(self):
        state = self.env.reset()
        self.episode_reward = 0
        return state

    @property
    def obs_shape(self):
        if isinstance(self.observation_space, Discrete):
            obs_shape = (1,)
        elif isinstance(self.observation_space, Box):
            obs_shape = self.observation_space.shape
        return obs_shape

    @property
    def action_shape(self):
        if isinstance(self.action_space, Box):
            action_shape = self.action_space.shape
        elif isinstance(self.action_space, Discrete):
            action_shape = (1,)
        return action_shape


def MarioEnv(env):
    env = GymWrapper(env)
    env = MarioPreprocessing(env, 4, True, 84)
    env = FrameStack(env)
    env = MarioWrapper(env)

    return env
