import json
import time
from typing import Optional, Tuple

import cv2
import gym_super_mario_bros
import numpy as np
import torch
from torch.utils.data import Dataset

_STAGE_ORDER = [
    (1, 1),
    (1, 2),
    (1, 3),
    (2, 2),
    (1, 4),
    (3, 1),
    (4, 1),
    (2, 1),
    (2, 3),
    (2, 4),
    (3, 2),
    (3, 3),
    (3, 4),
    (4, 2),
]


def make_next_stage(world, stage, num):

    if num < len(_STAGE_ORDER):
        world = _STAGE_ORDER[num][0]
        stage = _STAGE_ORDER[num][1]

    else:
        if stage >= 4:
            stage = 1
            if world >= 8:
                world = 1
            else:
                world += 1
        else:
            stage += 1

    return world, stage, "SuperMarioBros-%s-%s-v0" % (str(world), str(stage))


class MarioExpertTransitions(Dataset):
    """Dataset of expert moves on Mario

    Args:
        session_path (str): path to data
        framestack (int): Number of frames to stack for each observation.
            Defaults to 4
        grayscale (bool): Whether to convert frames to grayscale.
            Defaults to True
        max_pool (bool): Whether to pool consecutive frames into one.
            Defaults to True.
        screen_size (int): Size of screen. Defaults to 84.
        device (str): device for torch tensors. Defaults to cpu.
        render (bool, optional): Whether to render the data while loading. Defaults to False.
        length (int, optional): Number of actions to consider.
            Defaults to None which implies complete dataset to be loaded
    """

    def __init__(
        self,
        session_path: str,
        framestack: int = 4,
        grayscale: bool = True,
        max_pool: bool = True,
        screen_size: int = 84,
        device: torch.device = "cpu",
        render: bool = False,
        length: Optional[int] = None,
    ):
        self._len = 0
        self.obs = None
        self.a = None
        self.screen_size = screen_size

        print(f"Loading data from {session_path} ...", end=" ")

        with open(session_path) as json_file:
            data = json.load(json_file)

        first_world = "SuperMarioBros-1-1-v0"
        env = gym_super_mario_bros.make(first_world)

        next_state = env.reset()

        world = 1
        stage = 1
        stage_num = 0

        frame_number = 1

        steps = 0

        observations = [
            torch.zeros(self.screen_size, self.screen_size, 3)
            .to(device)
            .to(torch.float)
        ]
        frame_stacked_obs = []
        max_pooled_obs = []
        max_pooled_framestaked_obs = []

        actions = []

        for i, action in enumerate(data["obs"]):
            if i >= length:
                break

            if render:
                env.render()

            next_state = cv2.resize(
                next_state,
                (self.screen_size, self.screen_size),
                interpolation=cv2.INTER_AREA,
            )
            obs = torch.tensor(next_state.copy()).to(device).to(torch.float)
            observations.append(obs)
            next_state, reward, done, info = env.step(action)
            a = torch.tensor(action).to(device)

            if max_pool:
                max_pooled_obs.append(torch.max(observations[-1], observations[-2]))

            if len(max_pooled_obs) >= framestack:
                if max_pool:
                    max_pooled_framestaked_obs.append(
                        torch.stack(max_pooled_obs[-framestack:])
                    )
                else:
                    frame_stacked_obs.append(torch.stack(observations[-framestack:]))

            actions.append(a)
            steps += 1

            is_first = True
            frame_number += 1

            if info["flag_get"]:
                finish = True

            if done:
                done = False
                end = time.time()

                if finish or steps >= 16000:
                    stage_num += 1
                    world, stage, new_world = make_next_stage(world, stage, stage_num)
                    env.close()
                    env = gym_super_mario_bros.make(new_world)
                    finish = False
                    steps = 0

                next_state = env.reset()

        if max_pool:
            observations = torch.stack(max_pooled_framestaked_obs)
        else:
            observations = torch.stack(frame_stacked_obs[1:])

        if grayscale:
            factor = torch.tensor([0.299, 0.587, 0.114]).to(device)
            observations = torch.matmul(observations, factor)

        if self.obs is None or self.a is None:
            self.obs = observations
            self.a = actions
        else:
            self.obs = torch.cat([self.obs, observations])
            self.a = torch.cat([self.a, actions])

        print(f"Complete!")

    def __len__(self) -> int:
        """Get length of dataset"""
        return len(self.obs)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get transition tuple at specific element"""
        return self.obs[index], self.a[index]

    def add(self, obs: torch.Tensor, a: torch.Tensor):
        """Add a transition tuple to dataset

        Args:
            obs (torch.Tensor): observation
            a (torch.Tensor): action taken for observations
        """
        if self.obs is not None:
            self.obs = torch.cat([self.obs, obs])
        else:
            self.obs = obs.unsqueeze(0)
        if self.a is not None:
            self.a = torch.cat([self.a, a])
        else:
            self.a = a.unsqueeze(0)
        self._len += 1

    def get_data(self, shuffle: bool = True):
        if shuffle is None:
            idx = list(np.random.permutation(self._len))
        else:
            idx = [i for i in range(self._len)]
        return self.obs[idx], self.a[idx]
