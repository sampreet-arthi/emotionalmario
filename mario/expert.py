import json
import time
import numpy as np
from typing import Tuple

import gym_super_mario_bros
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
    (4, 2)
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


class ExpertTransitionDataset(Dataset):
    """Dataset of expert moves

    Args:
        session_path (str): path to data
        device (torch.device): device for torch tensors.
        render (bool, optional): Whether to render the data while loading. Defaults to False.
    """
    def __init__(self, session_path: str, device: torch.device, render: bool = False):
    
        self._len = 0
        self.obs = None
        self.a = None
        self._load(session_path, device, render)
    
    def _load(self, session_path: str, device: torch.device, render: bool = False):
        """Load with data

        Args:
            session_path (str): path to data
            device (torch.device): device for torch tensors.
            render (bool, optional): Whether to render the data while loading. Defaults to False.
        """

        print(f"Loading data from {session_path} ...", end=' ')

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

        observations = []
        actions = []

        for action in data["obs"]:
            
            if render:
                env.render()
            obs = torch.tensor(next_state.copy()).to(device)        
            next_state, reward, done, info = env.step(action)
            a = torch.tensor(action).to(device)
            observations.append(obs)
            actions.append(a)
            self._len += 1
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
        
        observations = torch.stack(observations)
        actions = torch.stack(actions)
        if self.obs is None or self.a is None:
            self.obs = observations
            self.a = actions
        else:
            self.obs = torch.cat([self.obs, observations])
            self.a = torch.cat([self.a, actions])

        print(f"Complete!")
        

    def __len__(self) -> int:
        """Get length of dataset"""
        return self._len

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get transition tuple at specific element"""
        return torch

    
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
