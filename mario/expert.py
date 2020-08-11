import json
from pathlib import Path
from typing import List, Optional, Tuple
import shutil
import warnings

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


def process_single_session(session_path, output_path=None, render=False, length=None):

    with open(session_path) as json_file:
        data = json.load(json_file)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True, parents=True)
        shutil.copyfile(session_path, output_path.joinpath("data.json"))
        output_path.joinpath("frames").mkdir(exist_ok=True)

    first_world = "SuperMarioBros-1-1-v0"
    env = gym_super_mario_bros.make(first_world)

    next_state = env.reset()

    world = 1
    stage = 1
    stage_num = 0
    frame_number = 0
    steps = 0

    for i, action in enumerate(data["obs"]):

        if length is not None:
            if i >= length:
                break

        if render:
            env.render()

        next_state, _, done, info = env.step(action)
        steps += 1

        if output_path is not None:
            cvt_state = cv2.cvtColor(next_state, cv2.COLOR_BGR2RGB)
            impath = str(output_path.joinpath(f"frames/frame_{frame_number}.png"))
            cv2.imwrite(impath, cvt_state)

        finish = False
        frame_number += 1

        if info["flag_get"]:
            finish = True

        if done:
            done = False

            if finish or steps >= 16000:
                stage_num += 1
                world, stage, new_world = make_next_stage(world, stage, stage_num)
                env.close()
                env = gym_super_mario_bros.make(new_world)
                finish = False
                steps = 0

            next_state = env.reset()


def process_multiple_sessions(
    data_dir: str,
    output_path: str,
    sessions: List[int] = list(range(10)),
    render: bool = False,
    length: Optional[int] = None,
):
    """Load multiple sessions into the dataset

    Args:
        data_dir (str): Path to directory of participants data
        output_path (str): Path to output directory
        sessions (List[int]): list of sessions to consider.
            Defaults to list(range(10)).
        render (bool, optional): Whether to render the data while loading.
            Defaults to False.
        length (int, optional): Number of actions to consider.
            Defaults to None which implies complete dataset to be loaded
    """
    data_dir = Path(data_dir)
    for i in sessions:
        session_path = data_dir.joinpath(
            f"participant_{i}/participant_{i}_session.json"
        )
        output_path = Path(output_path).joinpath(f"participant_{i}")
        if not session_path.exists():
            raise FileNotFoundError(f"{session_path.absolute()} does not exist")
        process_single_session(session_path, output_path, render, length)


class MarioExpertTransitions(Dataset):
    """Dataset of expert moves on Mario

    Args:
        data_path (str): path to data.
        sessions (List[int]): list of sessions to consider.
            Defaults to list(range(10)).
        framestack (int): Number of frames to stack for each observation.
            Defaults to 4
        grayscale (bool): Whether to convert frames to grayscale.
            Defaults to True
        max_pool (bool): Whether to pool consecutive frames into one.
            Defaults to True.
        screen_size (int): Size of screen. Defaults to 84.
        device (str): device for torch tensors. Defaults to cpu.
        length (int, optional): Number of actions to consider.
            Defaults to None which implies complete dataset to be loaded
    """

    def __init__(
        self,
        data_path: str,
        sessions: List[int] = list(range(10)),
        framestack: int = 4,
        grayscale: bool = True,
        max_pool: bool = True,
        screen_size: int = 84,
        device: str = "cpu",
        length: Optional[int] = None,
    ):
        self.framestack = framestack
        self.grayscale = grayscale
        self.max_pool = max_pool
        self.screen_size = screen_size
        self.device = device

        self.obs = None
        self.a = None
        self._load_multiple_sessions(data_path, sessions, length)

    def _load_multiple_sessions(
        self,
        data_dir: str,
        sessions: List[int] = list(range(10)),
        length: Optional[int] = None,
    ):
        """Load multiple sessions into the dataset

        Args:
            session_path (str): Path to directory of participants data
            sessions (List[int]): list of sessions to consider.
                Defaults to list(range(10)).
            length (int, optional): Number of actions to consider.
                Defaults to None which implies complete dataset to be loaded
        """
        data_dir = Path(data_dir)
        for i in sessions:
            session_path = data_dir.joinpath(f"participant_{i}")
            if not session_path.exists():
                warnings.warn(f"{session_path.absolute()} does not exist! Skipping.")
                continue
            self._load_single_session(session_path, length)

    def _load_single_session(self, session_path: str, length: int = None):
        """Load a session into the dataset

        Args:
            session_path (str): path to data
            length (int, optional): Number of actions to consider.
                Defaults to None which implies complete dataset to be loaded
        """

        print(f"Loading data from {session_path} ...", end=" ")

        session_path = Path(session_path)
        session_data_path = session_path.joinpath("data.json")
        with open(session_data_path) as json_file:
            data = json.load(json_file)

        observations = [
            torch.zeros(self.screen_size, self.screen_size, 3)
            .to(self.device)
            .to(torch.float)
        ]
        frame_stacked_obs = []
        max_pooled_obs = []
        max_pooled_framestaked_obs = []

        actions = []

        for i, action in enumerate(data["obs"]):
            if length is not None:
                if i >= length:
                    break

            a = torch.tensor(action).to(self.device)
            actions.append(a)

            obs = cv2.imread(str(session_path.joinpath(f"frames/frame_{i}.png")))
            obs = cv2.resize(
                obs, (self.screen_size, self.screen_size), interpolation=cv2.INTER_AREA,
            )
            obs = torch.tensor(obs.copy()).to(self.device).to(torch.float)
            observations.append(obs)

            if self.max_pool:
                max_pooled_obs.append(torch.max(observations[-1], observations[-2]))

            if len(max_pooled_obs) >= self.framestack:
                if self.max_pool:
                    max_pooled_framestaked_obs.append(
                        torch.stack(max_pooled_obs[-self.framestack :])
                    )
                else:
                    frame_stacked_obs.append(
                        torch.stack(observations[-self.framestack :])
                    )

        if self.max_pool:
            observations = torch.stack(max_pooled_framestaked_obs)
        else:
            observations = torch.stack(frame_stacked_obs[1:])

        if self.grayscale:
            factor = torch.tensor([0.299, 0.587, 0.114]).to(self.device)
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
