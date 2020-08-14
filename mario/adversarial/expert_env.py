import os
import json
import gym

import torch

import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


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


class MarioExpertEnv(gym.Wrapper):
    def __init__(self, root_dir, env_id="SuperMarioBros-1-1-v0", file_num=0):
        self.root_dir = root_dir

        self.world = 1
        self.stage = 1
        self.stage_num = 0
        self.frame_number = 0
        self.step_count = 0

        env = gym_super_mario_bros.make(env_id)
        self.env = env

        if file_num is None:
            self._load_all()
        else:
            self._load(file_num)

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    def __getattr__(self, name):
        env = super().__getattribute__("env")
        return getattr(env, name)

    def _load_all(self):
        raise NotImplementedError

    def _load(self, file_num):
        file_path = "participants/participant_{}/participant_{}_session.json".format(file_num, file_num)
        path = os.path.join(self.root_dir, file_path)

        with open(path) as f:
            data = json.load(f)

        self.trajectory = torch.LongTensor(data["obs"])

    def step(self, action):
        action = self.trajectory[self.step_count].item()
        next_state, reward, done, info = self.env.step(action)
        self.step_count += 1

        if info["flag_get"]:
            finish = True
        else:
            finish = False

        if done:
            done = False
            
            if finish or self.step_count >= 16000:
                self.stage_num += 1
                self.world, self.stage, self.new_world = make_next_stage(self.world, self.stage, self.stage_num)
                self.env.close()
                self.env = gym_super_mario_bros.make(self.new_world)
                finish = False
                self.step_count = 0

            next_state = self.reset()

        return next_state, reward, done, info

    def reset(self):
        state = self.env.reset()
        return state
