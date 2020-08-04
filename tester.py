from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from mario.wrapper import MarioEnv
from genrl import DQN
from mario.trainer import Trainer


env = gym_super_mario_bros.make("SuperMarioBros-v2")
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = MarioEnv(env)

agent = DQN("cnn", env)
trainer = Trainer(agent, env, log_interval=1, epochs=10, steps_per_epoch=5000, render=True)

trainer.train()
