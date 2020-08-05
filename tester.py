import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

from genrl import DQN
from mario.trainer import Trainer
from mario.wrapper import MarioEnv

env = gym_super_mario_bros.make("SuperMarioBros-v2")
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = MarioEnv(env)

agent = DQN("cnn", env, replay_size=100000, epsilon_decay=100000)
trainer = Trainer(agent, env, log_interval=1, epochs=10, steps_per_epoch=5000)

trainer.train()
trainer.evaluate()
