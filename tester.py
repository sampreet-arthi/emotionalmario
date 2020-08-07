import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

from genrl.deep.agents import PPO1
from genrl.deep.agents.dqn import DoubleDQN, NoisyDQN, PrioritizedReplayDQN
from mario.trainer import MarioTrainer
from mario.wrapper import MarioEnv

env = gym_super_mario_bros.make("SuperMarioBros-v2")
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = MarioEnv(env)

# agent = DoubleDQN("cnn", env, replay_size=10000, epsilon_decay=10000)
# trainer = MarioTrainer(agent, env, off_policy=True, log_interval=10, epochs=200, max_ep_len=5000)

agent = PPO1("cnn", env, rollout_size=2048)
trainer = MarioTrainer(agent, env, off_policy=False, log_interval=10, epochs=200)

trainer.train()
trainer.evaluate()
