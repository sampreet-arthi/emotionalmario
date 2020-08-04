from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


env = gym_super_mario_bros.make("SuperMarioBros-v0")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

state = env.reset()
for timestep in range(100):
    state, reward, done, info = env.step(env.action_space.sample())
    if done:
        state = env.reset()
    env.render()

env.close()
