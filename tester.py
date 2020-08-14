from mario.base.wrapper import MarioEnv
from mario.adversarial.expert_env import MarioExpertEnv
from genrl.environments import FrameStack, GymWrapper
from mario.base.wrapper import MarioPreprocessing

# from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
# from nes_py.wrappers import JoypadSpace

env = MarioExpertEnv("../toadstool", file_num=1)
# env = JoypadSpace(env, COMPLEX_MOVEMENT)
env = GymWrapper(env)
env = MarioPreprocessing(env)
env = FrameStack(env)

state = env.reset()
for t in range(10000):
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    if True:
        env.render()

    if done:
        state = env.reset()
