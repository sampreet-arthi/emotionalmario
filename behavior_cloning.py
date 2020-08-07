import argparse

import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

from genrl import DQN
from mario.supervised import SupervisedTrainer
from mario.wrapper import MarioEnv

argument_parser = argparse.ArgumentParser(
    description="A script used to clone expert data into agent."
)
argument_parser.add_argument("-i", "--input-path", type=str, required=True)
argument_parser.add_argument("-e", "--epochs", type=int, default=10)
argument_parser.add_argument("--lr", type=float, default=1e-3)
argument_parser.add_argument("-b", "--batch-size", type=int, default=64)
argument_parser.add_argument("--length", type=int, default=None)
argument_parser.add_argument("-r", "--render", action="store_true")
argument_parser.add_argument("--enable-cuda", action="store_true")
args = argument_parser.parse_args()

if args.enable_cuda:
    device = "cpu"
else:
    device = "cpu"

env = gym_super_mario_bros.make("SuperMarioBros-v0")
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = MarioEnv(env)
agent = DQN("cnn", env, replay_size=100000, epsilon_decay=100000)

trainer = SupervisedTrainer(
    agent=agent,
    env=env,
    dataset=args.input_path,
    possible_actions=SIMPLE_MOVEMENT,
    render=args.render,
    device=device,
    length=args.length,
)
trainer.train(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
