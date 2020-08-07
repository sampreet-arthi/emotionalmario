import argparse

import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

from genrl.deep.agents.dqn import (DQN, DoubleDQN, DuelingDQN,
                                   PrioritizedReplayDQN)
from genrl.agents import A2C, PPO1

from mario.trainer import MarioTrainer
from mario.wrapper import MarioEnv


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(
        description="A script used to clone expert data into agent."
    )
    argument_parser.add_argument("-a", "--agent", type=str, default="dqn")
    argument_parser.add_argument("-e", "--epochs", type=int, default=10)
    argument_parser.add_argument("-l", "--log-interval", type=int, default=5)
    argument_parser.add_argument("--lr-policy", type=float, default=3e-4)
    argument_parser.add_argument("--lr-value", type=float, default=1e-3)
    argument_parser.add_argument("-b", "--batch-size", type=int, default=64)
    argument_parser.add_argument("-o", "--off-policy", action="store_false")
    argument_parser.add_argument("--replay-size", type=int, default=10000)
    argument_parser.add_argument("--rollout-size", type=int, default=2048)
    argument_parser.add_argument("-t", "--max_ep_len", type=int, default=5000)
    argument_parser.add_argument("-r", "--render", action="store_true")
    argument_parser.add_argument("--enable-cuda", action="store_true")
    args = argument_parser.parse_args()

    if args.enable_cuda:
        device = "cuda"
    else:
        device = "cpu"

    env = gym_super_mario_bros.make("SuperMarioBros-v2")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = MarioEnv(env)

    if "dqn" in args.agent:
        if "double" in args.agent or args.agent == "ddqn":
            dqn_class = DoubleDQN
        elif "prioritized" in args.agent:
            dqn_class = PrioritizedReplayDQN
        elif "dueling" in args.agent:
            dqn_class = DuelingDQN
        else:
            dqn_class = DQN
        agent = dqn_class(
            "cnn",
            env,
            replay_size=args.replay_size,
            batch_size=args.batch_size,
            lr_value=args.lr_value,
            device=device,
        )
    else:
        if args.agent == "ppo1" or args.agent == "ppo":
            agent_class = PPO1
        elif args.agent == "a2c":
            agent_class = A2C
        else:
            raise NotImplementedError
        agent = agent_class(
            "cnn",
            env,
            rollout_size=args.rollout_size,
            batch_size=args.batch_size,
            lr_policy=args.lr_policy,
            lr_value=args.lr_value,
            device=device,
        )

    trainer = MarioTrainer(
        agent,
        env,
        log_interval=args.log_interval,
        epochs=args.epochs,
        off_policy=args.off_policy,
        render=args.render,
        max_ep_len=args.max_ep_len,
        evaluate_episodes=25,
    )
    trainer.train()
    trainer.evaluate()