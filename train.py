import argparse

import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

from genrl.deep.agents.dqn import (DQN, DoubleDQN, DuelingDQN,
                                   PrioritizedReplayDQN)
from genrl.deep.agents import PPO1, A2C

from mario.agents import MarioPPO
from mario.mdp_trainer import MDPTrainer
from mario.wrapper import MarioEnv
from mario.buffers import MarioRollout


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(
        description="A script used to train RL agents on Super Mario Bros."
    )
    argument_parser.add_argument("-a", "--agent", type=str, default="dqn")
    argument_parser.add_argument("-e", "--epochs", type=int, default=100)
    argument_parser.add_argument("-l", "--log-interval", type=int, default=5)
    argument_parser.add_argument("-b", "--batch-size", type=int, default=64)
    argument_parser.add_argument("-t", "--max_ep_len", type=int, default=5000)
    argument_parser.add_argument("-r", "--render", action="store_true")
    argument_parser.add_argument("-p", "--load-model", type=str, default=None)
    argument_parser.add_argument("-x", "--evaluate", action="store_true")
    argument_parser.add_argument("--eval-episodes", type=int, default=20)
    argument_parser.add_argument("--lr-policy", type=float, default=3e-4)
    argument_parser.add_argument("--lr-value", type=float, default=1e-3)
    argument_parser.add_argument("--epsilon-decay", type=float, default=200000)
    argument_parser.add_argument("--replay-size", type=int, default=100000)
    argument_parser.add_argument("--rollout-size", type=int, default=2048)
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
        off_policy = True
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
            epsilon_decay=args.epsilon_decay,
            device=device,
        )
    else:
        off_policy = False
        if args.agent == "ppo1" or args.agent == "ppo":
            agent_class = MarioPPO
        elif args.agent == "a2c":
            agent_class = A2C
        else:
            raise NotImplementedError
        agent = agent_class(
            "cnn",
            env,
            buffer_class=MarioRollout,
            rollout_size=args.rollout_size,
            batch_size=args.batch_size,
            lr_policy=args.lr_policy,
            lr_value=args.lr_value,
            device=device,
        )

    trainer = MDPTrainer(
        agent,
        env,
        log_interval=args.log_interval,
        epochs=args.epochs,
        off_policy=off_policy,
        render=args.render,
        max_ep_len=args.max_ep_len,
        evaluate_episodes=args.eval_episodes,
        load_model=args.load_model,
        save_interval=10,
        save_model="checkpoints"
    )

    if args.evaluate:
        trainer.evaluate()
    else:
        trainer.train()

