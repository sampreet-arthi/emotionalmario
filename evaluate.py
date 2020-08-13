import argparse

import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

from genrl.deep.agents.dqn import DQN, DoubleDQN, DuelingDQN, PrioritizedReplayDQN
from genrl.deep.agents import A2C

from mario.agents import MarioPPO
from mario.base.trainer import MarioTrainer
from mario.base.wrapper import MarioEnv
from mario.base.buffers import MarioRollout


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(
        description="A script used to clone expert data into agent."
    )
    argument_parser.add_argument("-a", "--agent", type=str, default="dqn")
    argument_parser.add_argument("-e", "--evaluate-episodes", type=int, default=20)
    argument_parser.add_argument("-p", "--path", type=str, default=None)
    argument_parser.add_argument("-t", "--max_ep_len", type=int, default=999999)
    argument_parser.add_argument("-r", "--render", action="store_true")
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

    trainer = MarioTrainer(
        agent,
        env,
        log_interval=args.log_interval,
        epochs=args.epochs,
        off_policy=off_policy,
        render=args.render,
        max_ep_len=args.max_ep_len,
        save_interval=10,
        load_model=args.path,
        save_model="checkpoints",
    )

    trainer.evaluate()
