import numpy as np

from genrl.deep.common.trainer import Trainer

class MarioTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(MarioTrainer, self).__init__(*args, **kwargs)

    def evaluate(self, render=True) -> None:
        print("Evaluating ...")

        self.eval_rewards = []

        for _ in range(self.evaluate_episodes):
            state = self.env.reset()
            for timestep in range(self.max_ep_len):
                if self.off_policy:
                    action = self.agent.select_action(state, deterministic=True)
                else:
                    action, _, _ = self.agent.select_action(state)

                action = int(action)
                next_state, reward, done, _ = self.env.step(action)

                if render:
                    self.env.render()

                state = next_state

                if done or timestep == self.max_ep_len - 1:
                    self.eval_rewards.append(self.env.episode_reward)
                    break

        print(
            "Evaluated for {} episodes, Mean Reward: {}, Std Deviation for the Reward: {}".format(
                self.evaluate_episodes,
                np.mean(self.eval_rewards),
                np.std(self.eval_rewards),
            )
        )
