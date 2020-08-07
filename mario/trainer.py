from abc import ABC

import numpy as np
import torch

from genrl.deep.common.logger import Logger
from genrl.deep.common.trainer import Trainer
from genrl.deep.common.utils import set_seeds, safe_mean


class MarioTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(MarioTrainer, self).__init__(*args, **kwargs)

        self.start_update = kwargs["start_update"] if "start_update" in kwargs else 1000
        self.update_interval = (
           kwargs["update_interval"] if "update_interval" in kwargs else 60
        )

    def evaluate(self, render=True) -> None:
        self.eval_rewards = []

        for episode in range(self.evaluate_episodes):
            state = self.env.reset()
            for timestep in range(self.env.max_ep_len):
                if self.off_policy:
                    action = self.agent.select_action(state, deterministic=True)
                else:
                    action, _, _ = self.agent.select_action(state)

                action = int(action)
                next_state, reward, done, _ = self.env.step(action)

                if render:
                    self.env.render()

                state = next_state

                if done:
                    self.eval_rewards.append(self.env.episode_reward)
                    break

        print(
            "Evaluated for {} episodes, Mean Reward: {}, Std Deviation for the Reward: {}".format(
                self.evaluate_episodes,
                np.mean(self.eval_rewards),
                np.std(self.eval_rewards),
            )
        )

    def train(self):
        if self.load_model is not None:
            self.load()

        print("Training starting...")
        print("Agent: {}, Env: {}, Epochs: {}".format(self.agent.__class__.__name__, self.env.unwrapped.spec.id, self.epochs))
        if self.off_policy:
            self.off_policy_train()
        else:
            self.on_policy_train()

    def off_policy_train(self):
        self.agent.update_target_model()

        timesteps = 0
        self.rewards = []

        for episode in range(1, self.epochs + 1):
            state = self.env.reset()
            for timestep in range(self.max_ep_len):
                self.agent.update_params_before_select_action(timesteps)

                action = int(self.agent.select_action(state))

                next_state, reward, done, info = self.env.step(action)

                if self.render:
                    self.env.render()

                self.buffer.push((state, action, reward, next_state, done))
                state = next_state.copy()

                if done or timestep == self.max_ep_len - 1:
                    timesteps += timestep
                    self.rewards.append(self.env.episode_reward)
                    if episode % self.log_interval == 0:
                        self.logger.write(
                            {
                                "timestep": timesteps,
                                "Episode": episode,
                                **self.agent.get_logging_params(),
                                "Mean Reward": safe_mean(self.rewards),
                            },
                            self.log_key,
                        )
                        self.rewards = []
                    break

                if (
                    timestep >= self.start_update
                    and timestep % self.update_interval == 0
                ):
                    self.agent.update_params(self.update_interval)

            if self.save_interval != 0 and episode % self.save_interval == 0:
                self.save(episode * self.agent.batch_size)

        self.env.close()
        self.logger.close()

    def on_policy_train(self):
        for episode in range(self.epochs):
            self.agent.rollout.reset()

            state = self.env.reset()
            values, done = self.agent.collect_rollouts(state)

            self.agent.get_traj_loss(values, done)

            self.agent.update_policy()

            if episode % self.log_interval == 0:
                self.logger.write(
                    {
                        "timestep": episode * self.agent.rollout_size,
                        "Episode": episode,
                        **self.agent.get_logging_params(),
                    },
                    self.log_key,
                )

            if self.render:
                self.env.render()

            if self.save_interval != 0 and episode % self.save_interval == 0:
                self.save(episode * self.agent.batch_size)

        self.env.close()
        self.logger.close()
