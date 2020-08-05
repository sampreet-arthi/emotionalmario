from abc import ABC

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from genrl.deep.common.logger import Logger
from genrl.deep.common.utils import safe_mean, set_seeds
from mario.expert import ExpertTransitionDataset


class Trainer(ABC):
    def __init__(
        self,
        agent,
        env,
        log_mode=["stdout"],
        log_key: str = "timestep",
        buffer=None,
        off_policy: bool = True,
        save_interval: int = 0,
        save_model: str = "checkpoints",
        run_num: int = None,
        load_model: str = None,
        render: bool = False,
        max_ep_len: int = 1000,
        distributed: bool = False,
        steps_per_epoch: int = 500,
        epochs: int = 10,
        device="cpu",
        log_interval: int = 10,
        evaluate_episodes: int = 500,
        logdir: str = "logs",
        batch_size: int = 50,
        seed=0,
        deterministic_actions: bool = False,
        warmup_steps: int = 1000,
        start_update: int = 1000,
        update_interval: int = 50,
    ):
        self.agent = agent
        self.env = env
        self.log_mode = log_mode
        self.log_key = log_key
        self.logdir = logdir
        self.off_policy = off_policy
        self.save_interval = save_interval
        self.save_model = save_model
        self.run_num = run_num
        self.load_model = load_model
        self.render = render
        self.max_ep_len = max_ep_len
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.device = device
        self.log_interval = log_interval
        self.evaluate_episodes = evaluate_episodes
        self.batch_size = batch_size
        self.deterministic_actions = deterministic_actions
        self.warmup_steps = warmup_steps
        self.update_interval = update_interval
        self.start_update = start_update
        self.network = self.agent.network
        self.buffer = self.agent.replay_buffer

        if seed is not None:
            set_seeds(seed, self.env)

        self.logger = Logger(logdir=logdir, formats=[*log_mode])

    def evaluate(self, render=False):
        episode, episode_reward = 0, 0
        episode_rewards = []
        state = self.env.reset()
        while True:
            if self.off_policy:
                action = self.agent.select_action(state, deterministic=True)
            else:
                action, _, _ = self.agent.select_action(state)

            if isinstance(action, torch.Tensor):
                action = action.numpy()

            next_state, reward, done, _ = self.env.step(action)

            if render:
                self.env.render()

            episode_reward += reward
            state = next_state
            if np.any(done):
                for i, di in enumerate(done):
                    if di:
                        episode += 1
                        episode_rewards.append(episode_reward[i])
                        episode_reward[i] = 0
            if episode == self.evaluate_episodes:
                print(
                    "Evaluated for {} episodes, Mean Reward: {}, Std Deviation for the Reward: {}".format(
                        self.evaluate_episodes,
                        np.mean(episode_rewards),
                        np.std(episode_rewards),
                    )
                )
                return

    def train(self):
        if self.load_model is not None:
            self.load()

        self.agent.update_target_model()

        timesteps = 0

        for episode in range(1, self.epochs + 1):
            state = self.env.reset()
            for timestep in range(self.steps_per_epoch):
                self.agent.update_params_before_select_action(timestep)

                action = int(self.agent.select_action(state))

                next_state, reward, done, info = self.env.step(action)

                if self.render:
                    self.env.render()

                self.buffer.push((state, action, reward, next_state, done))
                state = next_state.copy()

                if done or timestep == self.steps_per_epoch - 1:
                    timesteps += timestep
                    self.logger.write(
                        {
                            "timestep": timesteps,
                            "Episode": episode,
                            **self.agent.get_logging_params(),
                            "Episode Reward": self.env.episode_reward,
                        },
                        self.log_key,
                    )

                    state = self.env.reset()
                    break

                if (
                    timestep >= self.start_update
                    and timestep % self.update_interval == 0
                ):
                    self.agent.update_params(self.update_interval)

        self.env.close()
        self.logger.close()


class SupervisedTrainer(Trainer):
    def __init__(
        self,
        agent,
        env,
        dataset,
        shuffle: bool = True,
        log_mode=["stdout"],
        render: bool = False,
        device="cpu",
        length=None,
    ):
        """Trainer for behavior cloning of dataset into agent

        Args:
            agent: Agent to train
            env: Env from which daa has originated
            dataset: Dataset to train on. Can be path to dataset or dataset object.
            shuffle (bool): Whether to shuffle dataset. Defaults to true
            log_mode (list, optional): Mode to log progress. Defaults to ["stdout"].
            render (bool, optional): Whether to render while loading dataset. 
                Applicable only if path given for dataset. Defaults to False.
            device (str, optional): Device . Defaults to "cpu".
            length (int, optional): Number of actions to consider.
                Applicable only if path given for dataset.
                Defaults to None which implies complete dataset to be loaded
        """
        super(SupervisedTrainer, self).__init__(
            agent, env, log_mode, "epoch", device=device, render=render
        )
        self.agent = agent

        if isinstance(dataset, str):
            self.dataset = ExpertTransitionDataset(dataset, device, render, length)
        elif isinstance(dataset, Dataset):
            self.dataset = dataset
        else:
            raise TypeError

        self.shuffle = shuffle

    def train(
        self, epochs: int = 1, lr: float = 1e-3, batch_size: int = 64,
    ):
        """Trains agent on dataset of expert transitions with MSE loss on action probabilities.

        Args:
            epochs (int): Epochs to train for. Defaults to 1
            lr (float): Learning rate. Defaults to 0.001
            batch_size (int, optional): Size of batch for gradient update. Defaults to 64.
        """
        dataloader = DataLoader(self.dataset, batch_size, self.shuffle)
        optim = torch.optim.Adam(self.agent.model.parameters(), lr=lr)
        losses = []
        for e in range(epochs):
            epoch_loss = 0.0
            for obs, a in dataloader:
                action_onehot = F.one_hot(a)
                import pdb

                pdb.set_trace()
                action_pred = self.agent.model(obs)
                loss = F.mse_loss(action_pred, action_onehot)
                optim.zero_grad()
                loss.backward()
                optim.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss)
            self.logger.write(
                {"epoch": e, "loss": epoch_loss,}, "epoch",
            )
