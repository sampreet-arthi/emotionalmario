from typing import List

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F

from mario.expert import MarioExpertTransitions
from mario.trainer import Trainer
from mario.utils import mask_raw_actions


class SupervisedTrainer(Trainer):
    def __init__(
        self,
        agent,
        env,
        dataset,
        possible_actions: List = None,
        embedding: bool = False,
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
            possible_actions (List, optional): All actions that are possible. 
                Defaults to None which implies all actions are valid.
            embedding (bool): Whether the agent is using an embedding for the
                action space. Defaults to False.
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
            self.dataset = MarioExpertTransitions(
                session_path=dataset,
                screen_size=84,
                grayscale=True,
                device=device,
                render=render,
                length=length,
            )
        elif isinstance(dataset, Dataset):
            self.dataset = dataset
        else:
            raise TypeError

        self.possible_actions = possible_actions
        self.embedding = embedding
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
                if self.possible_actions is not None:
                    masked_actions = mask_raw_actions(a, self.possible_actions)
                    target_actions = F.one_hot(
                        masked_actions, num_classes=len(self.possible_actions)
                    ).to(torch.float32)
                else:
                    target_actions = F.one_hot(a, num_classes=len(256)).to(torch.float32)

                if self.embedding:
                    pred_embedding = self.agent.embed(obs)
                    target_embedding = self.agent.embed_actions(target_actions)
                    loss = F.mse_loss(pred_embedding, target_embedding)
                else:
                    action_pred = self.agent.model(obs)
                    loss = F.mse_loss(action_pred, target_actions)
                    # loss = F.nll_loss(action_pred, masked_actions)

                optim.zero_grad()
                loss.backward()
                optim.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss)
            self.logger.write(
                {"epoch": e, "loss": epoch_loss,}, "epoch",
            )
