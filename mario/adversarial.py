from typing import List
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from genrl.deep.common.utils import get_env_properties, cnn

from mario.expert import MarioExpertTransitions
from mario.trainer import MarioTrainer
from mario.utils import mask_raw_actions


class Discriminator(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dims=[128, 64]):
        super(Discriminator, self).__init__()
        self.conv, self.conv_output_size = cnn()
        self.fc1 = nn.Linear(self.conv_output_size, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0] + action_dim, hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], 1)

    def forward(self, s, a):
        x = self.conv(s).view(-1, self.conv_output_size)
        x = F.relu(self.fc1(x))
        x = torch.cat([x, a], dim=-1)
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x


class AdversariaTrainer(MarioTrainer):
    def __init__(
        self,
        agent,
        env,
        dataset,
        possible_actions: List = None,
        embedding: bool = False,
        shuffle: bool = True,
        log_mode=["stdout"],
        device="cpu",
        length=None,
        **kwargs
    ) -> None:
        """Trainer for generative adversarial training of an agent

        References:
            https://arxiv.org/pdf/1606.03476.pdf

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
            device (str, optional): Device . Defaults to "cpu".
            length (int, optional): Number of actions to consider.
                Applicable only if path given for dataset.
                Defaults to None which implies complete dataset to be loaded
        """
        super(AdversariaTrainer, self).__init__(
            agent, env, log_mode, "epoch", device=device, **kwargs
        )
        self.agent = agent

        if isinstance(dataset, (str, os.PathLike)):
            self.dataset = MarioExpertTransitions(
                data_path=dataset,
                screen_size=84,
                grayscale=True,
                device=device,
                length=length,
            )
        elif isinstance(dataset, Dataset):
            self.dataset = dataset
        else:
            raise TypeError

        input_dim, action_dim, _, _ = get_env_properties(self.env, "cnn")
        self.discriminator = Discriminator(input_dim, action_dim)

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

        data = DataLoader(self.dataset, batch_size, self.shuffle)

        policy_optim = torch.optim.Adam(self.agent.model.parameters(), lr=lr)
        discriminator_optim = torch.optim.Adam(self.discriminator.parameters(), lr=lr)

        for e in range(epochs):
            dataiter = iter(data)
            total_discriminator_loss = 0.0
            total_policy_loss = 0.0

            for _ in range(len(dataiter) // 2):

                # Sample data
                exp_s, exp_a = dataiter.next()
                s, _ = dataiter.next()

                if self.possible_actions is not None:
                    target_actions = mask_raw_actions(exp_a, self.possible_actions)
                    target_actions = F.one_hot(
                        target_actions, num_classes=len(self.possible_actions)
                    ).to(torch.float32)
                else:
                    target_actions = F.one_hot(exp_a, num_classes=256).to(torch.float32)

                action_pred = self.agent.model(s)

                # Discriminator update
                exp_label = self.discriminator(exp_s, target_actions)
                model_label = self.discriminator(s, action_pred.detach())
                discriminator_loss = F.binary_cross_entropy(
                    exp_label, torch.ones(exp_label.shape[0], 1).to(self.device)
                ) + F.binary_cross_entropy(
                    model_label, torch.zeros(model_label.shape[0], 1).to(self.device)
                )
                discriminator_optim.zero_grad()
                discriminator_loss.backward()
                discriminator_optim.step()
                total_discriminator_loss += discriminator_loss.item()

                # Policy update
                policy_loss = -self.discriminator(s, action_pred).mean()
                policy_optim.zero_grad()
                policy_loss.backward()
                policy_optim.step()
                total_policy_loss += policy_loss.item()

            self.logger.write(
                {
                    "epoch": e,
                    "loss/policy": total_policy_loss,
                    "loss/discriminator": total_discriminator_loss,
                },
                "epoch",
            )
