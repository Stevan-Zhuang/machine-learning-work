# Machine learning
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Environment
import gym

# Extra tools
import random
import math
from copy import deepcopy
from collections import deque, namedtuple
from torch.utils.data import IterableDataset

# Policy and target network
class DQNNet(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(DQNNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)

# Dataset to store and replay experiences
field_names = ('state', 'action', 'reward', 'next_state', 'done')
Experience = namedtuple('Experience', field_names)

class ReplayMemory:

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def __len__(self):
        raise Exception()
        return len(self.memory)

    def push(self, experience):
        self.memory.append(experience)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

# Dataloader wrapper for replay memory
class RLDataset(IterableDataset):

    def __init__(self, replay_memory, sample_size):
        self.replay_memory = replay_memory
        self.sample_size = sample_size

    def __iter__(self):
        samples = self.replay_memory.sample(self.sample_size)
        for idx in range(self.sample_size):
            yield samples[idx]

# Agent that chooses actions in environment
class Agent:

    def __init__(self, env, replay_memory):
        self.env = env
        self.replay_memory = replay_memory
        self.state = None
        self.reset()

    def reset(self):
        self.state = self.env.reset()

    def get_action(self, policy_net, epsilon):
        if random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state = torch.tensor([self.state], dtype=torch.float)
            q_values = policy_net(state)
            action = q_values.argmax(dim=1).item()
        return action

    def play_step(self, policy_net, epsilon=0.0):
        action = self.get_action(policy_net, epsilon)

        new_state, reward, done, _ = self.env.step(action)
        exp = Experience(self.state, action, reward, new_state, done)
        self.replay_memory.push(exp)

        self.state = new_state
        if done:
            self.reset()
        return reward, done

# Pytorch Lightning DQN system
class DQNModel(pl.LightningModule):

    def __init__(self, hparams):
        super(DQNModel, self).__init__()
        self.hparams = hparams

        self.env = gym.make(self.hparams.env)
        obs_size = self.env.observation_space.shape[0]
        hidden_size = self.hparams.hidden_size
        n_actions = self.env.action_space.n
        self.policy_net = DQNNet(obs_size, hidden_size, n_actions)
        self.target_net = deepcopy(self.policy_net)

        self.replay_memory = ReplayMemory(self.hparams.replay_size)
        self.agent = Agent(self.env, self.replay_memory)

        self.populate(self.hparams.start_steps)

    def populate(self, steps):
        for _ in range(steps):
            self.agent.play_step(self.policy_net, epsilon=1.0)

    def get_epsilon(self, epsilon_start, epsilon_end, epsilon_decay, global_step):
        return epsilon_end + (epsilon_start - epsilon_end) * math.exp(
            -1.0 * global_step / epsilon_decay
        )

    def forward(self, x):
        return self.policy_net(x)

    def dqn_mse_loss(self, batch):
        states, actions, rewards, next_states, dones = batch
        states, next_states = states.float(), next_states.float()
        rewards = rewards.float()

        q_values = self.policy_net(states)
        q_values = q_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)

        next_actions = self.target_net(next_states).argmax(1)
        next_q_values = self.target_net(next_states)
        next_q_values = next_q_values.gather(1, next_actions.unsqueeze(-1)).squeeze(-1)
        next_q_values = next_q_values.detach()
        next_q_values[dones] = 0.0
        expected_q_values = next_q_values * self.hparams.gamma + rewards

        criterion = nn.MSELoss()
        loss = criterion(q_values, expected_q_values)
        return loss

    def training_step(self, batch, batch_idx):
        epsilon = self.get_epsilon(self.hparams.eps_start, self.hparams.eps_end,
                                   self.hparams.eps_decay, self.global_step)
        reward, done = self.agent.play_step(self.policy_net, epsilon)

        loss = self.dqn_mse_loss(batch)

        if self.global_step % self.hparams.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss

    def configure_optimizers(self):
        return optim.Adam(self.policy_net.parameters(), lr=self.hparams.lr)

    def train_dataloader(self):
        dataset = RLDataset(self.replay_memory, self.hparams.episode_size)
        return DataLoader(dataset, batch_size=self.hparams.batch_size)

# Run code
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v0")
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--replay_size", type=int, default=1000)
    parser.add_argument("--episode_size", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--start_steps", type=int, default=1000)
    parser.add_argument("--eps_start", type=float, default=1.0)
    parser.add_argument("--eps_end", type=float, default=0.01)
    parser.add_argument("--eps_decay", type=float, default=1000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--target_update", type=int, default=10)

    model = DQNModel(parser.parse_args())

    trainer = pl.Trainer(max_epochs=500)
    trainer.fit(model)

    model.eval()

    state = model.env.reset()
    while True:
        model.env.render()
        _, done = model.agent.play_step(model.policy_net)
        if done:
            break
