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
import os
import random
import math
from copy import deepcopy
from collections import deque, namedtuple
from torch.utils.data import IterableDataset

# Dataset to store and replay experiences
field_names = ('state', 'action', 'reward', 'next_state', 'done')
Experience = namedtuple('Experience', field_names)

class ReplayMemory(IterableDataset):

    def __init__(self, capacity, sample_size):
        self.memory = deque(maxlen=capacity)
        self.sample_size = sample_size

    def __iter__(self):
        samples = random.sample(self.memory, self.sample_size)
        for sample in samples:
            yield sample

    def push(self, experience):
        self.memory.append(experience)

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

# Agent that chooses actions in environment
class Agent:

    def __init__(self, env, replay_memory, policy_net):
        self.env = env
        self.replay_memory = replay_memory
        self.policy_net = policy_net
        self.state = None
        self.reset()

    def reset(self):
        self.state = self.env.reset()

    def get_action(self, epsilon):
        if random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state = torch.tensor([self.state], dtype=torch.float)
            q_values = self.policy_net(state)
            action = q_values.argmax(dim=1).item()
        return action

    def play_step(self, epsilon):
        action = self.get_action(epsilon)

        new_state, reward, done, _ = self.env.step(action)
        exp = Experience(self.state, action, reward, new_state, done)
        self.replay_memory.push(exp)

        self.state = new_state
        if done:
            self.reset()
        return reward, done

    def play(self):
        self.reset()
        done = False
        while not done:
            self.env.render()
            reward, done = self.play_step(epsilon=0.0)

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

        self.replay_memory = ReplayMemory(self.hparams.replay_size,
                                          self.hparams.episode_size)
        self.agent = Agent(self.env, self.replay_memory, self.policy_net)

        self.episode_reward = 0
        self.total_reward = 0
        self.populate(self.hparams.start_steps)

    def populate(self, steps):
        for _ in range(steps):
            self.agent.play_step(epsilon=1.0)

    def get_epsilon(self, epsilon_start, epsilon_end, epsilon_decay, global_step):
        return epsilon_end + (epsilon_start - epsilon_end) * math.exp(
            -1.0 * global_step / epsilon_decay)

    def forward(self, x):
        return self.policy_net(x)

    def dqn_mse_loss(self, batch):
        states, actions, rewards, next_states, dones = batch
        states = states.float()
        next_states = next_states.float()
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
        reward, done = self.agent.play_step(epsilon)
        self.episode_reward += reward

        loss = self.dqn_mse_loss(batch)
        if self.global_step % self.hparams.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        if done:
            self.total_reward = self.episode_reward
            self.episode_reward = 0
        self.log("total_reward", self.total_reward)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.policy_net.parameters(), lr=self.hparams.lr)

    def train_dataloader(self):
        return DataLoader(self.replay_memory, batch_size=self.hparams.batch_size)

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
    agent = model.agent
    agent.play()
