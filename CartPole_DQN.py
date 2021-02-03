import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, IterableDataset
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb

import math
import random
import numpy as np
import matplotlib.pyplot as plt

import gym

from collections import namedtuple

field_names = ('state', 'action', 'reward', 'done', 'next_state')
Experience = namedtuple('Experience', field_names)

from collections import deque

class ReplayMemory(IterableDataset):

    def __init__(self, capacity, sample_size):
        self.memory = deque(maxlen=capacity)
        self.sample_size = sample_size

    def __iter__(self):
        samples = self.sample()
        for sample in zip(*samples):
            yield sample

    def push(self, experience):
        self.memory.append(experience)

    def sample(self):
        samples = random.sample(self.memory, self.sample_size)
        states, actions, rewards, dones, next_states = zip(*samples)
        return (torch.tensor(states), torch.tensor(actions),
                torch.tensor(rewards, dtype=torch.float32),
                torch.tensor(dones), torch.tensor(next_states))

class DQN(nn.Module):

    def __init__(self, state_size, hidden_size, n_actions):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x.float())


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
            state = torch.tensor([self.state])
            q_values = policy_net(state)
            action = q_values.argmax(dim=1).item()
        return action

    def play_step(self, policy_net, epsilon):
        action = self.get_action(policy_net, epsilon)

        new_state, reward, done, _ = self.env.step(action)
        exp = Experience(self.state, action, reward, done, new_state)
        self.replay_memory.push(exp)

        self.state = new_state
        if done:
            self.reset()
        return reward, done


class DQNModel(pl.LightningModule):

    def __init__(self):
        super(DQNModel, self).__init__()
        self.env = gym.make(ENV)
        self.replay_memory = ReplayMemory(REPLAY_SIZE, SAMPLE_SIZE)
        self.agent = Agent(self.env, self.replay_memory)

        obs_size = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n
        self.policy_net = DQN(obs_size, HIDDEN_SIZE, n_actions)
        self.target_net = DQN(obs_size, HIDDEN_SIZE, n_actions)

        self.populate(START_STEPS)

    def populate(self, steps):
        for idx in range(steps):
            self.agent.play_step(self.policy_net, epsilon=1.0)

    def forward(self, x):
        return self.policy_net(x)

    def dqn_mse_loss(self, batch):
        states, actions, rewards, dones, next_states = batch
        state_action_values = self.policy_net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_state_values, _ = self.target_net(next_states).max(1)
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * GAMMA + rewards

        criterion = nn.MSELoss()
        loss = criterion(state_action_values, expected_state_action_values)
        return loss

    def training_step(self, batch, batch_idx):
        epsilon = max(EPS_END, EPS_START - self.global_step + 1 / EPS_DECAY)
        reward, done = self.agent.play_step(self.policy_net, epsilon)

        loss = self.dqn_mse_loss(batch)

        if self.global_step % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=LEARN_RATE)

    def train_dataloader(self):
        return DataLoader(self.replay_memory, batch_size=BATCH_SIZE)


ENV = "CartPole-v0"
LEARN_RATE = 0.001
HIDDEN_SIZE = 128
BATCH_SIZE = 128
SAMPLE_SIZE = 200
REPLAY_SIZE = 1000
START_STEPS = 1000
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10


model = DQNModel()

trainer = pl.Trainer(max_epochs=1000, progress_bar_refresh_rate=30)
trainer.fit(model)

env = gym.make(ENV)
state = env.reset()
model.eval()
while True:
    env.render()
    action = model(torch.tensor([state])).argmax(dim=1).item()
    new_state, _, done, _ = env.step(action)
    state = new_state
    if done:
        break
