import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.metrics import Accuracy
from pytorch_lightning import seed_everything
import wandb

import numpy as np
import matplotlib.pyplot as plt
import random
import argparse
import os

from io import open

class NameDataset(Dataset):

    def __init__(self, languages, names, letter2index, lang2index, n_letters):
        self.samples = []
        for lang_idx, lang in enumerate(languages):
            for name in names[lang_idx]:
                name_tensor = NameDataset.name2tensor(name, letter2index, n_letters)
                lang_tensor = NameDataset.lang2tensor(lang, lang2index)
                self.samples.append((name_tensor, lang_tensor))

    @staticmethod
    def name2tensor(name, letter2index, n_letters):
        name_tensor = torch.zeros(len(name), n_letters)
        for letter_idx, letter in enumerate(name):
            name_tensor[letter_idx, letter2index[letter]] = 1.
        return name_tensor

    @staticmethod
    def lang2tensor(lang, lang2index):
        return torch.tensor(lang2index[lang], dtype=torch.long)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class NameDataModule(pl.LightningDataModule):

    def __init__(self):
        super(NameDataModule, self).__init__()

        self.letters = set()
        self.languages = []
        self.names = []
        self.letter2index = {}
        self.lang2index = {}

    def setup(self, stage=None):
        path = os.getcwd() + '/datasets/name_data/'
        for txt_file_name in os.listdir(path):
            language = txt_file_name[:-len('.txt')]
            self.languages.append(language)

            with open(path + txt_file_name, encoding='utf-8') as file:
                file_names = file.read().strip().split('\n')
                self.names.append(file_names)

                for name in file_names:
                    self.letters.update(name)

        self.n_languages = len(self.languages)
        self.n_letters = len(self.letters) + 1

        for idx, letter in enumerate(self.letters):
            self.letter2index[letter] = idx
        for idx, lang in enumerate(self.languages):
            self.lang2index[lang] = idx

        self.dataset = NameDataset(
            self.languages, self.names,
            self.letter2index, self.lang2index,
            self.n_letters,
        )

    def train_dataloader(self):
        return DataLoader(self.dataset, shuffle=True)

class NameGenerator(pl.LightningModule):

    def __init__(self, hparams, input_size, output_size):
        super(NameGenerator, self).__init__()
        self.hparams = hparams

        self.rnn = nn.RNN(input_size, self.hparams.hidden_size, self.hparams.n_layers,
                          batch_first=True)
        self.fc = nn.Linear(self.hparams.hidden_size, output_size)
        self.train_acc = Accuracy()

    def forward(self, x):
        out, hidden = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out, hidden

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, hidden = self(x)
        self.train_acc(y_hat, y)
        loss = F.cross_entropy(y_hat, y)
        self.log('loss', loss)
        return loss
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.learn_rate)

if __name__ == '__main__':
    seed_everything(6)

    name_dm = NameDataModule()
    name_dm.setup()

    parser = argparse.ArgumentParser()
    parser.add_argument('--learn_rate', type=float, default=3e-4)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--n_layers', type=int, default=1)
    args = parser.parse_args()

    model = NameGenerator(args, name_dm.n_letters, name_dm.n_languages)

    trainer = pl.Trainer(
        max_epochs=3,
        callbacks=[EarlyStopping('loss')]
    )
    trainer.fit(model, name_dm)

    print(model.train_acc.compute())
