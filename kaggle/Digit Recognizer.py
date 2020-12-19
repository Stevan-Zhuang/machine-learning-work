import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split


x_train = pd.read_csv(r"..\kaggle_datasets\MNIST\train.csv")
x_test = pd.read_csv(r"..\kaggle_datasets\MNIST\test.csv")

y_train = x_train['label']
y_train = pd.get_dummies(y_train)
x_train.drop('label', axis=1, inplace=True)

test_id = x_test.index

x_train = torch.tensor(x_train.values, dtype=torch.float)
y_train = torch.tensor(y_train.values, dtype=torch.float)
x_test = torch.tensor(x_test.values, dtype=torch.float)

x_train = (x_train - x_train.mean()) / x_train.std()
x_test = (x_test - x_test.mean()) / x_test.std()

train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)


input_layer = 28 * 28
hidden_layers = [128, 64]
output_layer = 10

model = nn.Sequential(
    nn.Linear(input_layer, hidden_layers[0]),
    nn.ReLU(),
    nn.Linear(hidden_layers[0], hidden_layers[1]),
    nn.ReLU(),
    nn.Linear(hidden_layers[1], output_layer),
    nn.Softmax(dim=1)
)

criterion = nn.BCELoss()

learn_rate = 1e-1
optimizer = optim.SGD(model.parameters(), lr=learn_rate)

num_epochs = 16
for epoch in range(num_epochs):
    for x_batch, y_batch in train_loader:
        model.train()

        y_pred = model(x_batch)

        loss = criterion(y_pred, y_batch)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

model.eval()
y_pred = model(x_test)
y_pred = y_pred.argmax(1)

submission = pd.DataFrame({'ImageId': test_id + 1, 'Label': y_pred.numpy()})
submission.to_csv(r"C:\Users\Steva\OneDrive\Desktop\Programming\saved\MNIST.csv", index=False)

# Current best score: 
