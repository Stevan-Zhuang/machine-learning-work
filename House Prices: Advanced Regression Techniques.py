# Import modules
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split


# Import and organize data
train_df = pd.read_csv(r"C:\Users\Steva\OneDrive\Desktop\Programming\kaggle_datasets\House Prices\train.csv")
test_df = pd.read_csv(r"C:\Users\Steva\OneDrive\Desktop\Programming\kaggle_datasets\House Prices\test.csv")

y_train = train_df["SalePrice"]
train_df.drop("SalePrice", axis=1, inplace=True)

x_order = test_df["Id"]

# Combine dataframes for easier handling
train_df["train"] = True
test_df["test"] = True

combined_df = pd.concat([train_df, test_df])

for feature in combined_df:
    if feature == "train" or feature == "test":
        continue

    if combined_df[feature].isna().mean() > 0.2:
        combined_df.drop(feature, axis=1, inplace=True)
        continue

    fill = combined_df[feature].mode()[0]
    if combined_df[feature].dtype != "object":
        fill = combined_df[feature].mean()
    combined_df[feature].fillna(fill, inplace=True)

    if combined_df[feature].dtype == "object":
        combined_df = pd.concat([combined_df, pd.get_dummies(combined_df[feature], prefix=feature)], axis=1)
        combined_df.drop(feature, axis=1, inplace=True)

# Separate dataframes and select best features
train_df = combined_df[combined_df["train"] == True]
test_df = combined_df[combined_df["test"] == True]

train_df = train_df.drop(["train", "test"], axis=1)
test_df = test_df.drop(["train", "test"], axis=1)

train_df = (train_df - train_df.mean()) / train_df.std()
test_df = (test_df - test_df.mean()) / test_df.std()

for feature in train_df:
    if abs(train_df[feature].corr(y_train)) < 0.45:
        train_df.drop(feature, axis=1, inplace=True)
        test_df.drop(feature, axis=1, inplace=True)

# Prepare data for training
x_train = torch.tensor(train_df.values, dtype=torch.float)
x_test = torch.tensor(test_df.values, dtype=torch.float)
y_train = torch.tensor(y_train, dtype=torch.float).view(-1, 1)

train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=100)


# Create model
class Model(nn.Module):
    def __init__(self, net, loss_func, learn_rate):
        super(Model, self).__init__()
        self.net = net

        self.criterion = loss_func
        self.optimizer = optim.SGD(self.parameters(), lr=learn_rate)

    def predict(self, x):
        return self.net(x)

    def train_step(self, num_epochs, data_loader):
        for epoch in range(num_epochs):
            for x_batch, y_batch in data_loader:
                self.train()

                y_pred = self.predict(x_batch)

                loss = self.criterion(y_pred, y_batch)
                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()
            print(y_pred)

        print(loss.item())

learn_rate = 1e-6
model = Model(nn.Sequential(
    nn.Linear(x_train.shape[1], 2),
    nn.BatchNorm1d(2),
    nn.ReLU(),
    nn.Linear(2, 1)
),
    nn.MSELoss(reduction="mean"),
    learn_rate
)

# Train model
num_epochs = 16
model.train_step(num_epochs, train_loader)

# Create submission
y_pred = model.predict(x_test)
y_pred = y_pred.squeeze().tolist()

submission = pd.DataFrame({"Id": x_order, "SalePrice": y_pred})
submission.to_csv(r"C:\Users\Steva\Downloads\Programming\saved\House Prices.csv", index = False)

# Current best score: 0.16452
