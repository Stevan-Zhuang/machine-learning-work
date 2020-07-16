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
train_df = pd.read_csv(r"C:\Users\Steva\OneDrive\Desktop\Programming\kaggle_datasets\Titanic\train.csv")
test_df = pd.read_csv(r"C:\Users\Steva\OneDrive\Desktop\Programming\kaggle_datasets\Titanic\test.csv")

y_train = train_df["Survived"]
train_df = train_df.drop("Survived", axis=1)

x_order = test_df["PassengerId"]

# Combine dataframes for easier handling
train_df["train"] = True
test_df["test"] = True

combined_df = pd.concat([train_df, test_df])
combined_df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)

for feature in combined_df:
    if feature == "train" or feature == "test":
        continue
    combined_df[feature].fillna(combined_df[feature].mean() if combined_df[feature].dtype != "object"
                                else combined_df[feature].mode()[0], inplace=True)
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
        self.optimizer = optim.SGD(self.parameters(), lr=learn_rate, momentum=0.9)

        self.learn_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.96)

    def predict(self, x):
        return self.net(x)

    def accuracy(self, y_pred, y):
        return torch.sum(torch.round(y_pred) == y).item() / y.shape[0]

    def train_step(self, num_epochs, data_loader):
        for epoch in range(num_epochs):
            for x_batch, y_batch in data_loader:
                self.train()

                y_pred = self.predict(x_batch)

                loss = self.criterion(y_pred, y_batch)
                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                self.learn_scheduler.step()

input_layer = x_train.shape[1]
hidden_layer = 4
output_layer = 1

learn_rate = 1
model = Model(nn.Sequential(
    nn.Linear(input_layer, hidden_layer),
    nn.ReLU(),
    nn.Linear(hidden_layer, output_layer),
    nn.Sigmoid()
),
    nn.BCELoss(),
    learn_rate
)

# Train model
num_epochs = 100
model.train_step(num_epochs, train_loader)

model.eval()
y_pred = model.predict(x_train)
print(model.accuracy(y_pred, y_train))

# Create submission
y_pred = model.predict(x_test)
y_pred = (y_pred > 0.5).int()
y_pred = y_pred.squeeze().tolist()

submission = pd.DataFrame({"PassengerId": x_order, "Survived": y_pred})
submission.to_csv(r"C:\Users\Steva\Downloads\Programming\saved\Titanic.csv", index = False)
