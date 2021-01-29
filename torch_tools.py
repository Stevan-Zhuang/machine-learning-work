import torch
from torch import nn
from torch.nn import functional as F

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd

def to_tensor(data):
    """
    Converts input data to torch tensor
    Parameters
    ----------
    data : array-like, matrix object
    Returns
    -------
    tensor : data in tensor form
    """
    # Torch tensor
    if torch.is_tensor(data):
        return data
    # Pandas object, convert to numpy array
    if hasattr(data, 'iloc'):
        data = data.values
    # Numpy array
    if isinstance(data, np.ndarray):
        try: return torch.tensor(data).view(-1, data.shape[1])
        except: return torch.tensor(data).view(-1, 1)

    raise TypeError("Object could not be converted to a torch tensor. "
                    "Expected data to be array-like, matrix object, "
                    "recieved object of type {} instead."
                    .format(type(data)))

class SciKitModel(nn.Module):
    """
    An extension of the ``torch.nn.Module`` that wraps an instance of a torch model
    and adds compatibility with sklearn, as well as regression and classification.
    Currently, the class has basic support for ``sklearn.pipeline`` and ``sklearn.GridSearchCV``.
    Parameters
    ----------
    net : torch model object from the ``torch.nn`` module.
    is_classifier : bool whether the model is a classifier. Changes criterion.
    has_logits : bool (default=False) whether the labels given will be in logit form
    criterion : loss function (default=nn.MSELoss() if not is_classifier else nn.NLLLoss()) from the ``torch.nn`` module.
    Recommended: default. Other loss functions are unsupported and may cause errors.
    learn_rate : float (default=0.1) Learning rate for optimizer.
    num_epochs : int (default=100) The number of iterations the model will train for when using ``.fit(x, y)``.
    Examples
    --------
    >>> model = SciKitModel(
    ...     nn.Sequential(nn.Linear(4, 2),
    ...                   nn.ReLU(),
    ...                   nn.Linear(2, 1)),
    ...     is_classifier=False
    ... )
    >>> grid_search = GridSearchCV(estimator=model,
    ...                            param_grid={'learn_rate': [1e-5, 1e-1, 1e-10],
    ...                                        'num_epochs': [10, 1, 100]})
    >>> result = grid_search.fit(x, y).best_params_
    >>> result
    {'learn_rate': 0.1, 'max_epochs': 10}
    """
    def __init__(self, net,
                 is_classifier,
                 criterion=None,
                 has_logits=False,
                 learn_rate=1e-1,
                 num_epochs=100):
        super(SciKitModel, self).__init__()

        self.learn_rate = learn_rate
        self.num_epochs = num_epochs

        self.net = net

        self.is_classifier = is_classifier
        self.criterion = (criterion if not criterion is None
                          else nn.MSELoss() if not is_classifier else nn.NLLLoss())
        self.has_logits = has_logits

        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learn_rate)

    def predict(self, x):
        """
        Predict labels of features using the model.
        Parameters
        ----------
        x : array-like, matrix object of shape (n_features) or (n_samples, n_features) Feature values.
        Returns
        -------
        y : tensor of shape (n_samples, n_labels) if not with_logits else tensor of shape (n_samples).
        Predicted label values.
        """
        x = to_tensor(x).float()
        return self.net(x)
    
    def fit(self, x, y):
        """
        Fits model to features and labels.
        Parameters
        ----------
        x : array-like, matrix object of shape (n_features) or (n_samples, n_features) Feature values.
        y : array-like, matrix object of shape (n_labels) or (n_samples, n_labels) Label values.
        Returns
        -------
        self : instance of self.
        """
        x = to_tensor(x).float()

        if not self.is_classifier:
            y = to_tensor(y).float()
        if self.is_classifier:
            y = to_tensor(y).long().argmax(1) if self.has_logits else to_tensor(y).long().view(-1)

        dataset = TensorDataset(x, y)
        dataloader = DataLoader(dataset, batch_size=100, shuffle=True)

        epoch = 0
        for epoch in range(self.num_epochs):
            for x_batch, y_batch in dataloader:
                self.train()

                y_pred = self.predict(x_batch)
                loss = self.criterion(y_pred, y_batch)
                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()
            
        return self

    def score(self, x, y):
        """
        Returns the score given by the assigned criterion.
        Called by sklearn's gridsearchcv for evaluations of models.
        Parameters
        ----------
        x : array-like, matrix object of shape (n_features) or (n_samples, n_features) Feature values.
        y : array-like, matrix object of shape (n_labels) or (n_samples, n_labels) Label values.
        Returns
        -------
        float : score of model prediction.
        """
        x = to_tensor(x).float()

        if not self.is_classifier:
            y = to_tensor(y).float()
        if self.is_classifier:
            y = to_tensor(y).long().argmax(1) if self.has_logits else to_tensor(y).long().view(-1)

        y_pred = self.predict(x)
        return self.criterion(y_pred, y).item() if self.is_classifier else -self.criterion(y_pred, y).item()

    def accuracy(self, x, y):
        """
        Returns mean absolute error loss if regressor and accuracy percentage if classifier
        Parameters
        ----------
        x : array-like, matrix object of shape (n_features) or (n_samples, n_features) Feature values.
        y : array-like, matrix object of shape (n_labels) or (n_samples, n_labels) Label values.
        Returns
        -------
        float : accuracy of model prediction.
        """
        x = to_tensor(x).float()

        if not self.is_classifier:
            y = to_tensor(y).float()
            y_pred = self.predict(x)
            return nn.L1Loss()(y_pred, y).item()
        
        if self.is_classifier:
            y = to_tensor(y).long().argmax(1) if self.has_logits else to_tensor(y).long().view(-1)
            y_pred = self.predict(x).argmax(1)
            return (y_pred == y).float().mean().item()

    def get_params(self, deep=True):
        """
        Returns model parameters.
        Called by sklearn's gridsearchcv for cloning of models.
        Returns
        -------
        params : dict of model parameters.
        
        """
        return {'net': self.net,
                'is_classifier': self.is_classifier,
                'criterion': self.criterion,
                'has_logits': self.has_logits,
                'learn_rate': self.learn_rate,
                'num_epochs': self.num_epochs}

    def set_params(self, **params):
        """
        Sets model parameters.
        Called by sklearn's gridsearchcv for parameter testing of models.
        Returns
        -------
        self : instance of self.
        """
        for key, value in params.items():
            setattr(self, key, value)
            if key == 'learn_rate':
                for group in self.optimizer.param_groups:
                    group['lr'] = value

        return self

class Reshape(nn.Module):
    """
    A reshape layer for torch, intended to be used with an ``nn.Sequential`` model.
    Parameters
    ----------
    out_dim : list or tuple The resulting output shape 
    """
    def __init__(self, out_dim):
        super(Reshape, self).__init__()
        self.out_dim = out_dim

    def forward(self, x):
        """
        Automatically called when calling the model.
        Parameters
        ----------
        x : A torch tensor
        Returns
        -------
        torch tensor : reshaped tensor
        """
        return x.view(*self.out_dim)
