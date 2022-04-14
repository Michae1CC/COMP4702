#!/usr/bin/env python3

__author__ = 'Michael Ciccotosto-Camp'
__version__ = ''

import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from data import load_data


def cifar10_example():
    """
    Examples of using neural networks on the cifar10 data set. Similar to
    week 6 assignment question.
    """
    X, y = load_data("cifar10", labels=True)
    # Normalize the data
    X /= 255
    # Get the number of samples and the dimension of each input
    n, d = X.shape
    # Get the number of classes
    n_classes = list(np.unique(y.astype(int)))
    n_classes = len(n_classes)
    # We shall build a neural network with 2 hidden layers, each with 100
    # nodes and with ReLu activation functions
    hl1 = 100
    hl2 = 100
    # Create a list with the size of each layer
    layer_sizes = [d, hl1, hl2]
    torch_layers = []
    for i in range(len(layer_sizes)-1):
        torch_layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        torch_layers.append(torch.nn.ReLU())
    # Add an output layer
    torch_layers.append(torch.nn.Linear(layer_sizes[-1], n_classes))
    # The Sequential function essentially constructs our model and returns it
    # as an object.
    nn_model = torch.nn.Sequential(*torch_layers)
    # The number of datapoints per batch that we do
    batch_size = 64
    # The number of batches that we train on
    optimisation_steps = int(1e4)
    # Define a loss function to use for training
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    # starting learning rate that we can tweak to increase performance
    learning_rate = 1e-4
    # model.parameters gives the weight matrices and biases to the optimiser (AKA trainable parameters)
    optimiser = torch.optim.SGD(nn_model.parameters(), lr=learning_rate)
    metrics = {
        "iter": [],
        "loss": [],
        "train_acc": [],
        "val_acc": []
    }
    for i in range(optimisation_steps + 1):
        # Randomly samples for this batch
        idx = np.random.randint(0, n, size=batch_size)
        X_batch = X[idx]
        y_batch = y[idx]
        # print(X_batch.shape)
        # Predict the classes for the batch inputs
        y_pred = nn_model(torch.from_numpy(X_batch).float())
        # Compute the loss by comparing the predicted labels vs the actual labels
        loss = criterion(y_pred, torch.from_numpy(
            y[idx]).type(torch.LongTensor))
        # Zero the gradients held by the optimiser
        optimiser.zero_grad()
        # Perform a backward pass to compute the gradients
        loss.backward()
        # Update the weights
        optimiser.step()
        if i % 1000 == 0:
            train_pred = nn_model(torch.from_numpy(X_batch).float())
            val_pred = nn_model(torch.from_numpy(X).float())
            train_accuracy = torch.mean(
                (train_pred.argmax(dim=1) == torch.from_numpy(y_batch)).float())
            val_accuracy = torch.mean(
                (val_pred.argmax(dim=1) == torch.from_numpy(y)).float())
            # print the loss every 100 steps
            metrics["iter"].append(i)
            metrics["loss"].append(val_accuracy)
            metrics["val_acc"].append(val_accuracy)
            metrics["train_acc"].append(val_accuracy)

    print(metrics["val_acc"])


def main():
    cifar10_example()


if __name__ == "__main__":
    main()
