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
    torch_layers = (
        [torch.nn.Linear(d, hl1)]
        + [torch.nn.LeakyReLU()]
        + [torch.nn.Linear(hl1, hl2)]
        + [torch.nn.LeakyReLU()]
        # Add an output layer
        + [torch.nn.Linear(hl2, n_classes)]
    )
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
    print(f"Completed: ", end="")
    for i in range(optimisation_steps + 1):
        # Randomly samples for this batch
        idx = np.random.randint(0, n, size=batch_size)
        X_batch = X[idx]
        y_batch = y[idx]
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
        # record the metrics every 1000 steps
        if i % 1000 == 0:
            print(f"{i},", end="", flush=True)
            train_pred = nn_model(torch.from_numpy(X_batch).float())
            val_pred = nn_model(torch.from_numpy(X).float())
            train_accuracy = torch.mean(
                (train_pred.argmax(dim=1) == torch.from_numpy(y_batch)).float())
            val_accuracy = torch.mean(
                (val_pred.argmax(dim=1) == torch.from_numpy(y)).float())
            metrics["iter"].append(i)
            metrics["loss"].append(loss.item())
            metrics["val_acc"].append(val_accuracy.item())
            metrics["train_acc"].append(train_accuracy.item())

    print()
    from pprint import pprint
    pprint(metrics)


def main():
    cifar10_example()


if __name__ == "__main__":
    main()
