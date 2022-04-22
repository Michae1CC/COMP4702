#!/usr/bin/env python3

__author__ = 'Michael Ciccotosto-Camp'
__version__ = ''

import torch
import torchvision.datasets as datasets
import numpy as np


def mnist_example():
    """
    Examples of using neural networks on the cifar10 data set. Similar to
    week 6 assignment question.
    """
    train_tensor = datasets.MNIST(root='./data', train=True, download=True)
    test_tensor = datasets.MNIST(root='./data', train=False, download=True)
    X_train = train_tensor.data.numpy().astype(int)
    y_train = train_tensor.targets.numpy().astype(int).squeeze()
    X_test = test_tensor.data.numpy().astype(int)[:5000]
    y_test = test_tensor.targets.numpy().astype(int).squeeze()[:5000]
    n_train, l, _ = X_train.shape
    n_test, l, _ = X_test.shape
    d = l * l
    X_train = X_train.reshape(n_train, d) / 255
    X_test = X_test.reshape(n_test, d) / 255
    # Get the number of classes
    n_classes = list(np.unique(y_train))
    n_classes = len(n_classes)
    # We shall build a neural network with 2 hidden layers, each with 100
    # nodes and with ReLu activation functions
    hl1 = 30
    # Create a list with the size of each layer
    torch_layers = (
        [torch.nn.Linear(d, hl1)]
        + [torch.nn.Sigmoid()]
        # Add an output layer
        + [torch.nn.Linear(hl1, n_classes)]
    )
    # The Sequential function essentially constructs our model and returns it
    # as an object.
    nn_model = torch.nn.Sequential(*torch_layers)
    print("Model Summary:")
    print(nn_model)
    # The number of datapoints per batch that we do
    batch_size = 512
    # The number of batches that we train on
    epochs = int(1e4)
    # Define a loss function to use for training
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    # starting learning rate that we can tweak to increase performance
    learning_rate = 1e-3
    # model.parameters gives the weight matrices and biases to the optimiser (AKA trainable parameters)
    optimiser = torch.optim.Adam(nn_model.parameters(), lr=learning_rate)
    metrics = {
        "iter": [],
        "loss": [],
        "train_acc": [],
        "val_acc": []
    }
    print(f"Completed: ", end="")
    for i in range(epochs + 1):
        # Randomly samples for this batch
        idx = np.random.randint(0, n_train, size=batch_size)
        X_batch = X_train[idx]
        y_batch = y_train[idx]
        # Predict the classes for the batch inputs
        y_pred = nn_model(torch.from_numpy(X_batch).float())
        # Compute the loss by comparing the predicted labels vs the actual labels
        loss = criterion(y_pred, torch.from_numpy(
            y_batch).type(torch.LongTensor))
        # Zero the gradients held by the optimiser
        optimiser.zero_grad()
        # Perform a backward pass to compute the gradients
        loss.backward()
        # Update the weights
        optimiser.step()
        # record the metrics every 1000 steps
        if i % 500 == 0:
            print(f"{i},", end="", flush=True)
            train_pred = nn_model(torch.from_numpy(X_batch).float())
            val_pred = nn_model(torch.from_numpy(X_test).float())
            train_accuracy = torch.mean(
                (train_pred.argmax(dim=1) == torch.from_numpy(y_batch)).float())
            val_accuracy = torch.mean(
                (val_pred.argmax(dim=1) == torch.from_numpy(y_test)).float())
            metrics["iter"].append(i)
            metrics["loss"].append(loss.item())
            metrics["val_acc"].append(val_accuracy.item())
            metrics["train_acc"].append(train_accuracy.item())

    print()
    from pprint import pprint
    pprint(metrics)

    return


def main():
    mnist_example()


if __name__ == "__main__":
    main()
