#!/usr/bin/env python3

__author__ = 'Michael Ciccotosto-Camp'
__version__ = ''

import torch.nn as nn
import torchvision.datasets as datasets
import numpy as np


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)

        def forward(self, x):
            x = self.conv1(x)
            # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
            x = self.conv2(x)
            x = x.view(x.size(0), -1)
            output = self.out(x)
            return output, x    # return x for visualization


def mnist_example():
    """
    Examples of using neural networks on the cifar10 data set.
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
    X_train = X_train / 255
    X_test = X_test / 255
    # Get the number of classes
    n_classes = list(np.unique(y_train))
    n_classes = len(n_classes)
    print(f"{n_train=} {l=} {d=}")
