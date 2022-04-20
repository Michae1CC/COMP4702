#!/usr/bin/env python3

__author__ = 'Michael Ciccotosto-Camp'
__version__ = ''

import torch
import torch.optim as optim
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from data import load_data


"""
An autoencoder is a type of artificial neural network used to learn 
efficient codings of unlabeled data (unsupervised learning). The encoding 
is validated and refined by attempting to regenerate the input from the 
encoding. The autoencoder learns a representation (encoding) for a set of 
data, typically for dimensionality reduction, by training the network to 
ignore insignificant data ("noise").

The two main applications of autoencoders are dimensionality reduction and 
information retrieval.
"""


class autoencode_IR(torch.nn.Module):
    def __init__(self, d, latent_dim):
        super(autoencode_IR, self).__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(d, latent_dim), torch.nn.ReLU())
        self.decoder = torch.nn.Linear(latent_dim, d)

    def forward(self, x):
        latent = self.encoder(x)
        decoded = self.decoder(latent)
        decoded = torch.sigmoid(decoded)
        return decoded, latent


def ir_example():
    """
    Examples of using an auto encoder on the ir data set.
    """
    X = load_data("ir_data", labels=False)
    # Normalize the data
    X /= np.max(X)
    # Get the number of samples and the dimension of each input
    n, d = X.shape
    # Use a latent layer that uses only 15% of the size of the input layer
    latent_dim = round(d * 0.15)
    # The Sequential function essentially constructs our model and returns it
    # as an object.
    auto_en = autoencode_IR(d, latent_dim)
    X_in, X_out = torch.tensor(
        X, dtype=torch.float), torch.tensor(X, dtype=torch.float)
    data_tuple = [[X_in[i], X_out[i]] for i in range(len(X_in))]
    batch_size = 25
    batch = torch.utils.data.DataLoader(data_tuple,
                                        batch_size=batch_size,
                                        shuffle=True)
    lr = 0.0005
    optimizer = optim.Adam(auto_en.parameters(),
                           lr=lr,
                           betas=(0.9, 0.999))
    # Define the number of epochs to use
    epochs = 100
    # Define a loss function to use for training
    loss_func = torch.nn.MSELoss()
    metrics = {
        "epoch": [],
        "loss": []
    }
    print(f"Epochs Completed: ", end="")
    for epoch in range(epochs):
        for batch_shuffle in batch:
            x, y = batch_shuffle
            optimizer.zero_grad()
            out, _ = auto_en(x)  # to get the decoded stuff only
            loss = loss_func(out, y)
            loss.backward()
            optimizer.step()
        # record the metrics every 10 steps
        if epoch % 10 == 0:
            print(f"{epoch + 1},", end="", flush=True)
            metrics["epoch"].append(epoch + 1)
            metrics["loss"].append(loss.item())

    print()
    from pprint import pprint
    pprint(metrics)


def main():
    ir_example()


if __name__ == "__main__":
    main()
