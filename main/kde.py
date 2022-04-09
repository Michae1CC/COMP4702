#!/usr/bin/env python3

__author__ = 'Michael Ciccotosto-Camp'
__version__ = ''

import os

import numpy as np
from sklearn.neighbors import KernelDensity

from data import load_data


def iris_example():
    """
    Examples of using density estimation using the iris data set. The example
    is mostly the week 3 prac demo question.
    """
    # Load the iris dataset, but only use the first feature. Note that x has
    # to be a column vector to fit to the KernelDensity.
    x = load_data("iris", labels=False)
    x = x[:, 0].reshape(-1, 1)
    x_min, x_max = np.min(x) - 2, np.max(x) + 2
    kde_model = KernelDensity(kernel='gaussian', bandwidth=0.5)
    kde_model.fit(x)
    x_range = np.linspace(x_min, x_max, 100).reshape(-1, 1)
    log_de = kde_model.score_samples(x_range)
    de = np.exp(log_de)

    from matplotlib import pyplot as plt
    import seaborn as sns
    plt.style.use("fast")
    sns.set_theme(style="whitegrid")
    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    plt.plot(x_range.squeeze(), de.squeeze(), color=r"#0000FF", linewidth=1.7)
    plt.plot(x.squeeze(), np.full_like(x, -0.01), '|k', markeredgewidth=1)
    plt.xlabel("x", fontsize=12, fontweight="bold")
    plt.ylabel("Density Estimation", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()


def main():
    iris_example()


if __name__ == "__main__":
    main()
