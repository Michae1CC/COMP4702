#!/usr/bin/env python3

__author__ = 'Michael Ciccotosto-Camp'
__version__ = ''

import numpy as np
from sklearn.manifold import Isomap

from data import load_data


def cifar10_example():
    """
    Examples of using isomapping on the cifar10 data set.
    """
    X, y = load_data("cifar10", labels=True)
    # Normalize the data
    X /= 255
    iso_model = Isomap(n_components=2)
    X_fitted = iso_model.fit_transform(X)
    classes = sorted(list(np.unique(y)))

    from matplotlib import pyplot as plt
    import seaborn as sns
    plt.style.use("fast")
    sns.set_theme(style="whitegrid")
    sns.set_style("whitegrid")
    colors = sns.color_palette("hls", 10)
    for cls_, clr in zip(classes, colors):
        cls_fitted = X_fitted[y == cls_]
        plt.scatter(cls_fitted[:, 0].squeeze(),
                    cls_fitted[:, 1].squeeze(), s=10, alpha=0.4, color=clr)
    plt.xlabel("Component 1", fontsize=12, fontweight="bold")
    plt.ylabel("Component 2", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()
    return


def swiss_roll_example():
    """
    Example of using isomapping on the swiss_roll data set.
    """
    # Reduce the number of dimensions from 3 to 2.
    X, y = load_data("swiss_roll", labels=True, n_samples=1500, noise=0.0)
    iso_model = Isomap(n_components=2)
    X_fitted = iso_model.fit_transform(X)

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(211, projection="3d")
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.Spectral)
    ax.set_title("Original data")

    ax = fig.add_subplot(212)
    ax.scatter(X_fitted[:, 0], X_fitted[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.axis("tight")
    plt.xticks([]), plt.yticks([])
    plt.title("Projected data")
    plt.show()

    return


def main():
    # cifar10_example()
    swiss_roll_example()


if __name__ == "__main__":
    main()
