#!/usr/bin/env python3

__author__ = 'Michael Ciccotosto-Camp'
__version__ = ''

import numpy as np

from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from data import load_data


def iris_example():
    """
    Example of using mean-shift clustering estimation using the iris data set.
    """
    x, labels = load_data("iris", labels=True)
    # A bit of trial and error needed before getting a suitable bandwidth
    ms_model = MeanShift(bandwidth=0.85, max_iter=300)
    ms_model.fit(x)
    # Get the predicted labels of each class after fitting
    pred_labels = ms_model.labels_
    n_cls = len(np.unique(pred_labels))

    # Graph the predictions alongside the actual labels
    fig = plt.figure(figsize=plt.figaspect(0.5))
    axs = fig.add_subplot(1, 2, 1, projection='3d')
    axs.scatter(x[:, 3], x[:, 0], x[:, 2],
                c=labels.astype(float), edgecolor="k")
    axs.w_xaxis.set_ticklabels([])
    axs.w_yaxis.set_ticklabels([])
    axs.w_zaxis.set_ticklabels([])
    axs.set_xlabel("Petal width")
    axs.set_ylabel("Sepal length")
    axs.set_zlabel("Petal length")
    axs.set_title("True labels")
    axs.dist = 12

    axs = fig.add_subplot(1, 2, 2, projection='3d')
    axs.scatter(x[:, 3], x[:, 0], x[:, 2],
                c=pred_labels.astype(float), edgecolor="k")
    axs.w_xaxis.set_ticklabels([])
    axs.w_yaxis.set_ticklabels([])
    axs.w_zaxis.set_ticklabels([])
    axs.set_xlabel("Petal width")
    axs.set_ylabel("Sepal length")
    axs.set_zlabel("Petal length")
    axs.set_title(f"Predicted labels ({n_cls=})")
    axs.dist = 12
    plt.show()

    return


def main():
    iris_example()


if __name__ == "__main__":
    main()
