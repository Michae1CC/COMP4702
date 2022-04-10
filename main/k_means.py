#!/usr/bin/env python3

__author__ = 'Michael Ciccotosto-Camp'
__version__ = ''

import numpy as np

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from data import load_data


def iris_example():
    """
    Examples of using k-means using the iris data set.
    """
    x, labels = load_data("iris", labels=True)
    # Get the number of classes from the dataset
    n_clusters = len(np.unique(labels))
    km_model = KMeans(init="k-means++", n_init=15,
                      max_iter=200, tol=1e-4, n_clusters=n_clusters)
    km_model.fit(x)
    # Get the predicted labels of each class after fitting
    pred_labels = km_model.labels_

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
    axs.set_title("Predicted labels")
    axs.dist = 12
    plt.show()

    return


def main():
    iris_example()


if __name__ == "__main__":
    main()
