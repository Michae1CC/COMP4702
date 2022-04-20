#!/usr/bin/env python3

__author__ = 'Michael Ciccotosto-Camp'
__version__ = ''

import numpy as np

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from data import load_data


"""
K-means algorithm is for clustering, that is, for finding groups in the
data, where the groups are represented by their centers, which are the
typical representatives of the groups. The goal of this algorithm is to find 
groups in the data, with the number of groups represented by the variable K. 
The algorithm works iteratively to assign each data point to one of K 
groups based on the features that are provided. Data points are clustered 
based on feature similarity. The results of the K-means clustering 
algorithm are:

- The centroids of the K clusters, which can be used to label new data

- Labels for the training data (each data point is assigned to a single cluster)

Rather than defining groups before looking at the data, clustering allows you 
to find and analyze the groups that have formed organically.

Each centroid of a cluster is a collection of feature values which 
define the resulting groups. Examining the centroid feature weights can 
be used to qualitatively interpret what kind of group each cluster represents.  

Sources:
Alpaydin Chp 7.3
https://blogs.oracle.com/ai-and-datascience/post/introduction-to-k-means-clustering
"""


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
