#!/usr/bin/env python3

__author__ = 'Michael Ciccotosto-Camp'
__version__ = ''

import numpy as np
from sklearn.manifold import Isomap

from data import load_data

"""
Isometric feature mapping (Isomap) (Tenenbaum, de Silva, and Langford
2000) estimates this distance and applies multidimensional scaling (MDS)
(section 6.5), using it for dimensionality reduction. Isomap uses the geodesic distances between all pairs of data points.
For neighboring points that are close in the input space, Euclidean distance
can be used; for small changes in pose, the manifold is locally
linear. For faraway points, geodesic distance is approximated by the sum
of the distances between the points along the way over the manifold.
This is done by defining a graph whose nodes correspond to the N data
points and whose edges connect neighboring points (those with distance
less than some \eps or one of the n nearest) with weights corresponding
to Euclidean distances. The geodesic distance between any two points is
calculated as the length of the shortest path between the corresponding
two nodes. For two points that are not close by, we need to hop over a
number of intermediate points along the way, and therefore the distance
will be the distance along the manifold, approximated as the sum of local
Euclidean distances. It is clear that the graph distances provide a better approximation as
the number of points increases, though there is the trade-off of longer
execution time; if time is critical, one can subsample and use a subset
of "landmark points" to make the algorithm faster.

One problem with Isomap, as with MDS, is that it places the N points in
a low-dimensional space, but it does not learn a generalmapping function
that will allow mapping a new test point; the new point should be added
to the dataset and the whole algorithm needs to be run once more using
N + 1 instances.

Isomap is a nonlinear dimensionality reduction method. It is a manifold 
learning algorithm which tries to preserve the geodesic 
distance between samples while reducing the dimension.
Isomap starts by creating a neighborhood network. After that, it uses 
graph distance to the approximate geodesic distance between all pairs of 
points. And then, through eigenvalue decomposition of the geodesic distance 
matrix, it finds the low dimensional embedding of the dataset. 
In non-linear manifolds, the Euclidean metric for distance holds good if a
nd only if neighborhood structure can be approximated as linear. If 
neighborhood contains holes, then Euclidean distances can be highly 
misleading. In contrast to this, if we measure the distance between two 
points by following the manifold, we will have a better approximation of 
how far or near two points are. Let's understand this with an extremely 
simple 2-D example. Suppose our data lies on a circular manifold in a 
2-D structure like in the image below.

Alpaydin Chp 6.6
https://blog.paperspace.com/dimension-reduction-with-isomap/
https://medium.com/data-science-in-your-pocket/dimension-reduction-using-isomap-72ead0411dec
"""


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
    cifar10_example()
    # swiss_roll_example()


if __name__ == "__main__":
    main()
