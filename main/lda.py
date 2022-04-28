#!/usr/bin/env python3

__author__ = 'Michael Ciccotosto-Camp'
__version__ = ''

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from data import load_data

"""
Linear discriminant analysis (LDA) is a supervised method for dimensionality
reduction for classification problems. Given samples from two classes 
C1 ... CN, we want to find the direction,
as defined by a vector w, such that when the data are projected onto
w, the examples from the two classes are as well separated as possible.

for the two classes to be well separated, we would like
the means to be as far apart as possible and the examples of classes be
scattered in as small a region as possible. So we want ||mi - mj||^2 to be
large. Fisher's linear discriminant is W that maximizes these differences.

Fisher's linear discriminant is optimal if the classes are normally 
distributed, but Fisher's linear discriminant can be used even when the classes
are not normal. Though LDA uses class separability as its goodness criterion, 
any classification method can be used in this new space for estimating the 
discriminants.

Sources:
Alpaydin Chp 6.6
"""


def cifar10_example():
    """
    Examples of using LDA on the cifar10 data set. 
    """
    X, y = load_data("cifar10", labels=True)
    # Normalize the data
    X /= 255
    lda_model = LDA(n_components=2)
    X_fitted = lda_model.fit_transform(X, y)
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


def main():
    cifar10_example()


if __name__ == "__main__":
    main()
