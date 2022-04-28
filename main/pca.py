#!/usr/bin/env python3

__author__ = 'Michael Ciccotosto-Camp'
__version__ = ''

import os

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from data import load_data

"""
In projection methods, we are interested in finding a mapping from the
inputs in the original d-dimensional space to a new (k < d)-dimensional
space, with minimum loss of information. Principal components analysis (PCA) is an unsupervised method in that
it does not use the output information; the criterion to be maximized is
the variance. The principal component is w1 such that the sample, after
projection on to w1, is most spread out so that the difference between
the sample points becomes most apparent.

The principal components of a collection of points in a real coordinate space 
are a sequence of p unit vectors, where the 
i-th vector is the direction of a line that best fits the data while 
being orthogonal to the first i-1 vectors. Here, a best-fitting line is 
defined as one that minimizes the average squared distance from the 
points to the line. These directions constitute an orthonormal basis 
in which different individual dimensions of the data are linearly 
uncorrelated. Principal component analysis (PCA) is the process of 
computing the principal components and using them to perform a change 
of basis on the data, sometimes using only the first few principal 
components and ignoring the rest.

Such dimensionality reduction can be a very useful step for visualising 
and processing high-dimensional datasets, while still retaining as much 
of the variance in the dataset as possible. For example, selecting L = 2 
and keeping only the first two principal components finds the 
two-dimensional plane through the high-dimensional dataset in which the 
data is most spread out, so if the data contains clusters these too may 
be most spread out, and therefore most visible to be plotted out in a 
two-dimensional diagram; whereas if two directions through the data 
(or two of the original variables) are chosen at random, the clusters 
may be much less spread apart from each other, and may in fact be much 
more likely to substantially overlay each other, making them indistinguishable.

Similarly, in regression analysis, the larger the number of explanatory 
variables allowed, the greater is the chance of overfitting the model, 
producing conclusions that fail to generalise to other datasets. 
One approach, especially when there are strong correlations between 
different possible explanatory variables, is to reduce them to a few 
principal components and then run the regression against them, a method 
called principal component regression.

Dimensionality reduction may also be appropriate when the variables 
in a dataset are noisy. If each column of the dataset contains 
independent identically distributed Gaussian noise, then the columns 
of T will also contain similarly identically distributed Gaussian 
noise (such a distribution is invariant under the effects of the 
matrix W, which can be thought of as a high-dimensional rotation of the 
co-ordinate axes). However, with more of the total variance concentrated 
in the first few principal components compared to the same noise variance, 
the proportionate effect of the noise is less—the first few components 
achieve a higher signal-to-noise ratio. PCA thus can have the effect 
of concentrating much of the signal into the first few principal 
components, which can usefully be captured by dimensionality reduction; 
while the later principal components may be dominated by noise, and so 
disposed of without great loss. If the dataset is not too large, the 
significance of the principal components can be tested using parametric 
bootstrap, as an aid in determining how many principal components to retain.

Sources:
https://en.wikipedia.org/wiki/Principal_component_analysis#Dimensionality_reduction
"""


def cifar10_example():
    """
    Examples of using PCA on the cifar10 data set. Similar to
    week 6 assignment question.
    """
    X, y = load_data("cifar10", labels=True)
    # Normalize the data
    X /= 255
    pca_model = PCA(n_components=2)
    X_fitted = pca_model.fit_transform(X)
    classes = sorted(list(np.unique(y)))

    # Should hopefully look something like this!:
    # https://res.cloudinary.com/dyd911kmh/image/upload/f_auto,q_auto:best/v1635528091/output_128_1_ceaxep.png
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
    plt.xlabel("PC 1", fontsize=12, fontweight="bold")
    plt.ylabel("PC 2", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()
    return


def main():
    cifar10_example()


if __name__ == "__main__":
    main()
