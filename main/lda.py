#!/usr/bin/env python3

__author__ = 'Michael Ciccotosto-Camp'
__version__ = ''

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from data import load_data


def cifar10_example():
    """
    Examples of using PCA on the cifar10 data set. Similar to
    week 6 assignment question.
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
