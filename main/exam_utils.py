#!/usr/bin/env python3

__author__ = 'Michael Ciccotosto-Camp'
__version__ = ''

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA


def true_vs_pred(X, y, X_train, y_train_pred, X_test, y_test_pred, reg=True, save_path=None, classes=None):
    """
    Graphs true values of y along side predictions.
    """

    fig = plt.figure()
    if reg:
        vmin, vmax = np.min(y), np.max(y)
        ax = fig.add_subplot(211)
        ax.set_title("True Value")
        ax.scatter(X[:, 0], X[:, 1], c=y,
                   cmap=plt.cm.Spectral, vmin=vmin, vmax=vmax, marker="^")

        ax = fig.add_subplot(212)
        ax.set_title("Predicted Value")
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train_pred,
                   cmap=plt.cm.Spectral, vmin=vmin, vmax=vmax,
                   marker="D")
        im = ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test_pred,
                        cmap=plt.cm.Spectral, vmin=vmin, vmax=vmax,
                        marker="o")
        plt.legend(["Training Set", "TestingSet"], edgecolor="black")
        plt.colorbar(im, orientation="horizontal")
    else:
        vmin, vmax = int(np.min(y)), int(np.max(y))
        n = np.unique(y)
        cmap = plt.cm.tab10
        # extract all colors from the .jet map
        cmaplist = [cmap(i) for i in n]
        # create the new map
        cmap = mpl.colors.LinearSegmentedColormap.from_list(
            'Custom cmap', cmaplist, cmap.N)
        ax = fig.add_subplot(211)
        ax.set_title("True Value")
        ax.scatter(X[:, 0], X[:, 1], c=y,
                   cmap=cmap, vmin=vmin, vmax=vmax, marker="^")

        ax = fig.add_subplot(212)
        ax.set_title("Predicted Value")
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train_pred,
                   cmap=cmap, vmin=vmin, vmax=vmax,
                   marker="D")
        im = ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test_pred,
                        cmap=cmap, vmin=vmin, vmax=vmax,
                        marker="o")
        plt.legend(["Training Set", "Testing Set"], edgecolor="black")
        cb = plt.colorbar(im, spacing='proportional', orientation="horizontal")
        loc = np.arange(0, vmax + 1, 1)
        cb.set_ticks(loc)
        cb.set_ticklabels(classes)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.cla()
    plt.clf()

    return


def graph_scree(X):
    _, s, _ = np.linalg.svd(X)
    plt.style.use('seaborn-deep')
    plt.figure(figsize=(9, 6))

    plt.plot(range(len(s)), s, 'bo--',
             markersize=5, linewidth=1)

    plt.xlabel("Rank")
    plt.ylabel("Value")
    plt.title("SCREE graph")

    plt.tight_layout()
    plt.show()
    return


def graph_confusion_matrix(y_true, y_pred, classes):
    from sklearn.metrics import confusion_matrix
    plt.title("Confusion Matrix")
    conf_mat = confusion_matrix(y_true, y_pred)
    ax = sns.heatmap(conf_mat, annot=True, fmt="d")
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    plt.tight_layout()
    plt.show()
    return


def boxplots(X, x, y):
    sns.set_theme(style="ticks")
    n = len(y)
    start_num = 0
    fig, axes = plt.subplots(1, n)
    for y_ in y:
        g = sns.boxplot(ax=axes[start_num], x=x, y=y_,
                        data=X
                        )
        start_num += 1
    plt.tight_layout()
    plt.show()

    return


def tests():

    # X = np.random.randn(200, 2)
    # y = np.round(np.random.randn(200))

    # X_test = np.random.randn(100, 2)
    # y_test_pred = np.round(np.random.randn(100))

    # X_train = np.random.randn(100, 2)
    # y_train_pred = np.round(np.random.randn(100))

    # true_vs_pred(X, y, X_train, y_train_pred, X_test, y_test_pred, reg=False)

    graph_scree(np.random.randn(200, 10))


def main():
    tests()


if __name__ == "__main__":
    main()
