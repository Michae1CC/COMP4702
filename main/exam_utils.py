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
    plt.cla()
    plt.clf()
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
    plt.cla()
    plt.clf()
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
    plt.cla()
    plt.clf()
    return


def covar_matrix(X, feats):

    data = X[feats].to_numpy()
    # print(data.shape)
    pcc = np.corrcoef(data, rowvar=False)
    # print(pcc)
    grid_kws = {"height_ratios": (.9, .05), "hspace": .3}
    f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws)
    ax = sns.heatmap(np.abs(pcc), ax=ax,
                     cbar_ax=cbar_ax,
                     cbar_kws={"orientation": "horizontal",
                     'label': 'Variance'})
    ax.set_title("Absolute Covariance")
    ax.set_xticklabels(feats)
    ax.set_yticklabels(feats, rotation=0, fontsize="10", va="center")
    # plt.tight_layout()
    plt.show()
    plt.cla()
    plt.clf()
    return


def plot_meteric(data, title, x_name, y_name):
    fig = plt.gcf()
    ax = plt.gca()
    fig.set_size_inches(9, 6)
    plt.style.use("fast")
    sns.set_theme(style="whitegrid")
    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    plt.yscale("log")
    legends = []

    for x, y, name in data:
        plt.plot(x, y, marker="o", markersize=4, linewidth=1.7)
        legends.append(name.replace('_', '-'))

    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel(x_name, fontsize=16, fontweight="bold")
    ax.set_ylabel(y_name, fontsize=16, fontweight="bold")
    ax.tick_params(labelsize=14)
    plt.grid(visible=True, which='major', color='black',
             linestyle='-', alpha=0.3, linewidth=0.2)
    plt.legend(legends, edgecolor="black")
    plt.show()
    plt.cla()
    plt.clf()
    return


def graph_reduced_dimensions(X, y, encoder=None, reg=True, method="PCA"):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    fig = plt.gcf()
    ax = plt.gca()
    model = None
    X_fitted = None
    if method == "PCA":
        model = PCA(n_components=2)
        X_fitted = model.fit_transform(X)
    elif method == "TSNE":
        model = TSNE(n_components=2, init="random",
                     perplexity=20.0, n_iter=1000, n_iter_without_progress=300)
        X_fitted = model.fit_transform(X)
    elif method == "LDA":
        model = LDA(n_components=2)
        X_fitted = model.fit_transform(X, y)
    else:
        raise NotImplementedError(f"No transformation method for {method:r}.")

    ax.set_title(f"Projected data via {method}")
    vmin, vmax = np.min(y), np.max(y)
    if reg:
        im = ax.scatter(X_fitted[:, 0], X_fitted[:, 1], c=y,
                        cmap=plt.cm.Spectral, vmin=vmin, vmax=vmax,
                        marker="D")
        plt.colorbar(im, orientation="horizontal")
    else:
        n = len(np.unique(y))
        legend = []
        colors = sns.color_palette("hls", n)
        for cls_, clr in zip(range(n), colors):
            cls_fitted = X_fitted[y == cls_]
            plt.scatter(cls_fitted[:, 0].squeeze(),
                        cls_fitted[:, 1].squeeze(), s=20, alpha=0.9, color=clr)
            legend.append(encoder.inverse_transform([cls_])[0])
    plt.legend(legend)
    plt.show()
    plt.cla()
    plt.clf()
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
