#!/usr/bin/env python3

__author__ = 'Michael Ciccotosto-Camp'
__version__ = ''

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.manifold import Isomap
from sklearn.manifold import TSNE

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from exam_utils import *


def load_abalone_data():
    import os
    PATH = os.path.join(os.getcwd(), "data", "abalone.csv")
    headers = "Sex,Length,Diameter,Height,Whole weight,Shucked weight,Viscera weight,Shell weight,Rings".split(
        sep=",")
    df = pd.read_csv(PATH, names=headers)
    # boxplots(df, "Shucked weight,Viscera weight,Shell weight,Rings".split(sep=","))
    y = df["Whole weight"]
    X = df["Sex,Length,Diameter,Height,Shucked weight,Viscera weight,Shell weight,Rings".split(
        sep=",")]
    sex_le = LabelEncoder()
    temp = sex_le.fit_transform(X["Sex"])
    X["Sex"] = temp
    # covar_matrix(X,
    #              feats="Sex,Length,Diameter,Height,Whole weight,Shucked weight,Viscera weight,Shell weight,Rings".split(
    #                  sep=",")
    #              )
    X = X.to_numpy(dtype=float)
    X = StandardScaler().fit_transform(X)

    return X, y.to_numpy(dtype=float), sex_le


def dim_red():
    X, y, sex_le = load_abalone_data()
    # graph_scree(X)
    # for tech in ["PCA"]:
    #     graph_reduced_dimensions(
    #         X,
    #         y,
    #         reg=True, method=tech
    #     )

    return


def lin_model():
    from sklearn.linear_model import LinearRegression
    X, y, sex_le = load_abalone_data()
    model = PCA(n_components=2)
    X_fitted = model.fit_transform(X)
    n, d = X_fitted.shape
    print(n, d)
    X_train_np, X_test_np, y_train_np, y_test_np, ind_train, ind_test = train_test_split(
        X_fitted, y, np.arange(n), test_size=0.2)

    r"""
    \begin{tabular}{rrr}
    \hline
    Repetition &   Train MSE &   Test MSE \\
    \hline
                1 &  0.00211079 & 0.00213318 \\
                2 &  0.00211079 & 0.00213318 \\
                3 &  0.00211079 & 0.00213318 \\
                4 &  0.00211079 & 0.00213318 \\
                5 &  0.00211079 & 0.00213318 \\
                6 &  0.00211079 & 0.00213318 \\
                7 &  0.00211079 & 0.00213318 \\
                8 &  0.00211079 & 0.00213318 \\
                9 &  0.00211079 & 0.00213318 \\
            10 &  0.00211079 & 0.00213318 \\
    \hline
    \end{tabular}
    """
    reg = LinearRegression().fit(X_train_np, y_train_np)
    print(mean_squared_error(y_test_np, reg.predict(X_test_np)))
    true_vs_pred(X_fitted, y, X_train_np, y_train_np,
                 X_test_np, y_test_np, reg=True)
    plot_final_errors_sklearn(X, y, LinearRegression, reg=True)
    return


def GPC_model():
    from sklearn.linear_model import LinearRegression
    X, y, sex_le = load_abalone_data()
    model = Isomap(n_components=2)
    X_fitted = model.fit_transform(X)
    n, d = X_fitted.shape
    print(n, d)
    X_train_np, X_test_np, y_train_np, y_test_np, ind_train, ind_test = train_test_split(
        X_fitted, y, np.arange(n), test_size=0.2)

    # CV_onevar_sklearn(GaussianProcessRegressor, X_fitted, y, "alpha",
    #                   np.logspace(-4, 1.5, 10, base=10, dtype=float)
    #                   )
    model_params = {
        "alpha": 10e-3
    }
    r"""
    \begin{tabular}{rrr}
    \hline
    Repetition &   Train MSE &   Test MSE \\
    \hline
                1 & 0.000557079 &  0.0189195 \\
                2 & 0.000557079 &  0.0189195 \\
                3 & 0.000557079 &  0.0189195 \\
                4 & 0.000557079 &  0.0189195 \\
                5 & 0.000557079 &  0.0189195 \\
                6 & 0.000557079 &  0.0189195 \\
                7 & 0.000557079 &  0.0189195 \\
                8 & 0.000557079 &  0.0189195 \\
                9 & 0.000557079 &  0.0189195 \\
                10 & 0.000557079 &  0.0189195 \\
    \hline
    \end{tabular}
    """
    # reg = GaussianProcessRegressor(**model_params)
    # reg.fit(X_train_np, y_train_np)
    # plot_final_errors_sklearn(X, y, GaussianProcessRegressor,
    #                           model_kwargs=model_params, reg=True)
    true_vs_pred(X_fitted, y, X_train_np, y_train_np,
                 X_test_np, y_test_np, reg=True)
    return


def nn_model():
    import torch
    import tensorflow as tf
    X, y, sex_le = load_abalone_data()
    n, d = X.shape
    X_train_np, X_test_np, y_train_np, y_test_np, ind_train, ind_test = train_test_split(
        X, y, np.arange(n), test_size=0.2)
    n, d = X_train_np.shape
    X_train = torch.tensor(X_train_np)
    X_test = torch.tensor(X_test_np)
    y_train = torch.tensor(y_train_np)
    y_test = torch.tensor(y_test_np)

    train_ds = tf.data.Dataset.from_tensor_slices(
        (X_train, y_train)).shuffle(10000).batch(32)

    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)
    num_epochs = 50
    lr = 10e-3

    layers = [
        tf.keras.layers.Dense(d, activation='relu'),
        tf.keras.layers.Dense(3, activation='relu'),
        tf.keras.layers.Dense(1),
    ]

    comp_args = {
        "optimizer": tf.keras.optimizers.Adam(lr),
        "loss": "mse",
        "metrics": [tf.keras.metrics.MeanSquaredError()]
    }

    # plot_lr_errors(X, y, layers, comp_args, num_epochs)

    model = tf.keras.models.Sequential(layers)
    model.compile(**comp_args)
    history = model.fit(train_ds, epochs=num_epochs,
                        validation_data=test_ds, verbose=0)
    # data = [
    #     (np.arange(1, num_epochs + 1, 1, dtype=int),
    #      history.history['loss'], "Training"),
    #     (np.arange(1, num_epochs + 1, 1, dtype=int),
    #      history.history['val_loss'], "Validation")
    # ]
    # plot_meteric(data, "Model Loss", "Epoch", "Loss", log_y=True)

    # data = [
    #     (np.arange(1, num_epochs + 1, 1, dtype=int),
    #      history.history['mean_squared_error'], "Training"),
    #     (np.arange(1, num_epochs + 1, 1, dtype=int),
    #      history.history['val_mean_squared_error'], "Validation")
    # ]
    # plot_meteric(data, "Model MSE", "Epoch",
    #              "MSE", log_y=True)
    r"""
    \begin{tabular}{rrr}
    \hline
    Repetition &   Train MSE &   Test MSE \\
    \hline
                1 &   0.0858929 &  0.0856366 \\
                2 &   0.0858611 &  0.0864535 \\
                3 &   0.0858439 &  0.086245  \\
                4 &   0.0855823 &  0.0856442 \\
                5 &   0.0855419 &  0.0858985 \\
                6 &   0.0856757 &  0.0863496 \\
                7 &   0.0856319 &  0.0860724 \\
                8 &   0.0855988 &  0.0857312 \\
                9 &   0.0859    &  0.0859951 \\
            10 &   0.08555   &  0.0855366 \\
    \hline
    \end{tabular}
    """
    # plot_final_errors(X, y, layers, comp_args,
    #                   num_epochs=num_epochs, reg=True)
    tsne_model = TSNE(n_components=2, init="random",
                      perplexity=20.0, n_iter=2000, n_iter_without_progress=300)
    X_tsne = tsne_model.fit_transform(X)
    X_train_pca = X_tsne[ind_train]
    X_test_pca = X_tsne[ind_test]
    y_pred_train = (model.predict(X_train_np))
    y_pred_test = (model.predict(X_test_np))
    true_vs_pred(X_tsne, y, X_train_pca, y_pred_train,
                 X_test_pca, y_pred_test, reg=True)


def main():
    # load_abalone_data()
    # dim_red()
    # lin_model()
    # GPC_model()
    nn_model()


if __name__ == "__main__":
    main()
