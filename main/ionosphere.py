#!/usr/bin/env python3

__author__ = 'Michael Ciccotosto-Camp'
__version__ = ''

from multiprocessing import reduction
import numpy as np
import pandas as pd

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

from exam_utils import *


def load_iono_data():
    import os
    PATH = os.path.join(os.getcwd(), "data", "ionosphere.csv")
    df = pd.read_csv(PATH, header=None)
    # Drop rows with Nan values
    df = df.dropna(axis=0)
    out_le = LabelEncoder()
    df[34] = out_le.fit_transform(df[34])
    X = df.iloc[:, 0:-1].to_numpy(dtype=float)
    X = StandardScaler().fit_transform(X)
    y = df.iloc[:, -1].to_numpy(dtype=int)
    # print(X)
    # print(y)

    return X, y, out_le


def check_dim_reduction():
    X, y, out_le = load_iono_data()
    graph_scree(X)
    # graph_reduced_dimensions(
    #     X,
    #     y,
    #     encoder=out_le, reg=False, method="TSNE"
    # )
    return


def use_LG_model():
    X, y, out_le = load_iono_data()
    model = TSNE(n_components=2, init="random",
                 perplexity=20.0, n_iter=2000, n_iter_without_progress=300)
    X_tsne = model.fit_transform(X)
    # CV_heatmap_sklearn(LogisticRegression, X_tsne, y, "C", "tol",
    #                    np.logspace(-5, -1, 12, base=10, dtype=float),
    #                    np.logspace(-4, 2, 10, base=10, dtype=float)
    #                    )
    X_train, X_test, y_train, y_test = train_test_split(
        X_tsne, y, test_size=0.2)
    svc_model = LogisticRegression(C=0.0081, tol=0.01)
    svc_model.fit(X_train, y_train)
    y_pred_train = svc_model.predict(X_train)
    y_pred_test = svc_model.predict(X_test)
    true_vs_pred(X_tsne, y, X_train, y_pred_train,
                 X_test, y_pred_test, reg=False, classes=out_le.classes_)


def use_NN_model():
    import torch
    import tensorflow as tf
    X, y, out_le = load_iono_data()
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
    num_epochs = 200

    layers = [
        tf.keras.layers.Dense(d, activation='relu'),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(1)
    ]

    comp_args = {
        "optimizer": tf.keras.optimizers.Adam(1e-3),
        "loss": "binary_crossentropy",
        "metrics": ["accuracy"]
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
    #      history.history['accuracy'], "Training"),
    #     (np.arange(1, num_epochs + 1, 1, dtype=int),
    #      history.history['val_accuracy'], "Validation")
    # ]
    # plot_meteric(data, "Model Accuracy", "Epoch",
    #              "Mean Accuracy", log_y=True)

    tsne_model = TSNE(n_components=2, init="random",
                      perplexity=20.0, n_iter=2000, n_iter_without_progress=300)
    X_tsne = tsne_model.fit_transform(X)
    X_train_pca = X_tsne[ind_train]
    X_test_pca = X_tsne[ind_test]
    y_pred_train = np.round(model.predict(X_train_np))
    y_pred_test = np.round(model.predict(X_test_np))
    true_vs_pred(X_tsne, y, X_train_pca, y_pred_train,
                 X_test_pca, y_pred_test, reg=False, classes=out_le.classes_)


def main():
    # load_iono_data()
    # check_dim_reduction()
    # use_LG_model()
    use_NN_model()


if __name__ == "__main__":
    main()
