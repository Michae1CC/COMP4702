import os
import pandas as pd
import numpy as np
from pprint import pprint
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from exam_utils import *


def load_data():
    NUTRITION_PATH = os.path.join(
        os.getcwd(), "data", "FoodNutrients1.csv")
    df = pd.read_csv(NUTRITION_PATH)
    drop_high_nan(df)
    # Drop any remaining rows that have nan values
    df.dropna(axis='rows', inplace=True)
    # Drop the food name
    df.drop(columns="Food Name", inplace=True)
    df = expand_col(df, "Public Food Key")
    df["Classification"] = df["Classification"].astype(int).astype(str)
    df = expand_col(df, "Classification")
    # predict energy
    y = df["Energy, with dietary fibre"]
    df.drop(columns=["Energy, with dietary fibre",
            "Energy, without dietary fibre"], inplace=True)

    vis_feats = [
        'Moisture (water)', 'Protein', 'Nitrogen', 'Total Fat', 'Ash',
        'Total dietary fibre', 'Alcohol', 'Fructose', 'Glucose', 'Sucrose',
        'Maltose', 'Lactose', 'Total sugars', 'Added sugars', 'Free sugars',
        'Starch', 'Available carbohydrate, without sugar alcohols',
        'Available carbohydrate, with sugar alcohols', 'Calcium (Ca)',
        'Iodine (I)', 'Iron (Fe)', 'Magnesium (Mg)', 'Phosphorus (P)',
        'Potassium (K)', 'Selenium (Se)', 'Sodium (Na)', 'Zinc (Zn)',
        'Retinol (preformed vitamin A)', 'Beta-carotene',
        'Beta-carotene equivalents (provitamin A)',
        'Vitamin A retinol equivalents', 'Thiamin (B1)',
        'Riboflavin (B2)', 'Niacin (B3)'
    ]

    df = df.apply(pd.to_numeric)

    # tabulate_describtion(df[vis_feats])

    # for c in chunk(vis_feats, 4):
    #     boxplots(df, c, log_y=True)

    # df[vis_feats] = StandardScaler().fit_transform(df[vis_feats])
    # covar_matrix(df, feats=vis_feats)

    # print(list(df.columns))
    X = df.to_numpy().astype(float)
    X = StandardScaler().fit_transform(X)
    # print(X)

    return X, y.to_numpy().astype(float)


def dim_red():
    X, y = load_data()
    # graph_scree(X)
    for tech in ["ISO", "TSNE"]:
        graph_reduced_dimensions(
            X,
            y,
            reg=True, method=tech
        )

    return


def gpr_model():
    X, y = load_data()
    model = TSNE(n_components=2)
    X_fitted = model.fit_transform(X)
    n, d = X_fitted.shape
    X_train_np, X_test_np, y_train_np, y_test_np, ind_train, ind_test = train_test_split(
        X_fitted, y, np.arange(n), test_size=0.2)
    # CV_onevar_sklearn(GaussianProcessRegressor, X_fitted, y, "alpha",
    #                   np.logspace(-4, 1.5, 10, base=10, dtype=float)
    #                   )
    model_params = {
        "alpha": 10e-3
    }
    plot_final_errors_sklearn(X_fitted, y, GaussianProcessRegressor,
                              model_kwargs=model_params, reg=True)
    # reg = GaussianProcessRegressor(**model_params)
    # reg.fit(X_train_np, y_train_np)
    # y_train_pred = reg.predict(X_train_np)
    # y_test_pred = reg.predict(X_test_np)
    # true_vs_pred(X_fitted, y, X_train_np, y_train_pred,
    #              X_test_np, y_test_pred, reg=True)
    return


def nn_model():
    import torch
    import tensorflow as tf
    X, y = load_data()
    n, d = X.shape
    X_train_np, X_test_np, y_train_np, y_test_np, ind_train, ind_test = train_test_split(
        X, y, np.arange(n), test_size=0.2)
    n, d = X_train_np.shape

    X_train = torch.tensor(X_train_np)
    X_test = torch.tensor(X_test_np)
    y_train = torch.tensor(y_train_np)
    y_test = torch.tensor(y_test_np)

    train_ds = tf.data.Dataset.from_tensor_slices(
        (X_train, y_train)).shuffle(10000).batch(250)

    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(250)
    num_epochs = 60
    lr = 10e-3

    layers = [
        tf.keras.layers.Dense(d, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1),
    ]

    comp_args = {
        "optimizer": tf.keras.optimizers.Adam(lr),
        "loss": "mse",
        "metrics": [tf.keras.metrics.MeanSquaredError()]
    }

    # plot_lr_errors(X, y, layers, comp_args, num_epochs=num_epochs)

    model = tf.keras.models.Sequential(layers)
    model.compile(**comp_args)
    history = model.fit(train_ds, epochs=num_epochs,
                        verbose=0, validation_data=test_ds)

    # tf.keras.utils.plot_model(
    #     model,
    #     to_file=os.path.join(
    #         os.getcwd(), "docs", "nutrients", "nutrients_model.png"),
    #     show_shapes=False,
    #     show_dtype=False,
    #     show_layer_names=True,
    #     rankdir="TB",
    #     expand_nested=False,
    #     dpi=96,
    #     layer_range=None,
    #     show_layer_activations=False,
    # )

    # data = [
    #     (np.arange(1, num_epochs + 1, 1, dtype=int),
    #      history.history['loss'], "Training"),
    #     (np.arange(1, num_epochs + 1, 1, dtype=int),
    #      history.history['val_loss'], "Validation")
    # ]
    # plot_meteric(data, "Model Loss", "Epoch", "Loss", log_y=True)
    r"""
    \begin{tabular}{rrr}
    \hline
    Repetition &   Train MSE &   Test MSE \\
    \hline
                1 &    315.635  & 16630.6    \\
                2 &    169.279  &   533.503  \\
                3 &    133.298  &   550.245  \\
                4 &     79.8252 &   178.792  \\
                5 &     66.5235 &   197.494  \\
                6 &     62.2626 &   218.932  \\
                7 &     36.0765 &   146.141  \\
                8 &     26.6836 &   130.57   \\
                9 &    138.952  &   247.078  \\
            10 &     12.1462 &    75.9735 \\
    \hline
    \end{tabular}
    """
    plot_final_errors(X, y, layers, comp_args,
                      num_epochs=num_epochs, reg=True)
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
    # load_data()
    # dim_red()
    # gpr_model()
    nn_model()


if __name__ == "__main__":
    main()
