import os
import pandas as pd
import numpy as np
from pprint import pprint
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from exam_utils import *

COL_NAMES = [
    "state",
    "county",
    "community",
    "communityname",
    "fold",
    "population",
    "householdsize",
    "racepctblack",
    "racePctWhite",
    "racePctAsian",
    "racePctHisp",
    "agePct12t21",
    "agePct12t29",
    "agePct16t24",
    "agePct65up",
    "numbUrban",
    "pctUrban",
    "medIncome",
    "pctWWage",
    "pctWFarmSelf",
    "pctWInvInc",
    "pctWSocSec",
    "pctWPubAsst",
    "pctWRetire",
    "medFamInc",
    "perCapInc",
    "whitePerCap",
    "blackPerCap",
    "indianPerCap",
    "AsianPerCap",
    "OtherPerCap",
    "HispPerCap",
    "NumUnderPov",
    "PctPopUnderPov",
    "PctLess9thGrade",
    "PctNotHSGrad",
    "PctBSorMore",
    "PctUnemployed",
    "PctEmploy",
    "PctEmplManu",
    "PctEmplProfServ",
    "PctOccupManu",
    "PctOccupMgmtProf",
    "MalePctDivorce",
    "MalePctNevMarr",
    "FemalePctDiv",
    "TotalPctDiv",
    "PersPerFam",
    "PctFam2Par",
    "PctKids2Par",
    "PctYoungKids2Par",
    "PctTeen2Par",
    "PctWorkMomYoungKids",
    "PctWorkMom",
    "NumIlleg",
    "PctIlleg",
    "NumImmig",
    "PctImmigRecent",
    "PctImmigRec5",
    "PctImmigRec8",
    "PctImmigRec10",
    "PctRecentImmig",
    "PctRecImmig5",
    "PctRecImmig8",
    "PctRecImmig10",
    "PctSpeakEnglOnly",
    "PctNotSpeakEnglWell",
    "PctLargHouseFam",
    "PctLargHouseOccup",
    "PersPerOccupHous",
    "PersPerOwnOccHous",
    "PersPerRentOccHous",
    "PctPersOwnOccup",
    "PctPersDenseHous",
    "PctHousLess3BR",
    "MedNumBR",
    "HousVacant",
    "PctHousOccup",
    "PctHousOwnOcc",
    "PctVacantBoarded",
    "PctVacMore6Mos",
    "MedYrHousBuilt",
    "PctHousNoPhone",
    "PctWOFullPlumb",
    "OwnOccLowQuart",
    "OwnOccMedVal",
    "OwnOccHiQuart",
    "RentLowQ",
    "RentMedian",
    "RentHighQ",
    "MedRent",
    "MedRentPctHousInc",
    "MedOwnCostPctInc",
    "MedOwnCostPctIncNoMtg",
    "NumInShelters",
    "NumStreet",
    "PctForeignBorn",
    "PctBornSameState",
    "PctSameHouse85",
    "PctSameCity85",
    "PctSameState85",
    "LemasSwornFT",
    "LemasSwFTPerPop",
    "LemasSwFTFieldOps",
    "LemasSwFTFieldPerPop",
    "LemasTotalReq",
    "LemasTotReqPerPop",
    "PolicReqPerOffic",
    "PolicPerPop",
    "RacialMatchCommPol",
    "PctPolicWhite",
    "PctPolicBlack",
    "PctPolicHisp",
    "PctPolicAsian",
    "PctPolicMinor",
    "OfficAssgnDrugUnits",
    "NumKindsDrugsSeiz",
    "PolicAveOTWorked",
    "LandArea",
    "PopDens",
    "PctUsePubTrans",
    "PolicCars",
    "PolicOperBudg",
    "LemasPctPolicOnPatr",
    "LemasGangUnitDeploy",
    "LemasPctOfficDrugUn",
    "PolicBudgPerPop",
    "ViolentCrimesPerPop",
]


def load_data():
    global COL_NAMES
    DATA_PATH = os.path.join(
        os.getcwd(), "data", "communities.csv")
    df = pd.read_csv(DATA_PATH)
    df.columns = COL_NAMES
    df.drop(columns=['communityname'], inplace=True)
    df = df.apply(pd.to_numeric, args=('coerce',))
    drop_high_nan(df, prop=0.1)
    h_to_drop = drop_low_var(df)
    df.drop(columns=h_to_drop, inplace=True)
    r"""
    \begin{tabular}{lrrrrrrrr}
    \hline
                        &   count &       mean &       std &   min &   25\% &   50\% &   75\% &   max \\
    \hline
    state               &    1993 & 28.6939    & 16.3951   &     1 & 12    & 34    & 42    &    56 \\
    fold                &    1993 &  5.49624   &  2.87265  &     1 &  3    &  5    &  8    &    10 \\
    racepctblack        &    1993 &  0.179709  &  0.25348  &     0 &  0.02 &  0.06 &  0.23 &     1 \\
    racePctWhite        &    1993 &  0.753643  &  0.244079 &     0 &  0.63 &  0.85 &  0.94 &     1 \\
    racePctHisp         &    1993 &  0.144009  &  0.232549 &     0 &  0.01 &  0.04 &  0.16 &     1 \\
    pctUrban            &    1993 &  0.696116  &  0.44487  &     0 &  0    &  1    &  1    &     1 \\
    PctIlleg            &    1993 &  0.25005   &  0.229991 &     0 &  0.09 &  0.17 &  0.32 &     1 \\
    PctRecentImmig      &    1993 &  0.18142   &  0.235837 &     0 &  0.03 &  0.09 &  0.23 &     1 \\
    PctRecImmig5        &    1993 &  0.182183  &  0.236379 &     0 &  0.03 &  0.08 &  0.23 &     1 \\
    PctRecImmig8        &    1993 &  0.184827  &  0.236787 &     0 &  0.03 &  0.09 &  0.23 &     1 \\
    PctRecImmig10       &    1993 &  0.18293   &  0.23487  &     0 &  0.03 &  0.09 &  0.23 &     1 \\
    MedNumBR            &    1993 &  0.314601  &  0.255212 &     0 &  0    &  0.5  &  0.5  &     1 \\
    MedYrHousBuilt      &    1993 &  0.494099  &  0.232499 &     0 &  0.35 &  0.52 &  0.67 &     1 \\
    PctHousNoPhone      &    1993 &  0.264541  &  0.242892 &     0 &  0.06 &  0.19 &  0.42 &     1 \\
    OwnOccMedVal        &    1993 &  0.263527  &  0.231594 &     0 &  0.09 &  0.17 &  0.39 &     1 \\
    OwnOccHiQuart       &    1993 &  0.268986  &  0.235303 &     0 &  0.09 &  0.18 &  0.38 &     1 \\
    RentHighQ           &    1993 &  0.422985  &  0.248346 &     0 &  0.22 &  0.37 &  0.59 &     1 \\
    PctForeignBorn      &    1993 &  0.2156    &  0.231182 &     0 &  0.06 &  0.13 &  0.28 &     1 \\
    LemasPctOfficDrugUn &    1993 &  0.0939388 &  0.240335 &     0 &  0    &  0    &  0    &     1 \\
    ViolentCrimesPerPop &    1993 &  0.237998  &  0.233042 &     0 &  0.07 &  0.15 &  0.33 &     1 \\
    \hline
    \end{tabular}
    """
    # tabulate_describtion(df)
    # for c in chunk(df.columns, 4):
    #     boxplots(df, c, log_y=False)
    df = (df-df.min())/(df.max()-df.min())
    # covar_matrix(df, feats=df.columns)
    y = df["ViolentCrimesPerPop"]
    X = df.drop(columns=["ViolentCrimesPerPop"])
    return X.to_numpy().astype(float), y.to_numpy().astype(float)


def dim_red():
    X, y = load_data()
    graph_scree(X)
    for tech in ["PCA", "ISO", "TSNE"]:
        graph_reduced_dimensions(
            X,
            y,
            reg=True, method=tech
        )

    return


def gpr_model():
    X, y = load_data()
    model = PCA(n_components=15)
    X_fitted = model.fit_transform(X)
    n, d = X_fitted.shape
    X_train_np, X_test_np, y_train_np, y_test_np, ind_train, ind_test = train_test_split(
        X_fitted, y, np.arange(n), test_size=0.2)
    # CV_onevar_sklearn(GaussianProcessRegressor, X_fitted, y, "alpha",
    #                   np.logspace(-4, 1.5, 10, base=10, dtype=float)
    #                   )
    model_params = {
        "alpha": 2.0
    }
    plot_final_errors_sklearn(X_fitted, y, GaussianProcessRegressor,
                              model_kwargs=model_params, reg=True)
    # reg = GaussianProcessRegressor(**model_params)
    # reg.fit(X_train_np, y_train_np)
    # y_train_pred = reg.predict(X_train_np)
    # y_test_pred = reg.predict(X_test_np)
    # model = PCA(n_components=2)
    # X_fitted = model.fit_transform(X)
    # X_train_np = model.fit_transform(X_train_np)
    # X_test_np = model.fit_transform(X_test_np)
    # true_vs_pred(X_fitted, y, X_train_np, y_train_pred,
    #              X_test_np, y_test_pred, reg=True)
    return


def rf_model():
    X, y = load_data()
    X_fitted = X
    n, d = X_fitted.shape
    print(f"{n=},{d=}")
    X_train_np, X_test_np, y_train_np, y_test_np, ind_train, ind_test = train_test_split(
        X_fitted, y, np.arange(n), test_size=0.2)
    CV_heatmap_sklearn(RandomForestRegressor, X_fitted, y, "max_depth", "max_features",
                       np.linspace(1, d, 10, dtype=int),
                       np.linspace(1, d, 10, dtype=int),
                       )
    model_params = {
        "max_depth": 19,
        "max_features": 19,
    }
    r"""
    \begin{tabular}{rrr}
    \hline
    Repetition &   Train MSE &   Test MSE \\
    \hline
                1 &  0.00278352 &  0.020284  \\
                2 &  0.00279283 &  0.0204085 \\
                3 &  0.00283465 &  0.0202092 \\
                4 &  0.00278479 &  0.0204838 \\
                5 &  0.00284724 &  0.0203834 \\
                6 &  0.00283097 &  0.02055   \\
                7 &  0.0027767  &  0.0203615 \\
                8 &  0.00287339 &  0.0202939 \\
                9 &  0.00281724 &  0.0202996 \\
            10 &  0.00280879 &  0.0202641 \\
    \hline
    \end{tabular}
    """
    plot_final_errors_sklearn(X_fitted, y, RandomForestRegressor,
                              model_kwargs=model_params, reg=True)
    # reg = RandomForestRegressor(**model_params)
    # reg.fit(X_train_np, y_train_np)
    # y_train_pred = reg.predict(X_train_np)
    # y_test_pred = reg.predict(X_test_np)
    # model = PCA(n_components=2)
    # X_fitted = model.fit_transform(X)
    # X_train_np = model.fit_transform(X_train_np)
    # X_test_np = model.fit_transform(X_test_np)
    # true_vs_pred(X_fitted, y, X_train_np, y_train_pred,
    #              X_test_np, y_test_pred, reg=True)


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
        tf.keras.layers.Dense(7, activation='relu'),
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
    #         os.getcwd(), "docs", "crimes", "crimes_model.png"),
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

    # plot_final_errors(X, y, layers, comp_args,
    #                   num_epochs=num_epochs, reg=True)
    tsne_model = PCA(n_components=2)
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
    # rf_model()
    nn_model()


if __name__ == "__main__":
    main()
