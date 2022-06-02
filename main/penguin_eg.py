#!/usr/bin/env python3

__author__ = 'Michael Ciccotosto-Camp'
__version__ = ''

import numpy as np
import pandas as pd

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from data import load_data
from exam_utils import *


def penguin_rfc_example():
    X, y = load_data("penguin", labels=True)
    # Remove the island, first column
    X = X[:, 1:]
    y = y[~pd.isna(X).any(axis=1)]
    X = X[~pd.isna(X).any(axis=1), :]
    X[X[:, -1] == "MALE", -1] = 1
    X[X[:, -1] == "FEMALE", -1] = 0
    X.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2)
    rfc_model = RandomForestClassifier()
    rfc_model.fit(X_train, y_train)

    print('Test acc:',
          accuracy_score(y_test, rfc_model.predict(X_test)))
    print('Train acc:',
          accuracy_score(y_train, rfc_model.predict(X_train)))
    return


def load_penguin_data():
    import os
    DEFAULT_PENGUINS_PATH = os.path.join(
        os.getcwd(), "data", "penguins.csv")
    df = pd.read_csv(DEFAULT_PENGUINS_PATH)
    # Drop rows with Nan values
    df = df.dropna(axis=0)
    # species,island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex
    sp_le = LabelEncoder()
    df["species"] = sp_le.fit_transform(df["species"])
    isl_le = LabelEncoder()
    df["island"] = isl_le.fit_transform(df["island"])
    sex_le = LabelEncoder()
    df["sex"] = sex_le.fit_transform(df["sex"])
    # graph_scree(
    #     df["island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex".split(",")].to_numpy())
    df.drop(columns=["island"], inplace=True)
    # covar_matrix(
    #     df, "bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex".split(","))
    df["species"] = sp_le.inverse_transform(df["species"])
    # boxplots(df, "species", "bill_length_mm,bill_depth_mm".split(","),
    #          )
    # boxplots(df, "species", "flipper_length_mm,body_mass_g".split(","),
    #          )
    df["species"] = sp_le.transform(df["species"])
    df["bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g".split(",")] = \
        StandardScaler().fit_transform(
            df["bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g".split(",")].to_numpy())
    # graph_scree(
    #     df["bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex".split(",")].to_numpy())
    # graph_reduced_dimensions(
    #     df["bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex".split(
    #         ",")],
    #     df["species"].to_numpy(),
    #     encoder=sp_le, reg=False, method="LDA"
    # )
    return df, sp_le, isl_le, sex_le


def penguin_svm_example():
    df, sp_le, isl_le, sex_le = load_penguin_data()

    X_ss = df["bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex".split(
        ",")].to_numpy()
    y = df["species"].to_numpy()

    print("Started LDA")
    X_lda = LDA(n_components=2).fit_transform(X_ss, y)
    X_train, X_test, y_train, y_test = train_test_split(
        X_lda, y, test_size=0.2)

    print("Started SVC")
    svc_model = SVC(kernel="rbf", gamma=1.0)
    svc_model.fit(X_train, y_train)
    print('Test acc:',
          accuracy_score(y_test, svc_model.predict(X_test)))
    print('Train acc:',
          accuracy_score(y_train, svc_model.predict(X_train)))
    # CV_heatmap_sklearn(SVC, X_lda, y, "C", "gamma",
    #                    np.logspace(-1, 1.5, 7, base=10),
    #                    np.logspace(-4, 1.5, 10, base=10)
    #                    )
    CV_onevar_sklearn(SVC, X_lda, y, "gamma",
                      np.logspace(-4, 1.5, 10, base=10)
                      )
    # X = X_lda
    # y_pred_train = svc_model.predict(X_train)
    # y_pred_test = svc_model.predict(X_test)
    # y_pred = svc_model.predict(X)
    # true_vs_pred(X, y, X_train, y_pred_train,
    #              X_test, y_pred_test, reg=False, classes=sp_le.classes_)

    # graph_confusion_matrix(y, y_pred, sp_le.classes_)
    return


def main():
    penguin_svm_example()
    # penguin_rfc_example()
    # load_penguin_data()


if __name__ == "__main__":
    main()
