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
from exam_utils import true_vs_pred, graph_confusion_matrix, boxplots


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


def penguin_svm_example():
    import os
    DEFAULT_PENGUINS_PATH = os.path.join(
        os.getcwd(), "data", "penguins.csv")
    df = pd.read_csv(DEFAULT_PENGUINS_PATH)
    df = df.dropna(axis=0)
    boxplots(df, "species", ["bill_length_mm",
             "bill_depth_mm", "flipper_length_mm", "body_mass_g"])
    print(df)
    return
    X, y = load_data("penguin", labels=True)
    # Remove the island, first column
    X = X[:, 1:]
    y = y[~pd.isna(X).any(axis=1)]
    X = X[~pd.isna(X).any(axis=1), :]
    X[X[:, -1] == "MALE", -1] = 1
    X[X[:, -1] == "FEMALE", -1] = 0
    X.astype(float)

    print("Started StandardScaler")
    X_ss = StandardScaler().fit_transform(X)
    X_ss[X[:, -1] == 1, -1] = 1
    X_ss[X[:, -1] == 0, -1] = 0

    print("Started LabelEncoder")
    le = LabelEncoder()
    y = le.fit_transform(y)

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

    X = X_lda
    y_pred_train = svc_model.predict(X_train)
    y_pred_test = svc_model.predict(X_test)
    # true_vs_pred(X, y, X_train, y_pred_train,
    #              X_test, y_pred_test, reg=False, classes=le.classes_)

    graph_confusion_matrix(y_test, y_pred_test, le.classes_)
    return


def main():
    penguin_svm_example()
    # penguin_rfc_example()


if __name__ == "__main__":
    main()
