#!/usr/bin/env python3

__author__ = 'Michael Ciccotosto-Camp'
__version__ = ''

import numpy as np

from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.manifold import Isomap
from sklearn.model_selection import train_test_split

from data import load_data


def spam_example():
    # Reduce the number of dimensions from 3 to 2.
    X, y = load_data("spam", labels=True)
    iso_model = LDA(n_components=1)
    # iso_model = Isomap(n_components=4)
    X_fitted = iso_model.fit_transform(X, y)
    X_train, X_test, y_train, y_test = train_test_split(
        X_fitted, y, test_size=0.2)
    svc_model = SVC(kernel="rbf", gamma=1.0)
    svc_model.fit(X_train, y_train)
    y_pred_test = svc_model.predict(X_test)
    y_pred_train = svc_model.predict(X_train)
    acc_test = int(sum(list(y_test == y_pred_test)))
    acc_train = int(sum(list(y_train == y_pred_train)))
    print(f"Val {acc_test / len(y_pred_test)}")
    print(f"Train {acc_train / len(y_train)}")
    return


def main():
    spam_example()


if __name__ == "__main__":
    main()
