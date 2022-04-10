#!/usr/bin/env python3

__author__ = 'Michael Ciccotosto-Camp'
__version__ = ''

from sklearn.neighbors import KNeighborsClassifier

from data import load_data


def iris_example():
    """
    Examples of using knn using the iris data set.
    """
    # Train and test on the same dataset (yeah I know, probably not a great
    # idea in general but this is more just to demonstrate usage)
    X_train, y_train = load_data("iris", labels=True)
    X_test, y_test = X_train, y_train
    knn_model = KNeighborsClassifier(n_neighbors=5, weights="uniform", p=2)
    knn_model.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)
    acc = int(sum(list(y_test == y_pred)))
    print(f"Correctly predicted {acc}/{len(y_pred)}")
    return


def main():
    iris_example()


if __name__ == "__main__":
    main()
