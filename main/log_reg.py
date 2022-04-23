#!/usr/bin/env python3

__author__ = 'Michael Ciccotosto-Camp'
__version__ = ''

from sklearn.linear_model import LogisticRegression

from data import load_data


def spam_example():
    """
    Example of using logistic regression using the spam data set.
    """
    # Train and test on the same dataset (yeah I know, probably not a great
    # idea in general but this is more just to demonstrate usage)
    data, labels = load_data("spam", labels=True)
    logr_model = LogisticRegression()
    logr_model.fit(data, labels)
    y_pred = logr_model.predict(data)
    acc = int(sum(list(labels == y_pred)))
    print(f"Correctly predicted {acc / len(y_pred)}")
    return


def main():
    spam_example()


if __name__ == "__main__":
    main()
