#!/usr/bin/env python3

__author__ = 'Michael Ciccotosto-Camp'
__version__ = ''

import numpy as np

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.decomposition import TruncatedSVD


def np_lr_example():
    data_train = fetch_20newsgroups(subset='train')
    x_train, y_train = data_train.data, data_train.target
    data_test = fetch_20newsgroups(subset='test')
    x_test, y_test = data_test.data, data_test.target

    print("Fitting TfidfVectorizer")
    vectorizer = TfidfVectorizer(
        stop_words='english', encoding='utf-8', decode_error='ignore',
        lowercase=True)
    x_train_vec = vectorizer.fit_transform(x_train)
    # x_train_vec = x_train_vec.todense()
    x_test_vec = vectorizer.transform(x_test)
    # x_test_vec = x_test_vec.todense()

    # print("Fitting PCA")
    # pca_model = TruncatedSVD(3000)
    # x_train_vec = pca_model.fit_transform(x_train_vec)
    # x_test_vec = pca_model.transform(x_test_vec)

    print("Fitting LogisticRegression")
    lr = LogisticRegression(solver='lbfgs', multi_class='multinomial')
    lr.fit(x_train_vec, y_train)
    print('Test set accuracy of logistic regression:',
          accuracy_score(y_test, lr.predict(x_test_vec)))

    return


def main():
    np_lr_example()


if __name__ == "__main__":
    main()
