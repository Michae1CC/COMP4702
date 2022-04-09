#!/usr/bin/env python3

__author__ = 'Michael Ciccotosto-Camp'
__version__ = ''

import os
import sys
import inspect
import types
import numpy as np
import pandas as pd
import scipy.io

DEFAULT_ABALONE_PATH = os.path.join(
    os.getcwd(), "data", "abalone.csv")
DEFAULT_WINE_PATH = os.path.join(
    os.getcwd(), "data", "Wine_data.csv")
DEFAULT_CIFAR10_PATH = os.path.join(
    os.getcwd(), "data", "cifar10_data_batch_2.mat")


def is_loader(object):
    return isinstance(object, types.FunctionType) and (object.__module__ == __name__) and ("load_" in str(object))


def load_abalone(path: str = DEFAULT_ABALONE_PATH, labels: bool = False):
    """
    CLASSIFICATION (Intended)
    REGRESSION (Shell weight)

    Loads the abalone Data Set.

    Source: https://archive.ics.uci.edu/ml/datasets/abalone
    Number of Instances: 4177
    Number of Attributes: 8
    Attribute Information (Raw):
        1. Sex / nominal / -- / M, F, and I (infant)
        2. Length / continuous / mm / Longest shell measurement
        3. Diameter / continuous / mm / perpendicular to length
        4. Height / continuous / mm / with meat in shell
        5. Whole weight / continuous / grams / whole abalone
        6. Shucked weight / continuous / grams / weight of meat
        7. Viscera weight / continuous / grams / gut weight (after bleeding)
        8. Shell weight / continuous / grams / after being dried
        9. Rings / integer / -- / +1.5 gives the age in years 
    Description:
    Predicting the age of abalone from physical measurements. 
    The age of abalone is determined by cutting the shell through the cone, 
    staining it, and counting the number of rings through a microscope -- 
    a boring and time-consuming task. Other measurements, which are easier 
    to obtain, are used to predict the age.
    Additional Notes:
    The loaded data set will have the Sex removed to better suit the RBF 
    kernel. Shell weight will be removed to become part of the test labelling.
    """
    headers = "Sex,Length,Diameter,Height,Whole weight,Shucked weight,Viscera weight,Shell weight,Rings".split(
        sep=",")
    df = pd.read_csv(path, names=headers)
    shell_weight = df["Shell weight"]
    df.drop(columns=["Sex", "Shell weight"], inplace=True)
    data = df.to_numpy(dtype=float)

    if labels:
        return data, shell_weight.to_numpy(dtype=float).squeeze()

    return data


def load_wine(path: str = DEFAULT_WINE_PATH, labels: bool = False):
    """
    CLASSIFICATION

    Loads the slice localization Data Set.

    Source: https://archive.ics.uci.edu/ml/datasets/Wine+Quality
    Number of Instances: 4898
    Number of Attributes: 12
    Attribute Information (Raw):
        1 - fixed acidity (continuous)
        2 - volatile acidity (continuous)
        3 - citric acid (continuous)
        4 - residual sugar (continuous)
        5 - chlorides (continuous)
        6 - free sulfur dioxide (continuous)
        7 - total sulfur dioxide (continuous)
        8 - density (continuous)
        9 - pH (continuous)
        10 - sulphates (continuous)
        11 - alcohol (%) (continuous)
        Output variable (based on sensory data):
        12 - quality (integer score between 0 and 10)
    Description:
    Two datasets are included, related to red and white vinho verde wine 
    samples, from the north of Portugal. The goal is to model wine quality 
    based on physicochemical tests.
    Additional Notes:
    The labels are the "quality" values.
    """

    df = pd.read_csv(path)
    labels_vec = df["quality"].to_numpy(dtype=int)
    df.drop(columns=["quality"], inplace=True)
    data = df.to_numpy(dtype=float)

    if labels:
        return data, labels_vec.squeeze()

    return data


def load_iris(labels: bool = False):
    """
    CLASSIFICATION

    Loads the iris data set from sklearn.

    Source: sklearn
    Number of Instances: 150
    Number of Attributes: 4
    Attribute Information (Raw):
        1. Sepal Length
        2. Sepal Width
        3. Petal Length
        4. Petal Width
        5. Iris Type
    Description:
    This data sets consists of 3 different types of irises' (Setosa, Versicolour, and Virginica) petal and sepal 
    length, stored in a 150x4 numpy.ndarray.
    """
    from sklearn import datasets
    iris = datasets.load_iris()

    data = iris.data
    if not labels:
        return data

    labels = (iris.target).squeeze()

    return data, labels


def load_cifar10(path: str = DEFAULT_CIFAR10_PATH, labels: bool = False):
    """
    CLASSIFICATION

    Loads the cifar10 data set.

    Source: Course website
    Number of Instances: 10000
    Number of Attributes: 3072
    Attribute Information (Raw):
        Pixel values ranging 0-255.
    Description:
    The CIFAR-10 dataset (Canadian Institute For Advanced Research) is a 
    collection of images that are commonly used to train machine learning 
    and computer vision algorithms.
    """
    mat = scipy.io.loadmat('data/cifar10_data_batch_2.mat')
    data = mat['data'].astype(float)

    if not labels:
        return data

    labels = mat['labels'].astype(int).squeeze()

    return data, labels


def load_data(data: str, path: str = None, labels: bool = False, **kwargs):
    """
    Loads the specified data set.

    Current supports loading the following values for the "data" parameter:
        - "abalone" (Abalone dataset)
        - "wine" (Wine Tasting dataset)
        - "iris" (Iris dataset)

    Parameters:
        data:
            The data set to load.
        path:
            The path to the data set.
        labels:
            If true, returns the labels as a separate vector.
    """
    data = "load_" + data
    data_map = dict(inspect.getmembers(
        sys.modules[__name__], predicate=is_loader))

    if data not in data_map:
        raise ValueError("No loader for " + str(data) + " found.")

    data_loader = data_map[data]

    if path is None:
        return data_loader(labels=labels, **kwargs)

    return data_loader(path=path, labels=labels, **kwargs)


def main():

    # Example of using the load_data function with the iris dataset
    data, labels = load_data("iris", labels=True)
    print(data)
    print(labels)
    print(data.shape)
    print(labels.shape)


if __name__ == "__main__":
    main()
