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
DEFAULT_SPAM_PATH = os.path.join(
    os.getcwd(), "data", "spambase.data")
DEFAULT_PENGUINS_PATH = os.path.join(
    os.getcwd(), "data", "penguins.csv")
DEFAULT_BIKE_SHARING_PATH = os.path.join(
    os.getcwd(), "data", "bike_sharing.csv")


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
    mat = scipy.io.loadmat(path)
    data = mat['data'].astype(float)

    if not labels:
        return data

    labels = mat['labels'].astype(int).squeeze()

    return data, labels


def load_penguin(path: str = DEFAULT_PENGUINS_PATH, labels: bool = False):
    """
    CLASSIFICATION

    Loads a STANDARDIZED VERSION of the spam data set.

    Source: https://github.com/mwaskom/seaborn-data/blob/master/penguins.csv
    Number of Instances: 344
    Number of Attributes: 7
    Attribute Information (Raw):
        species: penguin species (Chinstrap, Adélie, or Gentoo)
        culmen_length_mm: culmen length (mm)
        culmen_depth_mm: culmen depth (mm)
        flipper_length_mm: flipper length (mm)
        body_mass_g: body mass (g)
        island: island name (Dream, Torgersen, or Biscoe) in the Palmer Archipelago (Antarctica)
        sex: penguin sex
    """
    df = pd.read_csv(path)
    labels_vec = df["species"].to_numpy(dtype=str)
    df.drop(columns=["species"], inplace=True)
    data = df.to_numpy()

    if labels:
        return data, labels_vec.squeeze()

    return data


def load_bike_sharing(path: str = DEFAULT_BIKE_SHARING_PATH, labels: bool = False):
    """
    REGRESSION

    Source: https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset
    Number of Instances: 17389
    Number of Attributes: 16
    Attribute Information (Raw):
        - instant: record index
        - dteday : date
        - season : season (1:winter, 2:spring, 3:summer, 4:fall)
        - yr : year (0: 2011, 1:2012)
        - mnth : month ( 1 to 12)
        - hr : hour (0 to 23)
        - holiday : weather day is holiday or not (extracted from [Web Link])
        - weekday : day of the week
        - workingday : if day is neither weekend nor holiday is 1, otherwise is 0.
        + weathersit :
        - 1: Clear, Few clouds, Partly cloudy, Partly cloudy
        - 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
        - 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
        - 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
        - temp : Normalized temperature in Celsius. The values are derived via (t-t_min)/(t_max-t_min), t_min=-8, t_max=+39 (only in hourly scale)
        - atemp: Normalized feeling temperature in Celsius. The values are derived via (t-t_min)/(t_max-t_min), t_min=-16, t_max=+50 (only in hourly scale)
        - hum: Normalized humidity. The values are divided to 100 (max)
        - windspeed: Normalized wind speed. The values are divided to 67 (max)
        - casual: count of casual users
        - registered: count of registered users
        - cnt: count of total rental bikes including both casual and registered 
    """
    df = pd.read_csv(path)
    labels_vec = df["cnt"].to_numpy(dtype=int)
    df.drop(columns=["cnt"], inplace=True)
    data = df.to_numpy()

    if labels:
        return data, labels_vec.squeeze()

    return data


def load_spam(path: str = DEFAULT_SPAM_PATH, labels: bool = False):
    """
    CLASSIFICATION

    Loads a STANDARDIZED VERSION of the spam data set.

    Source: https://archive.ics.uci.edu/ml/datasets/spambase
    Number of Instances: 4601
    Number of Attributes: 57
    Attribute Information (Raw):
        - 48 continuous real [0,100] attributes of type word_freq_WORD
        = percentage of words in the e-mail that match WORD, i.e. 100 * (number of times the WORD appears in the e-mail) / total number of words in e-mail. A "word" in this case is any string of alphanumeric characters bounded by non-alphanumeric characters or end-of-string.
        - 6 continuous real [0,100] attributes of type char_freq_CHAR]
        = percentage of characters in the e-mail that match CHAR, i.e. 100 * (number of CHAR occurences) / total characters in e-mail
        - 1 continuous real [1,...] attribute of type capital_run_length_average
        = average length of uninterrupted sequences of capital letters
        - 1 continuous integer [1,...] attribute of type capital_run_length_longest
        = length of longest uninterrupted sequence of capital letters
        - 1 continuous integer [1,...] attribute of type capital_run_length_total
        = sum of length of uninterrupted sequences of capital letters
        = total number of capital letters in the e-mail
        - 1 nominal {0,1} class attribute of type spam
        = denotes whether the e-mail was considered spam (1) or not (0), i.e. unsolicited commercial e-mail.

        - labels denote whether the e-mail was considered spam (1) or not (0)
    """
    from sklearn import preprocessing
    data = np.loadtxt(path, delimiter=",", skiprows=0)
    data_pro = data[:, 0:-1].astype(float)
    scaler = preprocessing.StandardScaler().fit(data_pro)
    data_pro = scaler.transform(data_pro)
    labels_pro = data[:, -1].astype(int).squeeze()

    if not labels:
        return data_pro

    return data_pro, labels_pro


def load_swiss_roll(labels: bool = False, n_samples=1500, noise=0.0):
    """
    REGRESSION

    Creates a swiss roll data set.

    Source: sklearn, see:
    https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_swiss_roll.html
    https://scikit-learn.org/stable/auto_examples/manifold/plot_swissroll.html#sphx-glr-auto-examples-manifold-plot-swissroll-py
    Number of Attributes: 3
    Attribute Information:
        X,Y and Z coordinates of each point
    """
    from sklearn.datasets import make_swiss_roll

    data, data_labels = make_swiss_roll(n_samples=n_samples, noise=noise)

    if not labels:
        return data

    return data, data_labels.squeeze()

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
