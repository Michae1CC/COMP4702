import os
import pandas as pd
import numpy as np
from pprint import pprint
# import torch
# import tensorflow as tf


def drop_high_nan(df, prop=0.2):

    n, _ = df.shape
    to_drop = []

    for col in df.columns:
        num_nans = (df[col]).isna().sum()
        if num_nans / n > prop:
            to_drop.append(col)

    df.drop(columns=to_drop, inplace=True)
    return


def expand_col(df, col_name):
    # df[col_name] = df[col_name].str.rstrip()
    pfk = df[col_name].str.split(pat="\s*", expand=True)
    n, d = pfk.shape
    to_drop = []
    for i in range(1, d):
        unique_chars = len(np.unique(pfk[i]))
        if unique_chars < 2:
            to_drop.append(i)
    pfk.drop(columns=to_drop, inplace=True)
    print(pfk)


NUTRITION_PATH = os.path.join(
    os.getcwd(), "data", "FoodNutrients1.csv")
df = pd.read_csv(NUTRITION_PATH)
print(df.shape)
# print(df.iloc[0])
drop_high_nan(df)
# Drop any remaining rows that have nan values
df.dropna(axis='rows', inplace=True)
print(df.shape)
# print(df)
pprint(list(df.columns))
# Drop the food name
df.drop(columns="Food Name", inplace=True)
expand_col(df, "Public Food Key")
