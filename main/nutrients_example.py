import os
import pandas as pd
import numpy as np
from pprint import pprint
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import torch
import tensorflow as tf


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
    rename = {}
    for col in pfk.columns:
        if col == 0:
            continue
        pfk[col] = preprocessing.LabelEncoder().fit_transform(pfk[col])
        rename[col] = f"{col_name} {col}"
    pfk.rename(columns=rename, inplace=True)
    df.drop(columns=[col_name], inplace=True)
    df = pd.concat([df, pfk], axis=1).reindex(df.index)
    df.drop(columns=[0], inplace=True)
    return df


NUTRITION_PATH = os.path.join(
    os.getcwd(), "data", "FoodNutrients1.csv")
df = pd.read_csv(NUTRITION_PATH)
# print(df.iloc[0])
drop_high_nan(df)
# Drop any remaining rows that have nan values
df.dropna(axis='rows', inplace=True)
# Drop the food name
df.drop(columns="Food Name", inplace=True)
df = expand_col(df, "Public Food Key")
df["Classification"] = df["Classification"].astype(int).astype(str)
df = expand_col(df, "Classification")
# predict energy
y = df["Energy, with dietary fibre"].to_numpy().astype(float)
df.drop(columns=["Energy, with dietary fibre",
        "Energy, without dietary fibre"], inplace=True)
x = df.to_numpy().astype(float)
print(x.shape)

# print(np.max(x), np.mean(y), np.min(y))
# exit()
X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
    x, y, test_size=0.2)
n, d = X_train_np.shape

X_train = torch.tensor(X_train_np)
X_test = torch.tensor(X_test_np)
y_train = torch.tensor(y_train_np)
y_test = torch.tensor(y_test_np)

train_ds = tf.data.Dataset.from_tensor_slices(
    (X_train, y_train)).shuffle(10000).batch(250)

test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(250)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(d, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=[
              tf.keras.metrics.MeanAbsoluteError()])
model.fit(train_ds, epochs=1000, validation_data=test_ds)

# print(np.round(model.predict(X_test_np)))
# print(y_test)
