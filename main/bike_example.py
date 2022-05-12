import os
import pandas as pd
import numpy as np
import torch
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from data import load_data
from exam_utils import true_vs_pred

DEFAULT_BIKE_SHARING_PATH = os.path.join(
    os.getcwd(), "data", "bike_sharing.csv")
df = pd.read_csv(DEFAULT_BIKE_SHARING_PATH)
y = df["cnt"].to_numpy(dtype=int)
df.drop(columns=["cnt"], inplace=True)
# Drop the instance number
df.drop(columns=["instant"], inplace=True)
df.dropna(axis='rows', inplace=True)
date_format = r"%Y-%m-%d"  # Reads as: yyyymmdd
DATA_COLS = df.columns.to_list()
cols = df["dteday"]
cols = pd.to_datetime(cols, format=date_format)
start_date = cols.iloc[0]
end_date = cols.iloc[-1]
cols = cols.diff(1).dt.days
cols = cols.fillna(0.0)
cols = cols.astype(float)
cols = cols.cumsum(axis="index", skipna=True)
df["dteday"] = cols.to_numpy(dtype=int)
df[["temp", "atemp", "hum", "windspeed"]] = StandardScaler(
).fit_transform(df[["temp", "atemp", "hum", "windspeed"]].to_numpy())
x = df.to_numpy()
X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
    x, y, test_size=0.2)
n, d = X_train_np.shape

X_train = torch.tensor(X_train_np)
X_test = torch.tensor(X_test_np)
y_train = torch.tensor(y_train_np)
y_test = torch.tensor(y_test_np)

train_ds = tf.data.Dataset.from_tensor_slices(
    (X_train, y_train)).shuffle(10000).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(d, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=[
              tf.keras.metrics.MeanAbsoluteError()])
model.fit(train_ds, epochs=8, validation_data=test_ds)

cpm_model = PCA(n_components=2)
X = x
X_pca = cpm_model.fit_transform(x)
X_train_pca = cpm_model.fit_transform(X_train_np)
X_test_pca = cpm_model.fit_transform(X_test_np)
y_pred_train = np.round(model.predict(X_train_np))
y_pred_test = np.round(model.predict(X_test_np))

true_vs_pred(X_pca, y, X_train_pca, y_pred_train,
             X_test_pca, y_pred_test, reg=True)

# print(np.round(model.predict(X_test_np)))
# print(y_test)
