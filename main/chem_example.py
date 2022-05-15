from matplotlib.pyplot import xcorr
import pandas as pd
import torch
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings

from exam_utils import *

warnings.filterwarnings("ignore")
# soldata = pd.read_csv('https://dataverse.harvard.edu/api/access/datafile/3407241?format=original&gbrecs=true')
# had to rehost because dataverse isn't reliable
soldata = pd.read_csv(
    "https://github.com/whitead/dmol-book/raw/master/data/curated-solubility-dataset.csv"
)

SMILES = soldata["SMILES"]
# make tokenizer with 128 size vocab and
# have it examine all text in dataset
vocab_size = 128
tokenizer = tf.keras.preprocessing.text.Tokenizer(
    vocab_size, filters="", char_level=True
)
tokenizer.fit_on_texts(SMILES)
# now get padded sequences
seqs = (tokenizer.texts_to_sequences(SMILES))
padded_seqs = np.array(tf.keras.preprocessing.sequence.pad_sequences(
    seqs, padding="post"))
phy_prop = soldata[["MolWt", "BertzCT",
                    "NumRotatableBonds", "NumValenceElectrons", "NumSaturatedRings", "NumAliphaticRings", "RingCount", "TPSA", "LabuteASA", "BalabanJ"]].to_numpy()
y = soldata["Solubility"].to_numpy()
x = np.hstack((padded_seqs, phy_prop))
x = StandardScaler().fit_transform(x)
print(np.min(y), np.max(y))
# x = PCA(n_components=50).fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2)
n, d = X_train.shape

X_train = torch.tensor(X_train)
X_test = torch.tensor(X_test)
y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)

train_ds = tf.data.Dataset.from_tensor_slices(
    (X_train, y_train)).shuffle(10000).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(d, activation='sigmoid'),
    tf.keras.layers.Dense(50, activation='sigmoid'),
    tf.keras.layers.Dense(1)
])

num_epochs = 30
model.compile(optimizer='adam', loss='mse', metrics=[
              tf.keras.metrics.MeanAbsoluteError()])
history = model.fit(train_ds, epochs=30, validation_data=test_ds)

data = [
    (np.arange(1, num_epochs + 1, 1, dtype=int),
     history.history['mean_absolute_error'], "Training"),
    (np.arange(1, num_epochs + 1, 1, dtype=int),
     history.history['val_mean_absolute_error'], "Validation")
]
plot_meteric(data, "Model Absolute Error", "Epoch", "Mean Absolute Error")
