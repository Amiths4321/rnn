from tensorflow import keras
import numpy as np

X_train = np.array([
    [1, 2, 3, 4],
    [2, 3, 4, 5],
    [3, 4, 5, 6],
    [4, 5, 6, 7],
    [5, 6, 7, 8]
], dtype=np.float32)

autoencoder1 = keras.Sequential([
    keras.layers.Input(shape=(4,)),
    keras.layers.Dense(2, activation="relu"),
    keras.layers.Dense(4, activation="linear")
])

autoencoder1.compile(
    optimizer="adam",
    loss="binary_crossentropy"
)

autoencoder1.fit(
    X_train,
    X_train,
    epochs=5,
    batch_size=128
)