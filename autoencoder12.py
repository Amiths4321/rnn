from tensorflow import keras

import numpy as np
from tensorflow import keras

# Define or load your training data
X_train = np.array([
    [1, 2, 3, 4],
    [2, 3, 4, 5],
    [3, 4, 5, 6],
    [4, 5, 6, 7],
    [5, 6, 7, 8]
], dtype=np.float32)

# Build and compile your model
inputs = keras.Input(shape=(4,))
encoded = keras.layers.Dense(2, activation="relu")(inputs)
decoded = keras.layers.Dense(4, activation="linear")(encoded)

model = keras.Model(inputs=inputs, outputs=decoded)
model.compile(optimizer="adam", loss="mse")

# Now X_train is defined and can be used here
model.fit(X_train, X_train, epochs=200, verbose=0)

autoencoder = keras.Sequential([
    keras.layers.Input(shape=(4,)),  # Change this from 784 to 4
    keras.layers.Dense(2, activation="relu"),
    keras.layers.Dense(4, activation="linear")
])

autoencoder.compile(
    optimizer="adam",
    loss="binary_crossentropy"
)

autoencoder.fit(
    X_train,
    X_train,
    epochs=5,
    batch_size=128,
    verbose=1
)