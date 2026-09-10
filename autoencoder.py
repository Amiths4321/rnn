import numpy as np
from tensorflow import keras

# Training data
X = np.array([
    [1, 2, 3, 4],
    [2, 3, 4, 5],
    [3, 4, 5, 6],
    [4, 5, 6, 7],
    [5, 6, 7, 8]
], dtype=np.float32)


# Autoencoder
model = keras.Sequential([

    # Encoder
    keras.layers.Input(shape=(4,)),
    keras.layers.Dense(2, activation="relu"),

    # Decoder
    keras.layers.Dense(4, activation="linear")
])


# Compile
model.compile(
    optimizer="adam",
    loss="mse"
)


# Train
model.fit(
    X,
    X,
    epochs=200,
    verbose=0
)


# Reconstruct
reconstructed = model.predict(
    X,
    verbose=0
)

print("Original:")
print(X[0])

print("\nReconstructed:")
print(reconstructed[0])