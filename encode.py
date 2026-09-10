from tensorflow import keras
import numpy as np

X = np.array([
    [1, 2, 3, 4],
    [2, 3, 4, 5],
    [3, 4, 5, 6],
    [4, 5, 6, 7],
    [5, 6, 7, 8]
], dtype=np.float32)

# Full autoencoder using Functional API for explicit input definition
inputs = keras.Input(shape=(4,))
encoded = keras.layers.Dense(2, activation="relu")(inputs)
decoded = keras.layers.Dense(4, activation="linear")(encoded)

model = keras.Model(inputs=inputs, outputs=decoded)

model.compile(
    optimizer="adam",
    loss="mse"
)

model.fit(
    X,
    X,
    epochs=200,
    verbose=0
)

# Extract encoder using the explicit input tensor and hidden layer output
encoder = keras.Model(
    inputs=model.input,
    outputs=model.get_layer(index=1).output
)

latent = encoder.predict(
    X,
    verbose=0
)

print("Original data:")
print(X)

print("\nLatent representation:")
print(latent)