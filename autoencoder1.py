from tensorflow import keras

# Load MNIST
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Normalize pixels
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# Flatten 28x28 → 784
X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)

print("Training shape:", X_train.shape)
print("Test shape:", X_test.shape)


# -----------------------------------
# Autoencoder
# -----------------------------------

model = keras.Sequential([

    # Encoder
    keras.layers.Input(shape=(784,)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(32, activation="relu"),

    # Decoder
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(784, activation="sigmoid")
])


# Compile
model.compile(
    optimizer="adam",
    loss="binary_crossentropy"
)


# Train
model.fit(
    X_train,
    X_train,
    epochs=2,
    batch_size=128,
    validation_split=0.1
)
import matplotlib.pyplot as plt

reconstructed = model.predict(
    X_test,
    verbose=0
)

plt.imshow(
    reconstructed[0].reshape(28, 28),
    cmap="gray"
)

plt.title("Reconstructed Image")
plt.axis("off")
plt.show()

plt.imshow(
    X_test[0].reshape(28, 28),
    cmap="gray"
)

plt.title("Original Image")
plt.axis("off")
plt.show()

encoder = keras.Model(
    inputs=model.inputs,
    outputs=model.layers[1].output
)

latent = encoder.predict(
    X_test[:5],
    verbose=0
)

plt.scatter(
    latent[:, 0],
    latent[:, 1],
    s=5
)

plt.xlabel("Latent Dimension 1")
plt.ylabel("Latent Dimension 2")
plt.title("Autoencoder Latent Space")

plt.show()

print("Latent representation shape:")
print(latent.shape)

print("\nFirst image's latent representation:")
print(latent[0])


