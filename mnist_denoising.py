from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Normalize
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# Flatten
X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)

# Add random noise
noise_factor = 0.3

X_train_noisy = X_train + noise_factor * np.random.normal(
    loc=0.0,
    scale=1.0,
    size=X_train.shape
)

X_test_noisy = X_test + noise_factor * np.random.normal(
    loc=0.0,
    scale=1.0,
    size=X_test.shape
)

# Keep values between 0 and 1
X_train_noisy = np.clip(X_train_noisy, 0.0, 1.0)
X_test_noisy = np.clip(X_test_noisy, 0.0, 1.0)

# Autoencoder
model = keras.Sequential([
    keras.layers.Input(shape=(784,)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(784, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy"
)

# IMPORTANT:
# noisy image → clean image
model.fit(
    X_train_noisy,
    X_train,
    epochs=2,
    batch_size=128,
    validation_split=0.1
)

reconstructed = model.predict(
    X_test_noisy[:1],
    verbose=0
)

plt.figure(figsize=(6, 2))

plt.subplot(1, 2, 1)
plt.imshow(
    X_test_noisy[0].reshape(28, 28),
    cmap="gray"
)
plt.title("Noisy")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(
    reconstructed[0].reshape(28, 28),
    cmap="gray"
)
plt.title("Denoised")
plt.axis("off")

plt.show()

