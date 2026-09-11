from tensorflow import keras

model = keras.Sequential([
    keras.layers.Input(shape=(784,)),

    keras.layers.Dense(128, activation="relu"),

    # Very small bottleneck
    keras.layers.Dense(2, activation="relu"),

    keras.layers.Dense(128, activation="relu"),

    keras.layers.Dense(784, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy"
)

model.summary()