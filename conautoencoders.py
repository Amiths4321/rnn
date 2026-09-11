from tensorflow import keras

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

X_train = X_train[..., None]
X_test = X_test[..., None]

model = keras.Sequential([
    keras.layers.Input(shape=(28, 28, 1)),

    # Encoder
    keras.layers.Conv2D(
        32,
        3,
        activation="relu",
        padding="same"
    ),

    keras.layers.MaxPooling2D(2),

    keras.layers.Conv2D(
        16,
        3,
        activation="relu",
        padding="same"
    ),

    keras.layers.MaxPooling2D(2),

    # Decoder
    keras.layers.UpSampling2D(2),

    keras.layers.Conv2D(
        32,
        3,
        activation="relu",
        padding="same"
    ),

    keras.layers.UpSampling2D(2),

    keras.layers.Conv2D(
        1,
        3,
        activation="sigmoid",
        padding="same"
    )
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy"
)

model.summary()