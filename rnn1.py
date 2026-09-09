import numpy as np
from tensorflow import keras


# Training sequences
X = np.array([
    [1, 2, 3],
    [2, 3, 4],
    [3, 4, 5],
    [4, 5, 6],
    [5, 6, 7],
    [6, 7, 8]
], dtype=float)


# Expected next number
y = np.array([
    4,
    5,
    6,
    7,
    8,
    9
], dtype=float)


# RNN model
model = keras.Sequential([

    keras.layers.Input(shape=[3, 1]),

    keras.layers.SimpleRNN(
        16,
        activation="tanh"
    ),

    keras.layers.Dense(1)
])


# Compile
model.compile(
    optimizer="adam",
    loss="mse"
)


# Train
model.fit(
    X.reshape(-1, 3, 1),
    y,
    epochs=200,
    verbose=0
)


# Prediction
test_sequence = np.array([
    [7],
    [8],
    [9]
], dtype=float)


prediction = model.predict(
    test_sequence.reshape(1, 3, 1),
    verbose=0
)


print("Predicted next number:", prediction[0][0])

import numpy as np

X = np.array([
    [1, 2, 3],
    [2, 3, 4],
    [3, 4, 5],
    [4, 5, 6],
    [5, 6, 7],
    [6, 7, 8]
], dtype=float)

X = X.reshape(-1, 3, 1)

print("Shape:", X.shape)