import numpy as np
from tensorflow import keras


# ============================================================
# TRAINING DATA
# ============================================================

X = np.array([
    [[1], [2], [3]],
    [[2], [3], [4]],
    [[3], [4], [5]],
    [[4], [5], [6]],
    [[5], [6], [7]],
    [[6], [7], [8]]
], dtype=float)


y = np.array([
    4,
    5,
    6,
    7,
    8,
    9
], dtype=float)


# ============================================================
# LSTM MODEL
# ============================================================

model = keras.Sequential([

    keras.layers.Input(shape=[3, 1]),

    keras.layers.LSTM(
        16
    ),

    keras.layers.Dense(1)
])


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer="adam",
    loss="mse"
)


# ============================================================
# TRAIN
# ============================================================

history = model.fit(
    X,
    y,
    epochs=100,
    verbose=0
)


# ============================================================
# TEST
# ============================================================

test_sequence = np.array([
    [[7], [8], [9]]
], dtype=float)


prediction = model.predict(
    test_sequence,
    verbose=0
)


print("Input:", test_sequence.flatten())

print(
    "Predicted:",
    prediction[0][0]
)

print("Expected: approximately 10")