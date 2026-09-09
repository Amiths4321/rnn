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


# Output sequence
y = np.array([
    [[2], [3], [4]],
    [[3], [4], [5]],
    [[4], [5], [6]],
    [[5], [6], [7]],
    [[6], [7], [8]],
    [[7], [8], [9]]
], dtype=float)


print("Input shape:", X.shape)
print("Output shape:", y.shape)


# ============================================================
# BUILD RNN
# ============================================================

model = keras.Sequential([

    keras.layers.Input(shape=[3, 1]),

    keras.layers.SimpleRNN(
        16,
        activation="tanh",
        return_sequences=True
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

model.fit(
    X,
    y,
    epochs=200,
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


print("\nInput sequence:")
print(test_sequence.flatten())

print("\nPredicted sequence:")
print(prediction.flatten())