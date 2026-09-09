import numpy as np
from tensorflow import keras


# ============================================================
# TRAINING DATA
# ============================================================

X = np.array([
    [[1], [2], [1]],
    [[2], [1], [2]],
    [[1], [1], [2]],

    [[8], [9], [8]],
    [[9], [8], [9]],
    [[8], [8], [9]]
], dtype=float)


# 0 = Low pattern
# 1 = High pattern

y = np.array([
    0,
    0,
    0,
    1,
    1,
    1
])


print("Input shape:", X.shape)
print("Output shape:", y.shape)


# ============================================================
# BUILD RNN CLASSIFIER
# ============================================================

model = keras.Sequential([

    keras.layers.Input(shape=[3, 1]),

    keras.layers.SimpleRNN(
        16,
        activation="tanh"
    ),

    keras.layers.Dense(
        1,
        activation="sigmoid"
    )
])


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# TRAIN
# ============================================================

model.fit(
    X,
    y,
    epochs=100,
    verbose=0
)


# ============================================================
# TEST
# ============================================================

test_sequence = np.array([
    [[9], [8], [9]]
], dtype=float)


prediction = model.predict(
    test_sequence,
    verbose=0
)[0][0]


print("\nPrediction probability:", prediction)


if prediction >= 0.5:
    print("Prediction: HIGH pattern")
else:
    print("Prediction: LOW pattern")