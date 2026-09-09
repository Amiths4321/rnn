import numpy as np
from tensorflow import keras


# ============================================================
# DATA
# ============================================================

X = []
y = []

for i in range(1, 18):

    X.append([
        [i],
        [i + 1],
        [i + 2]
    ])

    y.append(i + 3)


X = np.array(X, dtype=float)
y = np.array(y, dtype=float)


print("X shape:", X.shape)
print("y shape:", y.shape)


# ============================================================
# DEEP RNN + DROPOUT
# ============================================================

model = keras.Sequential([

    keras.layers.Input(shape=[3, 1]),

    keras.layers.SimpleRNN(
        32,
        return_sequences=True
    ),

    keras.layers.Dropout(0.3),

    keras.layers.SimpleRNN(
        32
    ),

    keras.layers.Dropout(0.3),

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
    epochs=100,
    verbose=0
)


# ============================================================
# PREDICTION
# ============================================================

test_sequence = np.array([
    [[18],
     [19],
     [20]]
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

print("Expected: approximately 21")