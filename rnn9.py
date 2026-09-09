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


# ============================================================
# SIMPLE RNN
# ============================================================

rnn_model = keras.Sequential([
    keras.layers.Input(shape=[3, 1]),
    keras.layers.SimpleRNN(16),
    keras.layers.Dense(1)
])

rnn_model.compile(
    optimizer="adam",
    loss="mse"
)

rnn_model.fit(
    X,
    y,
    epochs=100,
    verbose=0
)


# ============================================================
# LSTM
# ============================================================

lstm_model = keras.Sequential([
    keras.layers.Input(shape=[3, 1]),
    keras.layers.LSTM(16),
    keras.layers.Dense(1)
])

lstm_model.compile(
    optimizer="adam",
    loss="mse"
)

lstm_model.fit(
    X,
    y,
    epochs=100,
    verbose=0
)


# ============================================================
# GRU
# ============================================================

gru_model = keras.Sequential([
    keras.layers.Input(shape=[3, 1]),
    keras.layers.GRU(16),
    keras.layers.Dense(1)
])

gru_model.compile(
    optimizer="adam",
    loss="mse"
)

gru_model.fit(
    X,
    y,
    epochs=100,
    verbose=0
)


# ============================================================
# TEST
# ============================================================

test_sequence = np.array([
    [[18],
     [19],
     [20]]
], dtype=float)


rnn_prediction = rnn_model.predict(
    test_sequence,
    verbose=0
)[0][0]

lstm_prediction = lstm_model.predict(
    test_sequence,
    verbose=0
)[0][0]

gru_prediction = gru_model.predict(
    test_sequence,
    verbose=0
)[0][0]


# ============================================================
# RESULTS
# ============================================================

print("Expected:", 21)

print("SimpleRNN:", rnn_prediction)

print("LSTM:", lstm_prediction)

print("GRU:", gru_prediction)