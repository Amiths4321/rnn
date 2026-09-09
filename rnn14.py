from tensorflow import keras
import numpy as np
# Define n_steps prior to building the model
n_steps = 3
series = np.arange(1, 25, dtype=float)

raw_data = np.arange(1, 25, dtype=float)
X, y = [], []
for i in range(len(raw_data) - n_steps):
    X.append(raw_data[i:i + n_steps])
    y.append(raw_data[i + n_steps])

X = np.array(X)[..., np.newaxis]  # Shape: (samples, n_steps, 1)
y = np.array(y)

model = keras.Sequential([

    keras.layers.Input(
        shape=[n_steps, 1]
    ),

    keras.layers.SimpleRNN(
        32,
        activation="tanh"
    ),

    keras.layers.Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse"
)

model.fit(
    X,
    y,
    epochs=2,
    verbose=1
)

print("Model trained successfully!")

test_sequence = series[:n_steps]

test_input = test_sequence.reshape(
    1,
    n_steps,
    1
)

prediction = model.predict(
    test_input,
    verbose=0
)

print("Predicted next value:", prediction[0][0])

print(
    "Actual next value:",
    series[n_steps]
)