import numpy as np
from tensorflow import keras

# Define n_steps and series data first
n_steps = 3
series = np.arange(1, 25, dtype=float)

# Initialize X and y lists before the loop
X, y = [], []
for i in range(len(series) - n_steps):
    X.append(series[i:i + n_steps])
    y.append(series[i + n_steps])

X = np.array(X)[..., np.newaxis]
y = np.array(y).reshape(-1, 1)

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
test_input = test_sequence.reshape(1, n_steps, 1)

prediction = model.predict(test_input, verbose=0)

print("Predicted next value:", prediction[0][0])
print("Actual next value:", series[n_steps])