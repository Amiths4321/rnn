import numpy as np
from tensorflow import keras

# 1. Define n_steps and the series data
n_steps = 3
series = np.arange(1, 25, dtype=float)

current_sequence = series[:n_steps].copy()

# 2. Generate training data
X, y = [], []
for i in range(len(series) - n_steps):
    X.append(series[i:i + n_steps])
    y.append(series[i + n_steps])

X = np.array(X)[..., np.newaxis]
y = np.array(y).reshape(-1, 1)

# 3. Build and compile model FIRST
model = keras.Sequential([
    keras.layers.Input(shape=[n_steps, 1]),
    keras.layers.SimpleRNN(32, activation="tanh"),
    keras.layers.Dense(1)
])

model.compile(optimizer="adam", loss="mse")

# 4. Train model
model.fit(X, y, epochs=2, verbose=1)

print("Model trained successfully!")

# 5. Predict (model is now defined and ready)
test_input = current_sequence.reshape(1, n_steps, 1)
predictions = model.predict(test_input, verbose=0)

print("Predicted next value:", predictions[0][0])
print("Actual next value:", series[n_steps])