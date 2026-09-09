import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

# 1. Define sequence and n_steps
n_steps = 3
series = np.arange(1, 25, dtype=float)
sequence = series.copy()

# 2. Generate training data
X, y = [], []
for i in range(len(sequence) - n_steps):
    X.append(sequence[i:i + n_steps])
    y.append(sequence[i + n_steps])

X = np.array(X)[..., np.newaxis]
y = np.array(y).reshape(-1, 1)

# 3. Build and train model
model = keras.Sequential([
    keras.layers.Input(shape=[n_steps, 1]),
    keras.layers.SimpleRNN(32, activation="tanh"),
    keras.layers.Dense(1)
])

model.compile(optimizer="adam", loss="mse")
model.fit(X, y, epochs=100, verbose=0)

print("Model trained successfully!")

# 4. Iteratively generate future values
generated_steps = 10
current_input = sequence[-n_steps:].copy()

for _ in range(generated_steps):
    test_input = current_input.reshape(1, n_steps, 1)
    pred = model.predict(test_input, verbose=0)[0][0]
    sequence = np.append(sequence, pred)
    current_input = np.append(current_input[1:], pred)

# 5. Plot results
plt.figure(figsize=(12, 5))

plt.plot(
    sequence,
    label="Original + Generated",
    marker="o"
)

plt.axvline(
    len(series) - 1,
    color="red",
    linestyle="--",
    label="Generation starts"
)

plt.title("Creative RNN")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.show()