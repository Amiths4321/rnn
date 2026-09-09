import numpy as np
from tensorflow import keras

# Create time series
t = np.linspace(0, 100, 1000)

series = np.sin(t)

n_steps = 20

X = []
y = []

for i in range(len(series) - n_steps):

    X.append(
        series[i:i + n_steps]
    )

    y.append(
        series[i + n_steps]
    )

X = np.array(X)
y = np.array(y)

X = X[..., np.newaxis]

print("X shape:", X.shape)
print("y shape:", y.shape)

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
    epochs=20,
    verbose=0
)

print("Training completed.")

sequence = list(series[:n_steps])

for _ in range(100):

    input_data = np.array(
        sequence[-n_steps:]
    ).reshape(
        1,
        n_steps,
        1
    )

    prediction = model.predict(
        input_data,
        verbose=0
    )

    next_value = prediction[0, 0]

    sequence.append(next_value)

print("Generated values:")
print(sequence[-20:])

