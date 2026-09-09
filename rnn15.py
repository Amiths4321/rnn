import numpy as np
import matplotlib.pyplot as plt

# Create time series
t = np.linspace(0, 100, 1000)

series = np.sin(t)

# Plot
plt.plot(series)
plt.title("Time Series")
plt.show()

n_steps = 20
n_outputs = 5

X = []
y = []

for i in range(
    len(series) - n_steps - n_outputs
):

    X.append(
        series[i:i + n_steps]
    )

    y.append(
        series[
            i + n_steps:
            i + n_steps + n_outputs
        ]
    )

X = np.array(X)
y = np.array(y)

# Add feature dimension
X = X[..., np.newaxis]

print("X shape:", X.shape)
print("y shape:", y.shape)

