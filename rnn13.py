import numpy as np
import matplotlib.pyplot as plt

# Create time values
t = np.linspace(0, 100, 1000)

# Create a repeating wave
series = np.sin(t)

# Plot
plt.plot(series)
plt.title("Time Series")
plt.xlabel("Time")
plt.ylabel("Value")
plt.show()