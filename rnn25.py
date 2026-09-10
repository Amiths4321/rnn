import numpy as np
from tensorflow import keras


X = np.array([
    [[1], [2], [3]],
    [[2], [3], [4]],
    [[3], [4], [5]],
    [[4], [5], [6]],
    [[5], [6], [7]]
], dtype=float)


y = np.array([
    [[2], [3], [4]],
    [[3], [4], [5]],
    [[4], [5], [6]],
    [[5], [6], [7]],
    [[6], [7], [8]]
], dtype=float)


print("Input shape:", X.shape)
print("Output shape:", y.shape)