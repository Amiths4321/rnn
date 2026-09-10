import numpy as np
from tensorflow import keras


# --------------------------------
# Input sequences
# --------------------------------

X = np.array([
    [[1], [2], [3]],
    [[2], [3], [4]],
    [[3], [4], [5]]
], dtype=float)


# --------------------------------
# Encoder
# --------------------------------

encoder = keras.Sequential([
    keras.layers.Input(shape=[3, 1]),
    keras.layers.LSTM(16)
])


context = encoder(X)

print("Input shape:")
print(X.shape)

print("\nEncoder output shape:")
print(context.shape)