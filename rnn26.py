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

encoder = keras.Sequential([
    keras.layers.Input(shape=[3, 1]),
    keras.layers.LSTM(16)
])


decoder = keras.Sequential([
    keras.layers.Input(shape=[3, 16]),
    keras.layers.LSTM(
        16,
        return_sequences=True
    ),
    keras.layers.Dense(1)
])

encoder_output = encoder(X)

decoder_input = keras.ops.repeat(
    encoder_output[:, None, :],
    3,
    axis=1
)

output = decoder(decoder_input)

print("Encoder output:", encoder_output.shape)
print("Decoder output:", output.shape)

