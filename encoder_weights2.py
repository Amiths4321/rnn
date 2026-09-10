import numpy as np

encoder_weights = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])

decoder_weights = encoder_weights.T

print("Encoder weights:")
print(encoder_weights)

print("\nDecoder weights:")
print(decoder_weights)