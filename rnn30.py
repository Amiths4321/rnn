import numpy as np
from tensorflow import keras

# Encoder outputs
encoder_outputs = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]
], dtype=np.float32)

# Attention scores
scores = np.array([
    2.0,
    1.0,
    0.5
], dtype=np.float32)

# Convert scores to attention weights
attention_weights = keras.activations.softmax(
    scores
).numpy()

# Calculate context vector
context_vector = np.sum(
    encoder_outputs * attention_weights[:, None],
    axis=0
)

print("Attention weights:")
print(attention_weights)

print("\nContext vector:")
print(context_vector)