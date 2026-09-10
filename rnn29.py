import numpy as np

# Encoder outputs
encoder_outputs = np.array([
    [1.0, 0.0],   # I
    [0.0, 1.0],   # drink
    [1.0, 1.0]    # milk
])

# Attention weights
attention_weights = np.array([
    0.1,
    0.8,
    0.1
])

# Calculate weighted context vector
context_vector = np.sum(
    encoder_outputs * attention_weights[:, None],
    axis=0
)

print("Context vector:")
print(context_vector)