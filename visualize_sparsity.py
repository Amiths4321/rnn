import numpy as np

latent = np.array([
    0.00,
    0.02,
    0.85,
    0.00,
    0.00,
    0.71,
    0.01,
    0.00
])

print("Latent representation:")
print(latent)

print("\nNumber of active neurons:")

active = np.sum(latent > 0.5)

print(active)