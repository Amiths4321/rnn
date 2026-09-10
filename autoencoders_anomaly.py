import numpy as np

actual = np.array([
    500,
    700,
    600,
    550
])

reconstructed = np.array([
    510,
    690,
    605,
    545
])

error = np.mean(
    (actual - reconstructed) ** 2
)

print("Reconstruction error:", error)