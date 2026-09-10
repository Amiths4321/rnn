import numpy as np

# Mean of latent distribution
mean = 0.0

# Standard deviation
std = 1.0

# Generate random latent values
latent_samples = np.random.normal(
    mean,
    std,
    10
)

print("Random latent samples:")
print(latent_samples)