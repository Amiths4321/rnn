import numpy as np

latent_dimension = 32

# Generate 5 random latent vectors
random_latent = np.random.normal(
    size=(5, latent_dimension)
)

print("Latent vector shape:")
print(random_latent.shape)

print("\nFirst latent vector:")
print(random_latent[0])