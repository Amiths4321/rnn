import numpy as np
import matplotlib.pyplot as plt

input_words = [
    "I",
    "drink",
    "milk"
]

output_words = [
    "Je",
    "bois",
    "lait"
]

attention = np.array([
    [0.8, 0.1, 0.1],
    [0.1, 0.8, 0.1],
    [0.1, 0.1, 0.8]
])

plt.imshow(attention)

plt.xticks(
    range(len(input_words)),
    input_words
)

plt.yticks(
    range(len(output_words)),
    output_words
)

plt.xlabel("Input Words")
plt.ylabel("Output Words")
plt.title("Attention Weights")

plt.colorbar(
    label="Attention Weight"
)

plt.show()