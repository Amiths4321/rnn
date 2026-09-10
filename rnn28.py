import numpy as np

encoder_words = [
    "I",
    "drink",
    "milk"
]

attention_weights = np.array([
    0.1,
    0.1,
    0.8
])

print("Attention weights:")

for word, weight in zip(
    encoder_words,
    attention_weights
):
    print(
        f"{word}: {weight:.1f}"
    )

focused_word = encoder_words[
    attention_weights.argmax()
]

print("\nDecoder is focusing on:", focused_word)