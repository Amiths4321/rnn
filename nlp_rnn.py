from tensorflow import keras


sentences = [
    "I love this movie",
    "This movie is excellent",
    "Amazing movie",
    "I hate this movie",
    "This movie is terrible",
    "Awful movie"
]


# Convert words to integer IDs
vectorizer = keras.layers.TextVectorization(
    max_tokens=1000,
    output_sequence_length=5
)

vectorizer.adapt(sentences)

X = vectorizer(sentences)

print("Word IDs:")
print(X)


# Create embedding layer
embedding = keras.layers.Embedding(
    input_dim=1000,
    output_dim=8
)

embedded_words = embedding(X)

print("\nEmbedding shape:")
print(embedded_words.shape)