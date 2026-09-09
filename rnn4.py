import numpy as np
import tensorflow as tf
from tensorflow import keras

# 1. Define data
sentences = [
    "I love this movie",
    "This movie is excellent",
    "I really enjoyed it",
    "Amazing movie",
    "I hate this movie",
    "This movie is terrible",
    "I really disliked it",
    "Awful movie"
]

labels = np.array([
    1,
    1,
    1,
    1,
    0,
    0,
    0,
    0
])

# 2. Vectorize text and convert EagerTensor to a NumPy array
vectorizer = keras.layers.TextVectorization(
    max_tokens=1000,
    output_sequence_length=5
)

vectorizer.adapt(sentences)
X = vectorizer(sentences).numpy()  # Convert EagerTensor to NumPy array

print(X)

# 3. Build model
model = keras.Sequential([
    keras.layers.Input(shape=(5,)),
    keras.layers.Embedding(
        input_dim=1000,
        output_dim=16
    ),
    keras.layers.SimpleRNN(16),
    keras.layers.Dense(
        1,
        activation="sigmoid"
    )
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# 4. Train model
model.fit(
    X,
    labels,
    epochs=100,
    verbose=0,
    validation_split=0.1
)

# 5. Predict
test_sentence = [
    "I love this"
]

test_vector = vectorizer(test_sentence)

prediction = model.predict(
    test_vector,
    verbose=0
)[0][0]

print("Probability:", prediction)

if prediction >= 0.5:
    print("Sentiment: POSITIVE")
else:
    print("Sentiment: NEGATIVE")