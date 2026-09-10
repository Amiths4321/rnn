from tensorflow import keras
import numpy as np


# --------------------------------
# Step 1 — Training data
# --------------------------------

sentences = [
    "I love this movie",
    "This movie is excellent",
    "Amazing movie",
    "I really enjoyed it",

    "I hate this movie",
    "This movie is terrible",
    "Awful movie",
    "I really disliked it"
]

labels = [
    1, 1, 1, 1,
    0, 0, 0, 0
]


# --------------------------------
# Step 2 — Convert text to word IDs
# --------------------------------

vectorizer = keras.layers.TextVectorization(
    max_tokens=1000,
    output_sequence_length=5
)

vectorizer.adapt(sentences)

X = vectorizer(sentences)

print("Word IDs:")
print(X)


# --------------------------------
# Step 3 — Build the NLP model
# --------------------------------

model = keras.Sequential([

    keras.layers.Input(shape=(5,)),

    keras.layers.Embedding(
        input_dim=1000,
        output_dim=16
    ),

    keras.layers.LSTM(16),

    keras.layers.Dense(
        1,
        activation="sigmoid"
    )
])


# --------------------------------
# Step 4 — Compile
# --------------------------------

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# --------------------------------
# Step 5 — Train
# --------------------------------

model.fit(
    x=X.numpy(),
    y=np.array(labels),
    epochs=100,
    verbose=0
)


# --------------------------------
# Step 6 — Test
# --------------------------------

test_sentence = [
    "I love this movie"
]

test_vector = vectorizer(test_sentence)

prediction = model.predict(
    test_vector,
    verbose=0
)[0][0]


print("\nPrediction probability:", prediction)

if prediction >= 0.5:
    print("Sentiment: POSITIVE")
else:
    print("Sentiment: NEGATIVE")