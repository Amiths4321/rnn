from tensorflow import keras
import numpy as np


# -----------------------------------
# 1. Training data
# -----------------------------------

sentences = [
    "I love this movie",
    "This movie is excellent",
    "Amazing movie",
    "I really enjoyed this movie",
    "This film was fantastic",
    "Very good movie",

    "I hate this movie",
    "This movie is terrible",
    "Awful movie",
    "I really disliked this movie",
    "This film was horrible",
    "Very bad movie"
]

# 1 = Positive
# 0 = Negative

labels = np.array([
    1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0
])


# -----------------------------------
# 2. Text → Word IDs
# -----------------------------------

vectorizer = keras.layers.TextVectorization(
    max_tokens=1000,
    output_sequence_length=6
)

vectorizer.adapt(sentences)

X = vectorizer(sentences)

print("Example sentence:")
print(sentences[0])

print("\nConverted to IDs:")
print(X[0].numpy())


# -----------------------------------
# 3. Build LSTM model
# -----------------------------------

model = keras.Sequential([
    keras.layers.Input(shape=(6,)),

    keras.layers.Embedding(
        input_dim=1000,
        output_dim=16
    ),

    keras.layers.LSTM(32),

    keras.layers.Dense(
        1,
        activation="sigmoid"
    )
])


# -----------------------------------
# 4. Compile
# -----------------------------------

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# -----------------------------------
# 5. Train
# -----------------------------------

model.fit(
    X,
    labels,
    epochs=100,
    verbose=0
)

print("\nTraining completed.")


# -----------------------------------
# 6. Test new sentences
# -----------------------------------

test_sentences = [
    "This movie was amazing",
    "I hate this film",
    "Very good film",
    "This movie was horrible"
]

test_X = vectorizer(test_sentences)

predictions = model.predict(
    test_X,
    verbose=0
)


# -----------------------------------
# 7. Display predictions
# -----------------------------------

for sentence, prediction in zip(
    test_sentences,
    predictions
):

    score = prediction[0]

    if score >= 0.5:
        sentiment = "POSITIVE"
    else:
        sentiment = "NEGATIVE"

    print("\nSentence:", sentence)
    print("Score:", round(float(score), 4))
    print("Sentiment:", sentiment)