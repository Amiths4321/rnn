import numpy as np
from tensorflow import keras

# --------------------------------------------------
# 1. Training data
# --------------------------------------------------

X = np.array([
    [[1], [2], [3]],
    [[2], [3], [4]],
    [[3], [4], [5]],
    [[4], [5], [6]],
    [[5], [6], [7]],
    [[6], [7], [8]],
    [[7], [8], [9]],
    [[8], [9], [10]]
], dtype=np.float32)

y = np.array([
    [[2], [3], [4]],
    [[3], [4], [5]],
    [[4], [5], [6]],
    [[5], [6], [7]],
    [[6], [7], [8]],
    [[7], [8], [9]],
    [[8], [9], [10]],
    [[9], [10], [11]]
], dtype=np.float32)


# --------------------------------------------------
# 2. Decoder input
# Teacher forcing
# --------------------------------------------------

decoder_input = np.concatenate(
    [
        np.zeros((len(X), 1, 1)),
        y[:, :-1, :]
    ],
    axis=1
)

print("Encoder input:", X.shape)
print("Decoder input:", decoder_input.shape)
print("Target:", y.shape)


# --------------------------------------------------
# 3. Encoder
# --------------------------------------------------

encoder_input = keras.Input(shape=(3, 1))

encoder_outputs = keras.layers.LSTM(
    32,
    return_sequences=True
)(encoder_input)


# --------------------------------------------------
# 4. Decoder
# --------------------------------------------------

decoder_input_layer = keras.Input(shape=(3, 1))

decoder_outputs = keras.layers.LSTM(
    32,
    return_sequences=True
)(decoder_input_layer)


# --------------------------------------------------
# 5. Attention
# --------------------------------------------------

attention_output = keras.layers.Attention()(
    [decoder_outputs, encoder_outputs]
)


# --------------------------------------------------
# 6. Combine decoder + attention
# --------------------------------------------------

combined = keras.layers.Concatenate()(
    [decoder_outputs, attention_output]
)


# --------------------------------------------------
# 7. Final prediction
# --------------------------------------------------

output = keras.layers.Dense(1)(combined)


# --------------------------------------------------
# 8. Create model
# --------------------------------------------------

model = keras.Model(
    inputs=[encoder_input, decoder_input_layer],
    outputs=output
)


# --------------------------------------------------
# 9. Compile
# --------------------------------------------------

model.compile(
    optimizer="adam",
    loss="mse"
)


# --------------------------------------------------
# 10. Train
# --------------------------------------------------

model.fit(
    [X, decoder_input],
    y,
    epochs=100,
    verbose=0
)


# --------------------------------------------------
# 11. Test
# --------------------------------------------------

test_input = np.array(
    [[[10], [11], [12]]],
    dtype=np.float32
)

test_decoder_input = np.array(
    [[[0], [13], [14]]],
    dtype=np.float32
)

prediction = model.predict(
    [test_input, test_decoder_input],
    verbose=0
)

print("\nPrediction:")
print(prediction.reshape(-1))