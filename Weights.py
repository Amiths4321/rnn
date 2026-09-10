from tensorflow import keras

# Define or load your trained model first so 'model' exists
model = keras.Sequential([
    keras.layers.Input(shape=(4,)),
    keras.layers.Dense(2, activation="relu"),
    keras.layers.Dense(4, activation="linear")
])

# Access the weights from the first Dense layer (index 1, since index 0 is the InputLayer)
weights, biases = model.layers[1].get_weights()

print("Weights:", weights)
print("Biases:", biases)