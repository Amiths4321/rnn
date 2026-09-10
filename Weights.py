from tensorflow import keras
import matplotlib.pyplot as plt

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

# plt.imshow(
#     weights[:, 0].reshape(28, 28),
#     cmap="gray"
# )

plt.title("Feature Learned by Neuron 1")
plt.axis("off")

# plt.show()

