import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

plt.plot(
    sequence,
    label="Original + Generated"
)

plt.axvline(
    n_steps,
    linestyle="--",
    label="Generation starts"
)

plt.title("Creative RNN")

plt.xlabel("Time")
plt.ylabel("Value")

plt.legend()

plt.show()