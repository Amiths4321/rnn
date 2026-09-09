gradient = 1.0

for step in range(10):

    gradient = gradient * 0.5

    print(
        f"Step {step + 1}: {gradient}"
    )