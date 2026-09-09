import numpy as np

# Previous memory
old_memory = 0.8

# New information
new_information = 0.6


# Gate values between 0 and 1
forget_gate = 0.2
input_gate = 0.9
output_gate = 0.8


# Step 1 — Forget some old memory
remaining_memory = old_memory * forget_gate

# Step 2 — Store some new information
updated_memory = (
    remaining_memory
    + new_information * input_gate
)

# Step 3 — Produce output
output = updated_memory * output_gate


print("Remaining old memory:", remaining_memory)
print("Updated memory:", updated_memory)
print("Output:", output)