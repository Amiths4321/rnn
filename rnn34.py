tokens = {
    "<go>": 0,
    "je": 1,
    "bois": 2,
    "lait": 3,
    "<eos>": 4
}

decoder_input = [
    tokens["<go>"],
    tokens["je"],
    tokens["bois"],
    tokens["lait"]
]

expected_output = [
    tokens["je"],
    tokens["bois"],
    tokens["lait"],
    tokens["<eos>"]
]

print("Decoder input:")
print(decoder_input)

print("\nExpected output:")
print(expected_output)