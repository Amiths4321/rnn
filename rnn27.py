target_sentence = [
    "<go>",
    "Je",
    "bois",
    "du",
    "lait"
]

expected_output = [
    "Je",
    "bois",
    "du",
    "lait",
    "<eos>"
]


for decoder_input, expected in zip(
    target_sentence,
    expected_output
):
    print(
        "Decoder input:",
        decoder_input,
        "→ Expected:",
        expected
    )