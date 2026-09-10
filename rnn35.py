id_to_word = {
    0: "<go>",
    1: "Je",
    2: "bois",
    3: "lait",
    4: "<eos>"
}

predicted_ids = [1, 2, 3, 4]

for token_id in predicted_ids:

    word = id_to_word[token_id]

    print("Model predicted:", word)

    if word == "<eos>":
        print("Translation finished.")
        break