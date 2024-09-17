def pad_variable_length_sequences(sequences):

    longest_sequence = len(max(sequences, key = len))
    #sequences = [sequence.ljust(longest_sequence, "-") for sequence in sequences]
    assert longest_sequence <= 100, "There are sequences longer than expected."
    sequences = [sequence.ljust(100, "-") for sequence in sequences]

    return sequences
