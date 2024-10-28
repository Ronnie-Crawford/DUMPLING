def pad_variable_length_sequences(datasets):

    for dataset in datasets:

        sequences = dataset.variant_aa_seqs
        longest_sequence = len(max(sequences, key = len))
        assert longest_sequence <= 100, "There are sequences longer than expected."
        dataset.variant_aa_seqs = [sequence.ljust(100, "-") for sequence in sequences]

    return datasets
