import pandas as pd
import numpy as np

def encode_sequences(sequences, amino_acids='ACDEFGHIKLMNPQRSTVWY'):
    """
    Encodes amino acid sequences into numerical format without padding.
    
    Parameters:
        sequences (list): List of amino acid sequences.
        amino_acids (str): String of unique amino acids used for encoding.
    
    Returns:
        sequences_int (list of lists): Encoded sequences represented as lists of integers.
        seq_lengths (list): Lengths of each sequence.
    """
    
    one_hot_residues = {residue : index + 1 for index, residue in enumerate(amino_acids)}                       # Dictionary to one-hot encode residues as ints
    encoded_sequences = [[one_hot_residues[residue] for residue in sequence] for sequence in sequences]         # Encodes all sequences using dictionary
    sequence_lengths = [len(sequence) for sequence in encoded_sequences]                                        # Finds the length of each sequence

    return encoded_sequences, sequence_lengths

def assign_wt_sequence(df, type_column='type', domain_column='domain', aa_seq_column='aa_seq'):
    """
    Assigns a wild type sequence to each variant based on the wild type variant for each domain.
    
    Parameters:
        df (DataFrame): The input data.
        type_column (str): The name of the column indicating the type of the variant.
        domain_column (str): The name of the column containing the domain information.
        aa_seq_column (str): The name of the column containing the amino acid sequences.
    
    Returns:
        DataFrame: DataFrame with an additional 'wt_sequence' column.
    """
    wt_sequences = df[df[type_column] == 'wt'].groupby(domain_column)[aa_seq_column].first().to_dict()
    df['wt_sequence'] = df[domain_column].map(wt_sequences)
    return df
