import torch
from torch.utils.data import Dataset

import pandas as pd

class DatasetInfo:
    
    def __init__(self, domain_column, aa_seq_column, wt_seq_column, fitness_column, type_column=None):
        
        self.domain_column = domain_column
        self.aa_seq_column = aa_seq_column
        self.wt_seq_column = wt_seq_column
        self.fitness_column = fitness_column
        self.type_column = type_column

DATASET_INFO = {
    
    'magda': DatasetInfo(
        domain_column='domain',
        aa_seq_column='aa_seq',
        fitness_column='scaled_fitness',
        type_column='type'
    ),
    'toni': DatasetInfo(
        domain_column='domain_ID',
        aa_seq_column='aa_seq',
        fitness_column='normalized_fitness',
        type_column=None
    ),
    'megascale': DatasetInfo(
        domain_column='family',
        aa_seq_column='amino_acid_seq',
        wt_seq_column='wt_seq',
        fitness_column='ddG_ML',
        type_column=None
    )
}

def get_dataset_info(dataset_name):
    
    return DATASET_INFO.get(dataset_name)

class ProteinDataset(Dataset):
    
    def __init__(self, variant_sequences, wildtype_sequences, fitness_values, variant_lengths, wildtype_lengths):
        
        self.variant_sequences = variant_sequences
        self.wildtype_sequences = wildtype_sequences
        self.fitness_values = fitness_values
        self.variant_lengths = variant_lengths
        self.wildtype_lengths = wildtype_lengths
        
    def __len__(self):
        
        return len(self.variant_sequences)

    def __getitem__(self, idx):
        
        variant_sequence = self.variant_sequences[idx]
        wildtype_sequence = self.wildtype_sequences[idx]
        fitness_value = self.fitness_values[idx]
        variant_length = self.variant_lengths[idx]
        wildtype_length = self.wildtype_lengths[idx]
        
        return variant_sequence, wildtype_sequence, fitness_value, variant_length, wildtype_length



