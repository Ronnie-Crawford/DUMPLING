import torch
import torch.nn as nn

def get_device():
    
    """
    Determines the best available device (GPU, MPS, or CPU).
    
    Returns:
        torch.device: The best available device.
    """
    
    if torch.cuda.is_available():
        
        return torch.device('cuda')
    
    elif torch.backends.mps.is_available():
        
        return torch.device('mps')
    
    else:
        
        return torch.device('cpu')

def collate_fn(batch):
    
    """
    Custom collate function to handle batches of variable-length sequences.
    
    Parameters:
        batch (list of tuples): List of (variant_sequence, wildtype_sequence, fitness_value, variant_length, wildtype_length) tuples.
    
    Returns:
        variant_sequences_padded (tensor): Padded variant sequences tensor.
        wildtype_sequences_padded (tensor): Padded wild type sequences tensor.
        fitness_values (tensor): Fitness values tensor.
        variant_lengths (tensor): Lengths tensor for variant sequences.
        wildtype_lengths (tensor): Lengths tensor for wild type sequences.
    """
    
    variant_sequences, wildtype_sequences, fitness_values, variant_lengths, wildtype_lengths = zip(*batch)

    variant_sequences = [torch.tensor(seq) for seq in variant_sequences]
    wildtype_sequences = [torch.tensor(seq) for seq in wildtype_sequences if seq is not None]
    fitness_values = torch.tensor(fitness_values).float()
    variant_lengths = torch.tensor(variant_lengths).long()
    wildtype_lengths = torch.tensor(wildtype_lengths).long() if wildtype_sequences else None

    variant_sequences_padded = torch.nn.utils.rnn.pad_sequence(variant_sequences, batch_first=True)
    wildtype_sequences_padded = torch.nn.utils.rnn.pad_sequence(wildtype_sequences, batch_first=True) if wildtype_sequences else None

    return variant_sequences_padded, wildtype_sequences_padded, fitness_values, variant_lengths, wildtype_lengths
