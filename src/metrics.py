import numpy as np
from scipy.stats import pearsonr, spearmanr
import torch.nn as nn
import torch
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader
from helpers import collate_fn

def compute_fitness_metrics(predictions, targets):
    
    """
    Computes the Mean Squared Error (MSE), Pearson's rank correlation, and Spearman's rank correlation
    for fitness predictions.
    
    Parameters:
        predictions (numpy array): Predicted fitness values.
        targets (numpy array): Actual fitness values.
    
    Returns:
        dict: Dictionary containing MSE, Pearson's correlation, and Spearman's correlation.
    """
    
    mse = np.mean((predictions - targets) ** 2)
    pearson_corr, _ = pearsonr(predictions, targets)
    spearman_corr, _ = spearmanr(predictions, targets)
    
    return {
        'MSE': mse,
        'Pearson Correlation': pearson_corr,
        'Spearman Correlation': spearman_corr
    }

def compute_autoencoder_metrics(reconstructed, original):
    
    """
    Computes the reconstruction loss (cross-entropy) for autoencoder reconstructions.
    
    Parameters:
        reconstructed (torch.Tensor): Reconstructed sequences (logits).
        original (torch.Tensor): Original sequences (one-hot encoded or indices).
    
    Returns:
        dict: Dictionary containing the reconstruction loss.
    """
    
    mse_loss = nn.MSELoss()
    loss = mse_loss(reconstructed, original)
    
    return {'Reconstruction Loss': loss.item()}

def validate_autoencoder(autoencoder, val_dataset, batch_size, device):
    
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)
    autoencoder.eval()
    total_reconstruction_loss = 0
    with torch.no_grad():
        for variant_sequences, wildtype_sequences, _, variant_lengths, wildtype_lengths in val_dataloader:
            variant_sequences = variant_sequences.to(device)
            wildtype_sequences = wildtype_sequences.to(device) if wildtype_sequences is not None else None
            variant_lengths = variant_lengths.cpu()
            wildtype_lengths = wildtype_lengths.cpu() if wildtype_lengths is not None else None
            seq_len = variant_sequences.size(1)
            reconstructed, _ = autoencoder(variant_sequences, variant_lengths, wildtype_sequences, wildtype_lengths, seq_len)
            
            reconstructed = reconstructed.squeeze(-1)
            variant_sequences = variant_sequences.float()
            reconstruction_loss = compute_autoencoder_metrics(reconstructed, variant_sequences)
            
            total_reconstruction_loss += reconstruction_loss['Reconstruction Loss']
    
    avg_reconstruction_loss = total_reconstruction_loss / len(val_dataloader)
    
    return avg_reconstruction_loss
