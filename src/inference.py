import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from helpers import collate_fn

def predict(autoencoder, predictor, dataset, batch_size, device):
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    autoencoder.eval()
    predictor.eval()
    predictions = []

    with torch.no_grad():
        for variant_sequences, wildtype_sequences, fitness_values, variant_lengths, wildtype_lengths in dataloader:
            variant_sequences = variant_sequences.to(device)
            wildtype_sequences = wildtype_sequences.to(device) if wildtype_sequences is not None else None
            variant_lengths = variant_lengths.cpu().to(torch.int64)
            wildtype_lengths = wildtype_lengths.cpu().to(torch.int64) if wildtype_lengths is not None else None

            variant_latent = autoencoder.encoder(variant_sequences, variant_lengths)
            wildtype_latent = autoencoder.encoder(wildtype_sequences, wildtype_lengths) if wildtype_sequences is not None else None
            combined_latent = torch.cat((variant_latent, wildtype_latent), dim=1) if wildtype_latent is not None else variant_latent
            predicted_fitness = predictor(combined_latent)
            predictions.append(predicted_fitness.cpu().numpy())

    return np.concatenate(predictions).ravel()









