import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from helpers import collate_fn

def train_autoencoder(autoencoder, dataset, epochs=10, batch_size=32, learning_rate=0.001, device='cpu'):
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    criterion_reconstruction = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
    
    autoencoder.to(device)
    
    for epoch in range(epochs):
        autoencoder.train()
        total_loss_reconstruction = 0
        for variant_sequences, wildtype_sequences, fitness_values, variant_lengths, wildtype_lengths in dataloader:
            variant_sequences = variant_sequences.to(device, dtype=torch.long)
            wildtype_sequences = wildtype_sequences.to(device, dtype=torch.long) if wildtype_sequences is not None else None
            fitness_values = fitness_values.to(device, dtype=torch.float32)
            variant_lengths = variant_lengths.to('cpu', dtype=torch.int64)  # Ensure int64 type on CPU
            wildtype_lengths = wildtype_lengths.to('cpu', dtype=torch.int64) if wildtype_sequences is not None else None
            seq_len = variant_sequences.size(1)
            optimizer.zero_grad()
            reconstructed, _ = autoencoder(variant_sequences, variant_lengths, wildtype_sequences, wildtype_lengths if wildtype_sequences is not None else None, seq_len)
            reconstructed = reconstructed.squeeze(-1)
            loss_reconstruction = criterion_reconstruction(reconstructed, variant_sequences.float())
            loss_reconstruction.backward()
            optimizer.step()
            total_loss_reconstruction += loss_reconstruction.item()
        
        avg_loss_reconstruction = total_loss_reconstruction / len(dataloader)
        
        print(f'Epoch [{epoch+1}/{epochs}], Reconstruction Loss: {avg_loss_reconstruction:.4f}')

def train_predictor(autoencoder, predictor, dataset, epochs=10, batch_size=32, learning_rate=0.001, device='cpu'):
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    criterion_prediction = nn.MSELoss()
    optimizer = torch.optim.Adam(predictor.parameters(), lr=learning_rate)
    
    autoencoder.encoder.to(device)
    predictor.to(device)
    
    for epoch in range(epochs):
        predictor.train()
        total_loss_prediction = 0
        for variant_sequences, wildtype_sequences, fitness_values, variant_lengths, wildtype_lengths in dataloader:
            variant_sequences = variant_sequences.to(device, dtype=torch.long)
            wildtype_sequences = wildtype_sequences.to(device, dtype=torch.long) if wildtype_sequences is not None else None
            fitness_values = fitness_values.to(device, dtype=torch.float32)
            variant_lengths = variant_lengths.to('cpu', dtype=torch.int64)
            wildtype_lengths = wildtype_lengths.to('cpu', dtype=torch.int64) if wildtype_sequences is not None else None
            with torch.no_grad():
                variant_latent = autoencoder.encoder(variant_sequences, variant_lengths)
                if wildtype_sequences is not None:
                    wildtype_latent = autoencoder.encoder(wildtype_sequences, wildtype_lengths)
                    combined_latent = torch.cat((variant_latent, wildtype_latent), dim=1)
                else:
                    combined_latent = torch.cat((variant_latent, torch.zeros_like(variant_latent)), dim=1)
            optimizer.zero_grad()
            predicted_fitness = predictor(combined_latent)
            loss_prediction = criterion_prediction(predicted_fitness.squeeze(), fitness_values)
            loss_prediction.backward()
            optimizer.step()
            total_loss_prediction += loss_prediction.item()
        
        avg_loss_prediction = total_loss_prediction / len(dataloader)
        
        print(f'Epoch [{epoch+1}/{epochs}], Prediction Loss: {avg_loss_prediction:.4f}')
