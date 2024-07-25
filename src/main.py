import pandas as pd
from config import DATA_PATH, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS, TRAIN_SIZE, VAL_SIZE, TEST_SIZE, BATCH_SIZE, LEARNING_RATE, EPOCHS, RANDOM_STATE
from preprocessing import encode_sequences, assign_wt_sequence
from datasets import ProteinDataset
from model import AutoencoderRNN, PredictorNN
from training import train_autoencoder, train_predictor
from inference import predict
from helpers import get_device, collate_fn
from metrics import compute_fitness_metrics, validate_autoencoder
import torch
from torch.utils.data import DataLoader
from splits import split_data

def main():
    # Load the data
    df = pd.read_csv(DATA_PATH)
    dataset_name = 'magda'
    
    # Assign wild type sequences
    df = assign_wt_sequence(df, type_column='type', domain_column='domain', aa_seq_column='aa_seq')
    
    # Split the data
    train_df, val_df, test_df = split_data(df, dataset_name, train_size=TRAIN_SIZE, val_size=VAL_SIZE, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    
    # Encode sequences and get lengths
    variant_train_sequences, variant_train_lengths = encode_sequences(train_df['aa_seq'].tolist())
    wildtype_train_sequences, wildtype_train_lengths = encode_sequences(train_df['wt_sequence'].tolist())
    variant_val_sequences, variant_val_lengths = encode_sequences(val_df['aa_seq'].tolist())
    wildtype_val_sequences, wildtype_val_lengths = encode_sequences(val_df['wt_sequence'].tolist())
    variant_test_sequences, variant_test_lengths = encode_sequences(test_df['aa_seq'].tolist())
    wildtype_test_sequences, wildtype_test_lengths = encode_sequences(test_df['wt_sequence'].tolist())
    
    # Create datasets
    train_dataset = ProteinDataset(variant_train_sequences, wildtype_train_sequences, train_df['scaled_fitness'].values, variant_train_lengths, wildtype_train_lengths)
    val_dataset = ProteinDataset(variant_val_sequences, wildtype_val_sequences, val_df['scaled_fitness'].values, variant_val_lengths, wildtype_val_lengths)
    test_dataset = ProteinDataset(variant_test_sequences, wildtype_test_sequences, test_df['scaled_fitness'].values, variant_test_lengths, wildtype_test_lengths)

    print("Length of train dataset:", len(train_dataset))
    print("Length of validation dataset:", len(val_dataset))
    print("Length of test dataset:", len(test_dataset))

    # Create the models
    autoencoder = AutoencoderRNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS)
    predictor = PredictorNN(HIDDEN_SIZE * 2, HIDDEN_SIZE, OUTPUT_SIZE)

    # Get the best available device
    DEVICE = get_device()
    autoencoder.to(DEVICE)
    predictor.to(DEVICE)
    print("Using device: ", DEVICE)
    
    # Train autoencoder
    train_autoencoder(autoencoder, train_dataset, epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, device=DEVICE)

    # Train predictor
    train_predictor(autoencoder, predictor, train_dataset, epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, device=DEVICE)

    # Predict
    predictions = predict(autoencoder, predictor, test_dataset, batch_size=BATCH_SIZE, device=DEVICE)

    # Compute fitness metrics
    actual_fitness = test_dataset.fitness_values
    fitness_metrics = compute_fitness_metrics(predictions, actual_fitness)
    print("Fitness Metrics:", fitness_metrics)
    
    # Validate autoencoder
    avg_reconstruction_loss = validate_autoencoder(autoencoder, val_dataset, batch_size=BATCH_SIZE, device=DEVICE)
    print("Average Reconstruction Loss on Validation Set:", avg_reconstruction_loss)


if __name__ == "__main__":
    main()
