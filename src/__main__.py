# Standard modules
import argparse
import sys

# Third-party modules
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Local modules
from config_loader import config
from helpers import get_device, compute_metrics
from preprocessing import pad_variable_length_sequences
from embedding import setup_esm, setup_amplify, fetch_esm_embeddings_batched, load_embeddings
from datasets import get_datasets
from splits import handle_splits_flag
from models import set_up_model
from training import train_fitness_finder_from_plm_embeddings_nn, train_energy_and_fitness_finder_from_plm_embeddings_nn
from inference import get_predictions
from tuning import optimise_hyperparameters
from visuals import plot_predictions_vs_true, plot_input_histogram, plot_embeddings

def main(device: str, embeddings: str, splits: str, tune: str):

    """
    Main function of the package; loads data into datasets, generates splits, embeds data, runs models and predicts results on test split.

    Parameters:
        - device (str): The value of the device flag, if any.
        - embeddings (str): The value of the embeddings flag, if any.
        - splits (str): The value of the splits flag, if any.
        - tune (str): The value of the tune flag, if any.
    """

    device = get_device(device)
    datasets = get_datasets()
    model_selections = [key for key, value in config["EMBEDDING_MODELS"].items() if value]

    #datasets = pad_variable_length_sequences(datasets)
    datasets, embedding_size = load_embeddings(embeddings, datasets, config["TRAINING_PARAMETERS"]["BATCH_SIZE"], device, config["DATASETS_IN_USE"], model_selections, config["EMBEDDING_LAYERS"], config["EMBEDDING_TYPES"])
    plot_embeddings(datasets, device = device)

    training_split, validation_split, testing_split = handle_splits_flag(splits, datasets)

    training_loader = DataLoader(training_split, batch_size = config["TRAINING_PARAMETERS"]["BATCH_SIZE"], shuffle = True, num_workers = 0)
    validation_loader = DataLoader(validation_split, batch_size = config["TRAINING_PARAMETERS"]["BATCH_SIZE"], shuffle = True, num_workers = 0)
    testing_loader = DataLoader(testing_split, batch_size = config["TRAINING_PARAMETERS"]["BATCH_SIZE"], shuffle = True, num_workers = 0)

    if tune == None:

        model, criterion, optimiser = set_up_model(embedding_size, config["MODEL_ARCHITECTURE"]["HIDDEN_LAYERS"], config["MODEL_ARCHITECTURE"]["DROPOUT_LAYERS"])
        #trained_model = train_fitness_finder_from_plm_embeddings_nn(model, training_loader, validation_loader, criterion, optimiser, config["TRAINING_PARAMETERS"]["MAX_EPOCHS"], config["TRAINING_PARAMETERS"]["PATIENCE"], device)
        trained_model = train_energy_and_fitness_finder_from_plm_embeddings_nn(model, training_loader, validation_loader, criterion, optimiser, config["TRAINING_PARAMETERS"]["MAX_EPOCHS"], config["TRAINING_PARAMETERS"]["PATIENCE"], device)

        predictions_df = get_predictions(trained_model, testing_loader, criterion, device, "models/plm_embedding_to_simple_nn")

        plot_input_histogram(predictions_df)
        plot_predictions_vs_true(predictions_df)
        compute_metrics("results/test_results.csv", "fitness")
        compute_metrics("results/test_results.csv", "energy")

    else:

        results = optimise_hyperparameters(device, training_loader, validation_loader, testing_loader, tune, embedding_size)
        print(results)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type = str, help = "Manually overides the automatic device detection to use specific device from CPU, MPS or CUDA (GPU).")
    parser.add_argument("--embeddings", type = str, help = "Specify whether new sequence embeddings should be generated, or previously generated embeddings: [new], [saved]")
    parser.add_argument("--splits", type = str, help = "Choose which type of splits to use: homologous-aware, random")
    parser.add_argument("--tune", type = str, help = "Instead of using config hyperparameters, runs tests to recommend optimal hyperparameters. Can use grid-search, random-search.")

    args = parser.parse_args()

    main(device = args.device, embeddings = args.embeddings, splits = args.splits, tune = args.tune)
