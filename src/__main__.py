# Set up environment to manage PyTorch memory
#import os
#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Standard modules
import argparse

# Third-party modules
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# Local modules
from config_loader import config
from helpers import get_device, compute_metrics, plot_predictions_vs_true
from preprocessing import pad_variable_length_sequences
from protein_language_models import setup_esm, setup_amplify, fetch_esm_embeddings_batched, load_embeddings
from datasets import get_datasets
from splits import handle_splits_flag
from models import set_up_model
from training import train_fitness_finder_from_plm_embeddings_nn, train_energy_and_fitness_finder_from_plm_embeddings_nn
from inference import get_predictions
from tuning import optimise_hyperparameters

def main(device: str, embeddings: str, splits: str, tune: str):

    if device == "cuda":
        
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    DEVICE = get_device(device)
    datasets = get_datasets()

    model_selections = [key for key, value in config["MODELS"]["PLM_STABILITY_MODULES"].items() if value]

    #setup_amplify()

    datasets, embedding_size = load_embeddings(datasets, embeddings, model_selections, DEVICE, config["DATASETS_IN_USE"])
    training_split, validation_split, testing_split = handle_splits_flag(splits, datasets)

    training_loader = DataLoader(training_split, batch_size = config["TRAINING_PARAMETERS"]["BATCH_SIZE"], shuffle = True, num_workers = 0)
    validation_loader = DataLoader(validation_split, batch_size = config["TRAINING_PARAMETERS"]["BATCH_SIZE"], shuffle = True, num_workers = 0)
    testing_loader = DataLoader(testing_split, batch_size = config["TRAINING_PARAMETERS"]["BATCH_SIZE"], shuffle = True, num_workers = 0)

    if tune == None:

        model, criterion, optimiser = set_up_model(embedding_size, config["MODEL_ARCHITECTURE"]["HIDDEN_LAYERS"], config["MODEL_ARCHITECTURE"]["DROPOUT_LAYERS"])
        #trained_model = train_fitness_finder_from_plm_embeddings_nn(model, training_loader, validation_loader, criterion, optimiser, config["TRAINING_PARAMETERS"]["MAX_EPOCHS"], config["TRAINING_PARAMETERS"]["PATIENCE"], DEVICE)
        trained_model = train_energy_and_fitness_finder_from_plm_embeddings_nn(model, training_loader, validation_loader, criterion, optimiser, config["TRAINING_PARAMETERS"]["MAX_EPOCHS"], config["TRAINING_PARAMETERS"]["PATIENCE"], DEVICE)

        predictions_df = get_predictions(trained_model, testing_loader, criterion, DEVICE, "models/plm_embedding_to_simple_nn")

        plot_predictions_vs_true(predictions_df)
        compute_metrics("results/test_results.csv", "fitness")
        compute_metrics("results/test_results.csv", "energy")

    else:

        results = optimise_hyperparameters(DEVICE, training_loader, validation_loader, testing_loader, tune, embedding_size)
        print(results)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type = str, help = "Manually overides the automatic device detection to use specific device from CPU, MPS or CUDA (GPU).")
    parser.add_argument("--embeddings", type = str, help = "Specify whether new sequence embeddings should be generated, or previously generated embeddings: [new], [saved]")
    parser.add_argument("--splits", type = str, help = "Choose which type of splits to use: homologous-aware, random")
    parser.add_argument("--tune", type = str, help = "Instead of using config hyperparameters, runs tests to recommend optimal hyperparameters. Can use grid-search, random-search.")

    args = parser.parse_args()

    main(device = args.device, embeddings = args.embeddings, splits = args.splits, tune = args.tune)
