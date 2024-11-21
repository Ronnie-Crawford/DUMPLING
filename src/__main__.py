# Standard modules
import argparse
import sys
import shutil

# Third-party modules
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

# Local modules
from config_loader import config
from helpers import get_device, setup_folders, get_results_path, get_homology_path, remove_homologous_sequences_from_inference
from preprocessing import pad_variable_length_sequences
from embedding import setup_esm, setup_amplify, fetch_esm_embeddings_batched, load_embeddings
from datasets import get_datasets
from splits import handle_splits_flag
from models import set_up_model
from training import train_fitness_finder_from_plm_embeddings_nn, train_energy_and_fitness_finder_from_plm_embeddings_nn
from inference import get_predictions
from tuning import optimise_hyperparameters
from visuals import handle_embedding_plots
from results import compute_metrics
from homology import handle_homology

def main(device_flag: str, splits_flag: str, tune_flag: str):

    """
    Main function of the package; loads data into datasets, generates splits, embeds data, runs models and predicts results on test split.

    Parameters:
        - device (str): The value of the device flag, if any.
        - splits (str): The value of the splits flag, if any.
        - tune (str): The value of the tune flag, if any.
    """

    device = get_device(device_flag)
    package_folder = setup_folders()
    results_path = get_results_path(package_folder)
    
    try:
    
        # Set up datasets
        all_dataset_names = config["DATASETS_FOR_TRAINING"] + list(set(config["DATASETS_FOR_INFERENCE"]) - set(config["DATASETS_FOR_TRAINING"]))
        all_datasets = get_datasets(all_dataset_names, package_folder)

        dataset_dict = dict(zip(all_dataset_names, all_datasets))
        
        # Check homology between domains if needed
        homology_path = get_homology_path(package_folder, all_dataset_names)
        
        if splits_flag == "homologous-aware":

            handle_homology(all_dataset_names, all_datasets, homology_path)

        embedding_models = config.get("EMBEDDING_MODELS")
        embedding_model_names = [key for key, value in embedding_models.items() if value]

        datasets, embedding_size = load_embeddings(
            all_dataset_names,
            all_datasets,
            config["TRAINING_PARAMETERS"]["BATCH_SIZE"],
            embedding_model_names,
            config["EMBEDDING_LAYERS"],
            config["EMBEDDING_TYPES"],
            device,
            package_folder
            )
        
        handle_embedding_plots(all_datasets, device, results_path)
        
        training_dataset_names = config["DATASETS_FOR_TRAINING"]
        inference_dataset_names = config["DATASETS_FOR_INFERENCE"]
        inference_only_dataset_names = list(set(inference_dataset_names) - set(training_dataset_names))
        overlapping_dataset_names = list(set(inference_dataset_names) & set(training_dataset_names))
        
        training_datasets = [dataset_dict[name] for name in training_dataset_names]
        inference_only_datasets = [dataset_dict[name] for name in inference_only_dataset_names]
        overlapping_datasets = [dataset_dict[name] for name in overlapping_dataset_names]
        
        inference_only_datasets = remove_homologous_sequences_from_inference(all_dataset_names, inference_only_datasets, training_datasets, homology_path)
        
        splits = handle_splits_flag(splits_flag, all_dataset_names, training_dataset_names, training_datasets, homology_path)
        splits["inference"] = inference_only_datasets

        splits["training"] = ConcatDataset(splits["training"])
        splits["validation"] = ConcatDataset(splits["validation"])
        splits["testing_inference"] = ConcatDataset(splits["testing"] + splits["inference"])

        training_loader = DataLoader(splits["training"], batch_size = config["TRAINING_PARAMETERS"]["BATCH_SIZE"], shuffle = True, num_workers = config["TRAINING_PARAMETERS"]["N_WORKERS"])
        validation_loader = DataLoader(splits["validation"], batch_size = config["TRAINING_PARAMETERS"]["BATCH_SIZE"], shuffle = True, num_workers = config["TRAINING_PARAMETERS"]["N_WORKERS"])
        testing_loader = DataLoader(splits["testing_inference"], batch_size = config["TRAINING_PARAMETERS"]["BATCH_SIZE"], shuffle = False, num_workers = config["TRAINING_PARAMETERS"]["N_WORKERS"])

        activation_functions = [key for key, value in config["MODEL_ARCHITECTURE"].get("ACTIVATION_FUNCTION", {}).items() if value]

        if tune_flag is None:

            model, criterion, optimiser = set_up_model(embedding_size, config["MODEL_ARCHITECTURE"]["HIDDEN_LAYERS"], config["MODEL_ARCHITECTURE"]["DROPOUT_LAYERS"], activation_functions, config["TRAINING_PARAMETERS"]["LEARNING_RATE"], config["MODEL_ARCHITECTURE"]["WEIGHT_DECAY"])
            #trained_model = train_fitness_finder_from_plm_embeddings_nn(model, training_loader, validation_loader, criterion, optimiser, config["TRAINING_PARAMETERS"]["MAX_EPOCHS"], config["TRAINING_PARAMETERS"]["PATIENCE"], device)
            trained_model = train_energy_and_fitness_finder_from_plm_embeddings_nn(model, training_loader, validation_loader, criterion, optimiser, results_path, config["TRAINING_PARAMETERS"]["MAX_EPOCHS"], config["TRAINING_PARAMETERS"]["PATIENCE"], device)

            predictions_df = get_predictions(trained_model, testing_loader, criterion, device, package_folder, results_path)

            compute_metrics(results_path, "fitness")
            compute_metrics(results_path, "energy")

        else:

            results = optimise_hyperparameters(device, training_loader, validation_loader, testing_loader, tune, embedding_size)
            print(results)
            
    except Exception as error:
        
        print("Error, deleting results.")
        shutil.rmtree(results_path)
        raise Exception(error)
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type = str, default = None, help = "Manually overides the automatic device detection to use specific device from CPU, MPS or CUDA (GPU).")
    parser.add_argument("--splits", type = str, default = "homologous-aware", help = "Choose which type of splits to use: homologous-aware, random")
    parser.add_argument("--tune", type = str, default = None, help = "Instead of using config hyperparameters, runs tests to recommend optimal hyperparameters. Can use grid-search, random-search.")
    parser.add_argument("--leakage", type = bool, default = False, help = "Whether to allow leakage between training data and data for inference, default: False.")

    args = parser.parse_args()

    main(device_flag = args.device, splits_flag = args.splits, tune_flag = args.tune)
