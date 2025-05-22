# Standard modules
import argparse
import sys
import shutil

# Third-party modules
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Local modules
from config_loader import config
from inference import get_predictions
from tuning import random_search

from embedding import handle_embeddings
from models import handle_models
from inference import handle_inference

from runner import train_and_test, train, test
from benchmarking import all_against_all_benchmarking
from visuals import plot_benchmark_grid

def main(
    device_flag: str,
    splits_flag: str,
    tune_flag: str
    ):

    """
    Main function makes decisions on how the package should be run: training / testing / benchmarking / tuning
    
    To do:
    - Remove handle tuning
    - Remove defunct flags
    - Add new flags and pipeline logic
    """

    train_and_test(config)
    #train(config)
    #test(config)
    #results_path = all_against_all_benchmarking(config)
    
    #for output_feature in config["PREDICTED_FEATURES_LIST"]:
       
    #    plot_benchmark_grid(results_path, output_feature)

def handle_tuning(splits_flag, tune_flag, config, all_dataset_names, all_datasets, dataset_dict, device, package_folder, homology_path, results_path, output_features):

    search_iterations = config["HYPERPARAMETER_TUNING"].get("SEARCH_ITERATIONS", 100)
    min_batch_size = config["HYPERPARAMETER_TUNING"].get("MIN_BATCH_SIZE", 4)
    max_batch_size = config["HYPERPARAMETER_TUNING"].get("MAX_BATCH_SIZE", 256)
    min_hidden_layers = config["HYPERPARAMETER_TUNING"].get("MIN_HIDDEN_LAYERS", 0)
    max_hidden_layers = config["HYPERPARAMETER_TUNING"].get("MAX_HIDDEN_LAYERS", 12)
    min_hidden_size = config["HYPERPARAMETER_TUNING"].get("MIN_HIDDEN_LAYER_SIZE", 2)
    max_hidden_size = config["HYPERPARAMETER_TUNING"].get("MAX_HIDDEN_LAYER_SIZE", 1024)
    min_dropout = config["HYPERPARAMETER_TUNING"].get("MIN_DROPOUT", 0)
    max_dropout = config["HYPERPARAMETER_TUNING"].get("MAX_DROPOUT", 0.5)
    min_learning_rate = config["HYPERPARAMETER_TUNING"].get("MIN_LEARNING_RATE", 0.001)
    max_learning_rate = config["HYPERPARAMETER_TUNING"].get("MAX_LEARNING_RATE", 0.01)
    min_weight_decay = config["HYPERPARAMETER_TUNING"].get("MIN_WEIGHT_DECAY", 0)
    max_weight_decay = config["HYPERPARAMETER_TUNING"].get("MAX_WEIGHT_DECAY", 0.001)
    loss_functions = config["HYPERPARAMETER_TUNING"].get("LOSS_FUNCTONS", ["MSELOSS", "MAELOSS"])
    activation_functions = config["HYPERPARAMETER_TUNING"].get("ACTIVATION_FUNCTIONS", ["LEAKYRELU"])
    
    # Run tuning process if flag provided
    if tune_flag == "grid-search":
    
        pass
    
    elif tune_flag == "random-search":

        random_configs_list = random_search(
            search_iterations,
            min_batch_size,
            max_batch_size,
            min_hidden_layers,
            max_hidden_layers,
            min_hidden_size,
            max_hidden_size,
            min_dropout,
            max_dropout,
            min_learning_rate,
            max_learning_rate,
            min_weight_decay,
            max_weight_decay,
            loss_functions,
            activation_functions,
        )
        
        best_energy_pearson = 0
        best_fitness_pearson = 0
        best_energy_config = None
        best_fitness_config = None
        
        for config in random_configs_list:
            
            downstream_models, activation_functions, loss_functions, optimisers, rnn_type, bidirectional = handle_setup()
            datasets, embedding_size = handle_embeddings(config, downstream_models, all_dataset_names, all_datasets, device, package_folder)
            dataloaders, splits_dict = handle_dataloading(config, dataset_dict, all_dataset_names, package_folder, homology_path, splits_flag)
            model, downstream_models, criterion = handle_models(
                config["hidden_layers"],
                config["dropout_layers"],
                config["learning_rate"],
                config["weight_decay"],
                config["min_epochs"],
                config["max_epochs"],
                config["patience"],
                downstream_models, 
                embedding_size,
                dataloaders, 
                output_features,
                activation_functions,
                loss_functions,
                optimisers,
                rnn_type,
                bidirectional,
                results_path,
                device
                )
            metrics = handle_inference(config, downstream_models, model, dataloaders, criterion, device, package_folder, output_features, results_path)
        
            if metrics["energy"]["Pearson"] > best_energy_pearson:
                
                best_energy_pearson = metrics["energy"]["Pearson"]
                best_energy_config = config
                print("New best Pearsons rank for energy: ", best_energy_pearson)
                print("New best energy config:")
                print("Downstream models: ", config["DOWNSTREAM_MODELS"])
                print("Batch size: ", config["TRAINING_PARAMETERS"]["BATCH_SIZE"])
                print("Hidden layers: ", config["MODEL_ARCHITECTURE"]["HIDDEN_LAYERS"])
                print("Dropout layers: ", config["MODEL_ARCHITECTURE"]["DROPOUT_LAYERS"])
                print("Learning rate: ", config["TRAINING_PARAMETERS"]["LEARNING_RATE"])
                print("Weight decay: ", config["MODEL_ARCHITECTURE"]["WEIGHT_DECAY"])
                print("Loss functions: ", config["MODEL_ARCHITECTURE"]["LOSS_FUNCTIONS"])
                print("Activation functions: ", config["MODEL_ARCHITECTURE"]["ACTIVATION_FUNCTIONS"])
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type = str, default = None, help = "Manually overides the automatic device detection to use specific device from CPU, MPS or CUDA (GPU).")
    parser.add_argument("--splits", type = str, default = "homologous-aware", help = "Choose which type of splits to use: homologous-aware, random")
    parser.add_argument("--tune", type = str, default = None, help = "Instead of using config hyperparameters, runs tests to recommend optimal hyperparameters. Can use grid-search, random-search.")
    parser.add_argument("--leakage", type = bool, default = False, help = "Whether to allow leakage between training data and data for inference, default: False.")

    args = parser.parse_args()

    main(device_flag = args.device, splits_flag = args.splits, tune_flag = args.tune)
