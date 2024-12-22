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
from helpers import get_device, setup_folders, get_results_path, get_homology_path, remove_homologous_sequences_from_inference, concat_splits, splits_to_loaders
from preprocessing import pad_variable_length_sequences
from embedding import load_embeddings
from datasets import get_datasets
from splits import handle_splits_flag
from models import set_up_model
from training import handle_training_models
from inference import get_predictions
from tuning import random_search
from visuals import handle_embedding_plots
from results import compute_metrics
from homology import handle_homology

def main(
    device_flag: str,
    splits_flag: str,
    tune_flag: str
    ):

    """
    Main function of the package; loads data into datasets, generates splits, embeds data, runs models and predicts results on test split.

    Parameters:
        - device (str): The value of the device flag, if any.
        - splits (str): The value of the splits flag, if any.
        - tune (str): The value of the tune flag, if any.
    """

    output_features = ["fitness"]

    flags_dict = {
        "device": device_flag,
        "splits": splits_flag,
        "tune": tune_flag
    }

    device = get_device(flags_dict["device"])
    paths_dict = {"base": setup_folders()}
    paths_dict["results"] = get_results_path(paths_dict["base"])
    
    try:
    
        datasets_dict, paths_dict = handle_data(config, flags_dict, paths_dict)
        
        # If not tuning, just train the model as normal with given parameters
        if tune_flag == None:
        
            downstream_models, activation_functions, loss_functions, optimisers, rnn_type, bidirectional = handle_setup()
            datasets_dict, embedding_size = handle_embeddings(config, downstream_models, datasets_dict, device, paths_dict)
            dataloaders_dict = handle_dataloading(config, datasets_dict, paths_dict, flags_dict)
            model, downstream_models, criterion = handle_models(
                config,
                downstream_models, 
                embedding_size,
                dataloaders_dict, 
                output_features,
                activation_functions,
                loss_functions,
                optimisers,
                rnn_type,
                bidirectional,
                paths_dict["results"],
                device
                )
            metrics = handle_inference(config, downstream_models, model, dataloaders_dict, criterion, device, output_features, paths_dict)
        
        else:
            
            handle_tuning(flags_dict, config, datasets_dict, device, paths_dict, output_features)
            
    except Exception as error:
        
        print("Error, deleting results.")
        shutil.rmtree(paths_dict["results"])
        raise Exception(error)

def handle_data(config: dict, flags_dict: dict, paths_dict: dict):
    
    # Set up datasets - need to sort out dictionary later
    all_dataset_names = config["DATASETS_FOR_TRAINING"] + list(set(config["DATASETS_FOR_INFERENCE"]) - set(config["DATASETS_FOR_TRAINING"]))
    all_datasets = get_datasets(all_dataset_names, paths_dict["base"])
    datasets_dict = {}
    datasets_dict["all"] = dict(zip(all_dataset_names, all_datasets))
    
    # Check homology between domains if needed
    paths_dict["homology"] = get_homology_path(paths_dict["base"], datasets_dict["all"].keys())
    
    if flags_dict["splits"] == "homologous-aware":

        new_handle_homology(datasets_dict, paths_dict["homology"])
        #handle_homology(datasets_dict, paths_dict["homology"])
    
    return datasets_dict, paths_dict

def handle_setup():
    
    # Fetch settings from config
    downstream_models = [key for key, value in config.get("DOWNSTREAM_MODELS", {}).items() if value]
    activation_functions = [key for key, value in config["MODEL_ARCHITECTURE"].get("ACTIVATION_FUNCTIONS", {}).items() if value]
    loss_functions = [key for key, value in config["MODEL_ARCHITECTURE"].get("LOSS_FUNCTIONS", {}).items() if value]
    optimisers = [key for key, value in config["MODEL_ARCHITECTURE"].get("OPTIMISERS", {}).items() if value]
    rnn_type = downstream_models[0].split("_")[0]
    bidirectional = False
    
    if "_" in downstream_models[0]:
        
        parts = downstream_models[0].split("_")
        bidirectional = (len(parts) > 1 and parts[1] == "BIDIRECTIONAL")
    
    return downstream_models, activation_functions, loss_functions, optimisers, rnn_type, bidirectional

def handle_embeddings(
    config: dict,
    downstream_models: list,
    datasets_dict: dict,
    device: str,
    paths_dict: dict
    ):
    
    embedding_model_names = [key for key, value in config.get("EMBEDDING_MODELS").items() if value]

    # RNN model can only use unprocessed embeddings
    if downstream_models != ["FFNN"]:
        
        embedding_types = ["RAW"]
    
    else:
        
        embedding_types = config.get("EMBEDDING_TYPES")

    datasets_dict, embedding_size = load_embeddings(
        datasets_dict,
        config["TRAINING_PARAMETERS"]["BATCH_SIZE"],
        embedding_model_names,
        config["EMBEDDING_LAYERS"],
        embedding_types,
        device,
        paths_dict["base"]
        )
    
    # Don't currently have a way to visualise unprocessed embeddings, would be good for the future
    #if "RAW" not in embedding_types:
    
    #    handle_embedding_plots(datasets_dict["all"].values(), device, paths_dict["results"])
    
    return datasets_dict, embedding_size

def handle_dataloading(
    config: str,
    datasets_dict: dict,
    paths_dict: dict,
    flags_dict: dict
):
    
    # Organise data for training, testing, and overlap
    training_dataset_names = config["DATASETS_FOR_TRAINING"]
    inference_dataset_names = config["DATASETS_FOR_INFERENCE"]
    inference_only_dataset_names = list(set(inference_dataset_names) - set(training_dataset_names))
    overlapping_dataset_names = list(set(inference_dataset_names) & set(training_dataset_names))

    datasets_dict["train"] = dict(zip(training_dataset_names, [datasets_dict["all"][name] for name in training_dataset_names]))
    datasets_dict["inference_only"] = dict(zip(inference_only_dataset_names, [datasets_dict["all"][name] for name in inference_only_dataset_names]))
    datasets_dict["overlap"] = dict(zip(overlapping_dataset_names, [datasets_dict["all"][name] for name in overlapping_dataset_names]))

    if flags_dict["splits"] == "homologous-aware":
        
        datasets_dict["inference_only"] = remove_homologous_sequences_from_inference(datasets_dict, paths_dict["homology"])

    # Split data based on organisation
    splits_dict = handle_splits_flag(flags_dict["splits"], datasets_dict, paths_dict["homology"])
    
    splits_dict["inference"] = datasets_dict["inference_only"]
    splits_dict = concat_splits(splits_dict)

    # Load data into dataloaders
    dataloaders_dict = splits_to_loaders(
        splits_dict,
        batch_size = config["TRAINING_PARAMETERS"]["BATCH_SIZE"],
        n_workers = config["TRAINING_PARAMETERS"]["N_WORKERS"]
    )

    return dataloaders_dict

def handle_models(
    config: dict,
    downstream_models: list,
    embedding_size: int,
    dataloaders_dict: dict,
    output_features: list,
    activation_functions: list,
    loss_functions: list,
    optimisers: list,
    rnn_type: str,
    bidirectional: bool,
    results_path: str,
    device: str
    ):
    
    # Set up untrained model, criterion and optimiser
    model, criterion, optimiser = set_up_model(
        downstream_models[0],
        embedding_size,
        config["MODEL_ARCHITECTURE"]["HIDDEN_LAYERS"],
        config["MODEL_ARCHITECTURE"]["DROPOUT_LAYERS"],
        output_features,
        activation_functions,
        rnn_type,
        bidirectional,
        loss_functions,
        config["TRAINING_PARAMETERS"]["LEARNING_RATE"],
        config["MODEL_ARCHITECTURE"]["WEIGHT_DECAY"]
        )
    
    # Train model
    trained_model = handle_training_models(
        downstream_models[0],
        model,
        dataloaders_dict,
        output_features,
        criterion,
        optimiser,
        results_path,
        config["TRAINING_PARAMETERS"]["MIN_EPOCHS"],
        config["TRAINING_PARAMETERS"]["MAX_EPOCHS"],
        config["TRAINING_PARAMETERS"]["PATIENCE"],
        device
    )
    
    return trained_model, downstream_models, criterion

def handle_inference(config, downstream_models, model, dataloaders_dict, criterion, device, output_features, paths_dict):
    
    predictions_df = get_predictions(downstream_models, model, dataloaders_dict["test_inference"], criterion, device, output_features, paths_dict)
    metrics = compute_metrics(paths_dict["results"], output_features)
    
    return metrics

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
            dataloaders = handle_dataloading(config, dataset_dict, all_dataset_names, package_folder, homology_path, splits_flag)
            model, downstream_models, criterion = handle_models(
                config,
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
