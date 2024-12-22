# Standard modules
import random
import copy

# Third-party modules
import numpy as np
import pandas as pd
import torch

# Local modules
from config_loader import config
from models import set_up_model
from inference import get_predictions
from results import compute_metrics

def optimise_hyperparameters(device, training_loader, validation_loader, testing_loader, search, embedding_size):

    results = []
    best_layers = None
    best_dropout = None
    best_pearson = 0

    if search == "grid-search":

        for hidden_layers, dropout_layers, metrics in grid_search(device, training_loader, validation_loader, testing_loader, embedding_size):

            if metrics["Pearson"] > best_pearson:

                best_layers = hidden_layers
                best_dropout = dropout_layers
                best_pearson = metrics["Pearson"]

                print(f"Found new best hidden layer configuration: hidden layers: {hidden_layers}\n dropout layers: {dropout_layers}, giving Pearson correlation of {metrics['Pearson']}.")

            result = {
                        'Layers': hidden_layers,
                        "Dropouts": dropout_layers,
                        **metrics
            }
            results.append(result)

            print(f"Completed configuration: hidden layers: {hidden_layers} dropout layers: {dropout_layers}.")

    elif search == "random-search":

        index = 1

        for hidden_layers, dropout_layers, metrics in random_search(device, training_loader, validation_loader, testing_loader, embedding_size):

            print(f"Attempting configuration [{index}/{config['HYPERPARAMETER_TUNING']['SEARCH_ITERATIONS']}:")
            print(f"hidden layers: {hidden_layers}")
            print(f"dropout layers: {dropout_layers}")

            if metrics["Pearson"] > best_pearson:

                best_layers = hidden_layers
                best_dropout = dropout_layers
                best_pearson = metrics["Pearson"]

            result = {
                        'Layers': hidden_layers,
                        "Dropouts": dropout_layers,
                        **metrics
            }
            results.append(result)

            print(f"Current best configuration:")
            print(f"hidden layers: {best_layers}")
            print(f"dropout layers: {best_dropout}")
            print(f"Pearson correlation of {best_pearson}")

    print(f"Best configuration: {best_layers}, giving Pearson correlation of {best_pearson}.")
    df = pd.DataFrame(results)
    df.to_csv('grid_search_results.tsv', sep='\t', index=False)

    return df

def grid_search(device: str, training_loader, validation_loader, testing_loader, embedding_size: int):

    for i in range(2, 10):

        for j in range(0, 10):

            for k in range(0, 10):

                for l in range(0, 10):

                    hidden_layers = [i, j, k, l]
                    hidden_layers = [layer for layer in hidden_layers if (layer != 0) and (layer != 1)]
                    model, criterion, optimiser = set_up_model(320, hidden_layers)
                    trained_model = train_fitness_finder_from_plm_embeddings_nn(model, training_loader, validation_loader, criterion, optimiser, config["TRAINING_PARAMETERS"]["MAX_EPOCHS"], config["TRAINING_PARAMETERS"]["PATIENCE"], DEVICE)

                    predictions_df = get_predictions(trained_model, testing_loader, criterion, DEVICE, "models/plm_embedding_to_simple_nn")

                    yield hidden_layers, compute_metrics("results/test_results.csv")

def random_search(
    search_iterations: int,
    min_batch_size: int,
    max_batch_size: int,
    min_hidden_layers: int,
    max_hidden_layers: int,
    min_hidden_size: int,
    max_hidden_size: int,
    min_dropout: float,
    max_dropout: float,
    min_learning_rate: float,
    max_learning_rate: float,
    min_weight_decay: float,
    max_weight_decay: float,
    loss_functions: list,
    activation_functions: list,
    ):

    random_configs_list = []
    
    for search_index in range(search_iterations):

        new_config = copy.deepcopy(config)

        # Choose random embedding model from models set to true in original config
        embedding_models = [key for key, value in config.get("EMBEDDING_MODELS", {}).items() if value]
        new_config["EMBEDDING_MODELS"] = {key: False for key in config["EMBEDDING_MODELS"].keys()}
        new_config["EMBEDDING_MODELS"][random.choice(embedding_models)] = True
        
        new_config["NORMALISE_EMBEDDINGS"] = random.choice([True, False])

        # Choose random downstream model from models set to true in original config
        downstream_models = [key for key, value in config.get("DOWNSTREAM_MODELS", {}).items() if value]
        new_config["DOWNSTREAM_MODELS"] = {key: False for key in config["DOWNSTREAM_MODELS"].keys()}
        new_config["DOWNSTREAM_MODELS"][random.choice(downstream_models)] = True
        
        # Scalar values set to random value between min and max
        new_config["TRAINING_PARAMETERS"]["BATCH_SIZE"] = int(np.random.randint(min_batch_size, max_batch_size))
        n_hidden_layers = int(np.random.randint(min_hidden_layers, max_hidden_layers))
        hidden_layer_sizes = np.random.randint(min_hidden_size, max_hidden_size, size=n_hidden_layers).tolist()
        new_config["MODEL_ARCHITECTURE"]["HIDDEN_LAYERS"] = [int(size) for size in hidden_layer_sizes]
        dropout_layers = np.random.uniform(min_dropout, max_dropout, size=n_hidden_layers)
        # Randomly set some dropout rates to 0.0
        num_zero_dropouts = np.random.randint(0, n_hidden_layers)
        zero_indices = np.random.choice(n_hidden_layers, size=num_zero_dropouts, replace=False)
        dropout_layers[zero_indices] = 0.0
        new_config["MODEL_ARCHITECTURE"]["DROPOUT_LAYERS"] = dropout_layers.tolist()
        new_config["TRAINING_PARAMETERS"]["LEARNING_RATE"] = float(np.random.uniform(min_learning_rate, max_learning_rate))
        new_config["MODEL_ARCHITECTURE"]["WEIGHT_DECAY"] = float(np.random.uniform(min_weight_decay, max_weight_decay))
        
        # Choose random loss function from functions set to true in original config
        loss_functions = [key for key, value in config["MODEL_ARCHITECTURE"].get("LOSS_FUNCTIONS", {}).items() if value]
        new_config["MODEL_ARCHITECTURE"]["LOSS_FUNCTIONS"] = {key: False for key in config["MODEL_ARCHITECTURE"]["LOSS_FUNCTIONS"].keys()}
        new_config["MODEL_ARCHITECTURE"]["LOSS_FUNCTIONS"][random.choice(loss_functions)] = True
        
        # Choose random activation function from functions set to true in original config
        activation_functions = [key for key, value in config["MODEL_ARCHITECTURE"].get("ACTIVATION_FUNCTIONS", {}).items() if value]
        new_config["MODEL_ARCHITECTURE"]["ACTIVATION_FUNCTIONS"] = {key: False for key in config["MODEL_ARCHITECTURE"]["ACTIVATION_FUNCTIONS"].keys()}
        new_config["MODEL_ARCHITECTURE"]["ACTIVATION_FUNCTIONS"][random.choice(activation_functions)] = True
        
        random_configs_list.append(new_config)
    
    return random_configs_list

def gradient_descent(device: str, training_loader, validation_loader, testing_loader, embedding_size: int):

    search_iterations = config["HYPERPARAMETER_TUNING"]["SEARCH_ITERATIONS"]
    learning_rate = config["TRAINING_PARAMETERS"]["LEARNING_RATE"]
    min_layers = config["HYPERPARAMETER_TUNING"]["MIN_HIDDEN_LAYERS"]
    max_layers = config["HYPERPARAMETER_TUNING"]["MAX_HIDDEN_LAYERS"]
    min_size = config["HYPERPARAMETER_TUNING"]["MIN_HIDDEN_LAYER_SIZE"]
    max_size = config["HYPERPARAMETER_TUNING"]["MAX_HIDDEN_LAYER_SIZE"]
    min_dropout = config["HYPERPARAMETER_TUNING"]["MIN_DROPOUT"]
    max_dropout = config["HYPERPARAMETER_TUNING"]["MAX_DROPOUT"]

    for n_layers in range(min_layers + 1, max_layers + 1):

        layer_sizes = torch.FloatTensor(n_layers).uniform_(np.log(min_size), np.log(max_size)).requires_grad_(True)
        dropout_rates = torch.FloatTensor(n_layers).uniform_(min_dropout, max_dropout).requires_grad_(True)

        for search in search_iterations:

            epsilon = 1e-4
            layer_sizes_plus = layer_sizes + epsilon
            layer_sizes_minus = layer_sizes - epsilon
            dropout_rates_plus = torch.clamp(dropout_rates + epsilon, min_dropout, max_dropout)
            dropout_rates_minus = torch.clamp(dropout_rates - epsilon, min_dropout, max_dropout)

            hidden_layers = torch.exp(layer_sizes).round().int().tolist()
            dropout_layers = dropout_rates.tolist()

            model, criterion, optimiser = set_up_model(embedding_size, hidden_layers, dropout_layers)
            trained_model = train_fitness_finder_from_plm_embeddings_nn(model, training_loader, validation_loader, criterion, optimiser, config["TRAINING_PARAMETERS"]["MAX_EPOCHS"], config["TRAINING_PARAMETERS"]["PATIENCE"], device)

            predictions_df = get_predictions(trained_model, testing_loader, criterion, device, "models/plm_embedding_to_simple_nn")

            yield hidden_layers, dropout_layers, compute_metrics("results/test_results.csv")
