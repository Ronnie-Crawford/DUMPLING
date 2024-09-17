# Third-party modules
import numpy as np
import pandas as pd
import torch

# Local modules
from config_loader import config
from models import set_up_model
from training import train_fitness_finder_from_plm_embeddings_nn
from inference import get_predictions
from helpers import compute_metrics

def optimise_hyperparameters(DEVICE, training_loader, validation_loader, testing_loader, search, embedding_size):

    results = []
    best_layers = None
    best_dropout = None
    best_pearson = 0

    if search == "grid-search":

        for hidden_layers, dropout_layers, metrics in grid_search(DEVICE, training_loader, validation_loader, testing_loader, embedding_size):

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

        for hidden_layers, dropout_layers, metrics in random_search(DEVICE, training_loader, validation_loader, testing_loader, embedding_size):

            print(f"Attempting configuration [{index}/{config["HYPERPARAMETER_TUNING"]["SEARCH_ITERATIONS"]}:")
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

def grid_search(DEVICE, training_loader, validation_loader, testing_loader, embedding_size):

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

def random_search(DEVICE: str, training_loader, validation_loader, testing_loader, embedding_size: int):

    n_arrays = config["HYPERPARAMETER_TUNING"]["SEARCH_ITERATIONS"]
    min_layers = config["HYPERPARAMETER_TUNING"]["MIN_HIDDEN_LAYERS"]
    max_layers = config["HYPERPARAMETER_TUNING"]["MAX_HIDDEN_LAYERS"]
    min_size = config["HYPERPARAMETER_TUNING"]["MIN_HIDDEN_LAYER_SIZE"]
    max_size = config["HYPERPARAMETER_TUNING"]["MAX_HIDDEN_LAYER_SIZE"]
    min_dropout = config["HYPERPARAMETER_TUNING"]["MIN_DROPOUT"]
    max_dropout = config["HYPERPARAMETER_TUNING"]["MAX_DROPOUT"]

    hidden_layer_arrays = []
    dropout_layer_arrays = []

    for _ in range(n_arrays):

        n_layers = np.random.randint(min_layers, max_layers)
        random_hidden_layers = np.random.randint(min_size, max_size, size=n_layers)
        random_dropout_layers = np.random.uniform(min_dropout, max_dropout, size=n_layers)
        random_dropout_layers[np.random.randint(0, n_layers, size=n_layers // 2)] = 0.0

        hidden_layer_arrays.append(random_hidden_layers)
        dropout_layer_arrays.append(random_dropout_layers)

    for hidden_layers, dropout_layers in zip(hidden_layer_arrays, dropout_layer_arrays):

        model, criterion, optimiser = set_up_model(embedding_size, hidden_layers, dropout_layers)
        trained_model = train_fitness_finder_from_plm_embeddings_nn(model, training_loader, validation_loader, criterion, optimiser, config["TRAINING_PARAMETERS"]["MAX_EPOCHS"], config["TRAINING_PARAMETERS"]["PATIENCE"], DEVICE)

        predictions_df = get_predictions(trained_model, testing_loader, criterion, DEVICE, "models/plm_embedding_to_simple_nn")
        yield hidden_layers, dropout_layers, compute_metrics("results/test_results.csv")

def gradient_descent(DEVICE: str, training_loader, validation_loader, testing_loader, embedding_size: int):

    search_iterations = config["HYPERPARAMETER_TUNING"]["SEARCH_ITERATIONS"]
    learning_rate = config["TRAINING_PARAMETERS"]["LEARNING_RATE"]
    min_layers = config["HYPERPARAMETER_TUNING"]["MIN_HIDDEN_LAYERS"]
    max_layers = config["HYPERPARAMETER_TUNING"]["MAX_HIDDEN_LAYERS"]
    min_size = config["HYPERPARAMETER_TUNING"]["MIN_HIDDEN_LAYER_SIZE"]
    max_size = config["HYPERPARAMETER_TUNING"]["MAX_HIDDEN_LAYER_SIZE"]
    min_dropout = config["HYPERPARAMETER_TUNING"]["MIN_DROPOUT"]
    max_dropout = config["HYPERPARAMETER_TUNING"]["MAX_DROPOUT"]

    for n_layers in range(min_layers + 1, max_layers + 1):

        ### We should set random values for the size and dropout of each layer based on the min max here ###
        layer_sizes = torch.FloatTensor(n_layers).uniform_(np.log(min_size), np.log(max_size)).requires_grad_(True)
        dropout_rates = torch.FloatTensor(n_layers).uniform_(min_dropout, max_dropout).requires_grad_(True)

        for search in search_iterations:

            ### We should test several similar arrays at once so that a gradient can be worked out ###
            epsilon = 1e-4
            layer_sizes_plus = layer_sizes + epsilon
            layer_sizes_minus = layer_sizes - epsilon
            dropout_rates_plus = torch.clamp(dropout_rates + epsilon, min_dropout, max_dropout)
            dropout_rates_minus = torch.clamp(dropout_rates - epsilon, min_dropout, max_dropout)

            hidden_layers = torch.exp(layer_sizes).round().int().tolist()
            dropout_layers = dropout_rates.tolist()


            ### This part runs the model ###
            model, criterion, optimiser = set_up_model(embedding_size, hidden_layers, dropout_layers)
            trained_model = train_fitness_finder_from_plm_embeddings_nn(model, training_loader, validation_loader, criterion, optimiser, config["TRAINING_PARAMETERS"]["MAX_EPOCHS"], config["TRAINING_PARAMETERS"]["PATIENCE"], DEVICE)

            predictions_df = get_predictions(trained_model, testing_loader, criterion, DEVICE, "models/plm_embedding_to_simple_nn")

            ### Then we can work out the gradient, and decide which values to try next here ####

            yield hidden_layers, dropout_layers, compute_metrics("results/test_results.csv")
