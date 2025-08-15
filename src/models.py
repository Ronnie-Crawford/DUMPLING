# Third-party modules
from pathlib import Path
import torch
from torch.cuda import device
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.transformer import _get_activation_fn

class FFNN(nn.Module):

    """
    Simple FFNN for regression from arbitrary input vector to flexible number of output features.
    """

    def __init__(
        self,
        input_size: int,
        output_features: list,
        trunk_hidden_layers: list,
        head_hidden_layers: list,
        dropout_layers: list,
        activation_functions: list
        ):

        super(FFNN, self).__init__()

        # Set up architecture of trunk, iterate through layers
        layers = []

        for index, hidden_size in enumerate(trunk_hidden_layers):

            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(get_activation_function(activation_functions[0]))

            if index < len(dropout_layers) and dropout_layers[index] > 0:

                layers.append(nn.Dropout(p = dropout_layers[index]))

            input_size = hidden_size

        self.trunk = nn.Sequential(*layers)

        # Add heads
        self.heads = nn.ModuleDict()

        for feature in output_features:

            self.heads[feature] = self._make_head(input_size, head_hidden_layers, activation_functions)

    def _make_head(self, input_size, head_hidden_layers, activation_functions):

        layers = []

        for hidden_size in head_hidden_layers:

            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(get_activation_function(activation_functions[0]))
            input_size = hidden_size

        layers.append(nn.Linear(input_size, 1)) # Final output of the head

        return nn.Sequential(*layers)

    def forward(self, x):

        trunk_output = self.trunk(x)
        outputs = []

        for feature, head in self.heads.items():

            output = head(trunk_output)
            outputs.append(output)

        return torch.cat(outputs, dim = 1)

def handle_models(
    dataloaders_dict: dict,
    downstream_models: list,
    embedding_size: int,
    predicted_features: list,
    hidden_layers: list,
    head_layers: list,
    dropout_layers: list,
    learning_rate: float,
    weight_decay: float,
    activation_functions: list,
    loss_functions_dict: dict,
    optimisers: list,
    results_path: Path,
    device: torch.device
    ):

    """
    Sets up model, optimser, and loss function.
    Currently can only set up one model, the first in the down stream models list,
    in the future maybe we could have multiple as an ensemble architecture.
    """

    model = get_model(
        downstream_models[0],
        embedding_size,
        predicted_features,
        hidden_layers,
        head_layers,
        dropout_layers,
        activation_functions
        )
    optimiser = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)

    # Change loss functions dict (selections) into criterion dict (instantiated objects)
    criterion_dict = {
        feature: get_loss_function(loss_functions_dict[feature])
        for feature in predicted_features
        }

    return model, criterion_dict, optimiser

def get_model(
    downsteam_model_choice,
    input_size,
    predicted_features,
    hidden_layers,
    head_layers,
    dropout_layers,
    activation_functions
    ):

    model = None

    match downsteam_model_choice:

        case "FFNN":

            model = FFNN(
                input_size,
                predicted_features,
                hidden_layers,
                head_layers,
                dropout_layers,
                activation_functions
                )

        case _:

            raise ValueError(f"Unknown model_selection: {downsteam_model_choice}")

    return model

class ModePullLoss(nn.Module):

    def __init__(self, base_loss, lambda_reg = 0.1):

        super().__init__()
        self.base = base_loss
        self.lam = lambda_reg

    def forward(self, preds, targets):

        # standard regression loss:
        reg_loss = self.base(preds, targets)

        # “pull‐to‐0 or –1” penalty:
        dist_to_zero = preds.pow(2)
        dist_to_neg1 = (preds + 1).pow(2)
        pull_penalty = torch.min(dist_to_zero, dist_to_neg1).mean()

        return reg_loss + self.lam * pull_penalty

def get_activation_function(activation_function_choice):

    match activation_function_choice:

        case "RELU": return nn.ReLU()
        case "LEAKYRELU": return nn.LeakyReLU(negative_slope = 0.01)
        case "ELU": return nn.ELU()
        case "GELU": return nn.GELU()
        case "SELU": return nn.SELU()
        case "_": raise ValueError("Did not recognise activation function choice: ", activation_function_choice)

def get_loss_function(loss_function_choice):

    criterion = None

    match loss_function_choice:

        case "MSELOSS":

            criterion = nn.MSELoss()

        case "MAELOSS":

            criterion = nn.L1Loss()

        case "MODE-PULL":

            criterion = ModePullLoss(base_loss = nn.MSELoss(), lambda_reg = 0.1)

    return criterion
