# Third-party modules
import torch
import torch.nn as nn

# Local modules
from config_loader import config

class FitnessFindingFromPLMEmbeddings(nn.Module):

    """
    Simple FFNN as fitness-finding module for PLM embeddings.
    """

    def __init__(self, input_size = 320, hidden_layers = [128, 64, 32, 16], dropout_layers=[0.0, 0.0, 0.0, 0.0]):

        super(FitnessFindingFromPLMEmbeddings, self).__init__()
        layers = []
        in_size = input_size

        for index, hidden_size in enumerate(hidden_layers):

            layers.append(nn.Linear(in_size, hidden_size))          # Linear layer
            layers.append(nn.ReLU())                                # Activation function (ReLU)
            if index < len(dropout_layers) and dropout_layers[index] > 0:
                layers.append(nn.Dropout(p = dropout_layers[index]))    # Dropout layer
                
            in_size = hidden_size

        # Output layer for regression (single output for fitness value)
        layers.append(nn.Linear(in_size, 1))  # Output size is 1 for regression

        self.network = nn.Sequential(*layers)

    def forward(self, x):

        return self.network(x)

def set_up_model(input_size: int, hidden_layers: list, dropout_layers: list):

    model = FitnessFindingFromPLMEmbeddings(input_size, hidden_layers, dropout_layers)
    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr = config["TRAINING_PARAMETERS"]["LEARNING_RATE"])

    return model, criterion, optimiser
