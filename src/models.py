# Third-party modules
import torch
import torch.nn as nn

# Local modules
from config_loader import config

class EnergyFitnessFindingFromPLMEmbeddings(nn.Module):

    """
    Simple FFNN as energy and fitness-finding module for PLM embeddings.
    """

    def __init__(self, input_size, hidden_layers = [], dropout_layers=[], activation_function = "RELU"):

        super(EnergyFitnessFindingFromPLMEmbeddings, self).__init__()
        layers = []
        in_size = input_size

        for index, hidden_size in enumerate(hidden_layers):

            layers.append(nn.Linear(in_size, hidden_size))
            
            
            match activation_function:
                
                case "RELU":
                    
                    layers.append(nn.ReLU())
                
                case "LEAKYRELU":
                    
                    layers.append(nn.LeakyReLU(negative_slope = 0.01))
                    
                case "ELU":
                    
                    layers.append(nn.ELU())
                    
                case "GELU":
                    
                    layers.append(nn.GELU())
                    
                case "SELU":
                    
                    layers.append(nn.SELU())
            
            if index < len(dropout_layers) and dropout_layers[index] > 0:
                
                layers.append(nn.Dropout(p = dropout_layers[index]))
                
            in_size = hidden_size

        # Output layer for regression (2 outputs, one for energy, one for fitness value)
        layers.append(nn.Linear(in_size, 2))

        self.network = nn.Sequential(*layers)

    def forward(self, x):

        return self.network(x)

class FitnessFindingFromPLMEmbeddings(nn.Module):

    """
    Simple FFNN as fitness-finding module for PLM embeddings.
    """

    def __init__(self, input_size = 320, hidden_layers = [128, 64, 32, 16], dropout_layers=[0.0, 0.0, 0.0, 0.0]):

        super(FitnessFindingFromPLMEmbeddings, self).__init__()
        layers = []
        in_size = input_size

        for index, hidden_size in enumerate(hidden_layers):

            layers.append(nn.Linear(in_size, hidden_size))                  # Linear layer
            layers.append(nn.ReLU())                                        # Activation function (ReLU)
            if index < len(dropout_layers) and dropout_layers[index] > 0:
                layers.append(nn.Dropout(p = dropout_layers[index]))        # Dropout layer
                
            in_size = hidden_size

        # Output layer for regression (single output for fitness value)
        layers.append(nn.Linear(in_size, 1))  # Output size is 1 for regression

        self.network = nn.Sequential(*layers)

    def forward(self, x):

        return self.network(x)

def set_up_model(input_size: int, hidden_layers: list, dropout_layers: list, activation_function: str, learning_rate: float, weight_decay: float):

    #model = FitnessFindingFromPLMEmbeddings(input_size, hidden_layers, dropout_layers)
    model = EnergyFitnessFindingFromPLMEmbeddings(input_size, hidden_layers, dropout_layers, activation_function)
    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)

    return model, criterion, optimiser
