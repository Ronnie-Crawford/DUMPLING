# Standard modules
import random

# Third-party modules
import torch
import torch.nn as nn

# Local modules
from config_loader import config

class FFNN(nn.Module):

    """
    Simple FFNN for regression from arbitrary input vector to flexible number of output features.
    """

    def __init__(
        self,
        input_size: int,
        output_features: list,
        hidden_layers: list,
        dropout_layers: list,
        activation_functions: list
        ):

        super(FFNN, self).__init__()
        layers = []

        print("Using activation function: ", activation_functions[0])

        for index, hidden_size in enumerate(hidden_layers):

            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            
            match activation_functions[0]:
                
                case "RELU": layers.append(nn.ReLU())
                case "LEAKYRELU": layers.append(nn.LeakyReLU(negative_slope = 0.01))    
                case "ELU": layers.append(nn.ELU())  
                case "GELU": layers.append(nn.GELU()) 
                case "SELU": layers.append(nn.SELU())
            
            if index < len(dropout_layers) and dropout_layers[index] > 0:
                
                layers.append(nn.Dropout(p = dropout_layers[index]))
                
            input_size = hidden_size

        layers.append(nn.Linear(input_size, len(output_features)))

        self.network = nn.Sequential(*layers)

    def forward(self, x):

        return self.network(x)

class RNN(nn.Module):
    
    """
    A flexible Recurrent Neural Network for processing sequences of embeddings.
    """
    
    def __init__(
        self,
        input_size: int,
        output_features: list,
        hidden_layers: list,
        dropout_layers: list,
        activation_functions: str,
        rnn_type: str,
        bidirectional: bool
    ):
        
        super(RNN, self).__init__()
        
        self.hidden_size = hidden_layers[0]
        self.bidirectional = bidirectional
        
        # Decide type of RNN
        if rnn_type == "LSTM":
            
            self.rnn = nn.LSTM(
                input_size, 
                hidden_layers[0], 
                len(hidden_layers),
                batch_first = True, 
                bidirectional = bidirectional,
                dropout = dropout_layers[0] if len(hidden_layers) > 0 else 0
            )
            
        elif rnn_type == "GRU":
            
            self.rnn = nn.GRU(
                input_size, 
                hidden_layers[0], 
                len(hidden_layers), 
                batch_first = True, 
                bidirectional = bidirectional,
                dropout = dropout_layers[0] if len(hidden_layers) > 0 else 0
            )
            
        else:
            
            raise ValueError("Invalid rnn_type. Expected 'LSTM' or 'GRU'")
        
        # Decide activation function and model size
        self.activation = None
        
        match activation_functions[0]:
                
            case "RELU": self.activation = nn.ReLU()
            case "LEAKYRELU": self.activation = nn.LeakyReLU(negative_slope = 0.01)    
            case "ELU": self.activation = nn.ELU()
            case "GELU": self.activation = nn.GELU()
            case "SELU": self.activation = nn.SELU()
        
        n_directions = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_layers[0] * n_directions, len(output_features))
    
    def forward(self, x, lengths):
        
        """
        x: Padded sequences of shape (batch_size, seq_length, input_size)
        lengths: Lengths of the sequences before padding
        """
        
        # Pack the padded batch of sequences
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first = True, enforce_sorted = False
        )
        
        # Pass through the RNN
        packed_output, _ = self.rnn(packed_input)
        
        # Unpack the sequences
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first = True
        )
        
        # Obtain the outputs from the last time step of each sequence
        # For bidirectional RNNs concatenate the forward and backward outputs
        if self.bidirectional:
            
            # Concatenate the outputs from the last time steps of both directions
            forward_output = output[range(len(output)), lengths - 1, : self.hidden_size]
            backward_output = output[:, 0, self.hidden_size:]
            last_output = torch.cat((forward_output, backward_output), dim = 1)
            
        else:
            
            # Get the outputs from the last time step
            last_output = output[range(len(output)), lengths - 1, :]
        
        # Apply activation function
        last_output = self.activation(last_output)
        
        # Pass through the fully connected layer
        out = self.fc(last_output)
        
        return out

def set_up_model(
    downstream_model: str,
    input_size: int,
    hidden_layers: list,
    dropout_layers: list,
    output_features: list,
    activation_functions: list,
    rnn_type: str,
    bidirectional: bool,
    loss_functions: list,
    learning_rate: float,
    weight_decay: float
    ):

    print("MODEL: ", downstream_model)

    match downstream_model:
        
        # Create a linear regression model - not fully implimented
        case "LINEAR_REGRESSION":
            
            model = nn.Linear(input_size, output_features)
        
        # Create a feedforward neural network
        case "FFNN":
            
            model = FFNN(
                input_size,
                output_features,
                hidden_layers,
                dropout_layers,
                activation_functions
                )
        
        # Create a recurrent neural network
        case "LSTM_UNIDIRECTIONAL" | "LSTM_BIDIRECTIONAL" | "GRU_UNIDIRECTIONAL" | "GRU_BIDIRECTIONAL":
            
            model = RNN(
                input_size,
                output_features,
                hidden_layers,
                dropout_layers,
                activation_functions,
                rnn_type,
                bidirectional
                )

    optimiser = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)
    criterion = None
    
    match loss_functions[0]:
        
        case "MSELOSS":
            
            criterion = nn.MSELoss()
            print("Using MSE loss function.")
        
        case "MAELOSS":
            
            criterion = nn.L1Loss()
            print("Using MAE loss function.")
        
        case "HUBERLOSS":

            criterion = nn.SmoothL1Loss(),
            print("Using Huber loss function.")

    return model, criterion, optimiser
