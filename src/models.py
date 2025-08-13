# Third-party modules
import torch
import torch.nn as nn
import torch.nn.functional as F

class FFNN(nn.Module):

    """
    Simple FFNN for regression from arbitrary input vector to flexible number of output features.
    """

    def __init__(
        self,
        input_size: int,
        output_features: list,
        hidden_layers: list,
        head_layers: dict,
        dropout_layers: list,
        activation_functions: list
        ):
        
        # Set up architecture of model, iterate through layers
        super(FFNN, self).__init__()
        layers = []

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

def handle_models(
    dataloaders_dict: dict,
    downstream_models: list,
    embedding_size: int,
    predicted_features: list,
    hidden_layers: list,
    dropout_layers: list,
    learning_rate: float,
    weight_decay: float,
    activation_functions: list,
    loss_functions_dict: dict,
    optimisers: list,
    results_path: str,
    device: str
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
    dropout_layers,
    activation_functions
    ):
    
    match downsteam_model_choice:
    
        case "FFNN":
                
            model = FFNN(
                input_size,
                predicted_features,
                hidden_layers,
                dropout_layers,
                activation_functions
                )
    
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
    
class ReliabilityBalancing(nn.Module):
    
    def __init__(self, base_loss, sigma_default = 0.05):
        
        super().__init__()
        self.base_loss = base_loss
        
    def forward(self, predictions, targets, sigmas):
        
        regularised_loss = self.base_loss(predictions, targets)
        
        reliability_loss = regularised_loss * (sigmas / sigma_default)
        
        return reliability_loss

def get_loss_function(loss_function_choice):
    
    match loss_function_choice:
        
        case "MSELOSS":
            
            criterion = nn.MSELoss()
        
        case "MAELOSS":
            
            criterion = nn.L1Loss()
        
        case "MODE-PULL":
            
            criterion = ModePullLoss(base_loss = nn.MSELoss(), lambda_reg = 0.1)
        
        case "RELIABILITY-BALANCE":
            
            criterion = ReliabilityBalancing(base_loss = nn.MSELoss(), sigma_default = 0.05)
          
    return criterion