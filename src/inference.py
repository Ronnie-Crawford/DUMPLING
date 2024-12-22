# Standard modules
import math

# Third party modules
import pandas as pd
import torch

# Local modules
from config_loader import config
from visuals import plot_input_histogram, plot_predictions_vs_true
from helpers import get_results_path

def get_predictions(
    downstream_models: list,
    trained_model,
    test_inference_loader,
    criterion,
    device: str,
    output_features: list,
    paths_dict: dict
    ):

    
    match downstream_models[0]:
        
        case "FFNN":
            
            test_loss, predictions_df = run_inference_on_ffnn(trained_model, test_inference_loader, criterion, device, output_features, paths_dict["results"])
    
        case "LSTM_UNIDIRECTIONAL" | "LSTM_BIDIRECTIONAL" | "GRU_UNIDIRECTIONAL" | "GRU_BIDIRECTIONAL":

            test_loss, predictions_df = run_inference_on_rnn(trained_model, test_inference_loader, criterion, device, output_features, paths_dict["results"])
    
    torch.save(trained_model.state_dict(), paths_dict["results"] / "downstream_model.pt")
    plot_input_histogram(predictions_df, output_features, paths_dict["results"])
    plot_predictions_vs_true(predictions_df, output_features, paths_dict["results"])

    return predictions_df

def run_inference_on_ffnn(model, test_inference_loader, criterion, device: str, output_features: list, results_path):

    model.eval()
    test_inference_loss = 0.0
    
    predictions = {}
    truths = {}
    
    for output_feature in output_features:
        
        predictions[output_feature] = []
        truths[output_feature] = []
        
    domains = []

    with torch.no_grad():
        
        for batch in test_inference_loader:
            
            inputs = batch["sequence_representation"].float().to(device)
            
            temp_predictions = {}
            temp_truths = {}
            masks = {}
            
            for output_feature in output_features:
                
                temp_truths[output_feature] = batch[f"{output_feature}_value"].float().to(device)
                masks[output_feature] = batch[f"{output_feature}_mask"].bool().to(device)

            outputs = model(inputs)

            if config["TRAINING_PARAMETERS"]["BATCH_SIZE"] > 1:
                
                for index, output_feature in enumerate(output_features):
                
                    temp_predictions[output_feature] = outputs[:, index].squeeze()
                
            else:
                
                for index, output_feature in enumerate(output_features):
                
                    temp_predictions[output_feature] = outputs[:, index]
            
            losses = {}
            
            for output_feature in output_features:
                
                losses[output_feature] = 0
                
            for output_feature in output_features:
                
                if temp_predictions[output_feature].masked_select(masks[output_feature]).nelement() > 0:
                    
                    losses[output_feature] = criterion(
                        temp_predictions[output_feature].masked_select(masks[output_feature]),
                        temp_truths[output_feature].masked_select(masks[output_feature])
                        )
                
            
            loss = sum(losses.values())
            test_inference_loss += loss.item()

            domains.extend(batch["domain_name"])
            
            for i in range(len(batch["domain_name"])):

                for output_feature in output_features:
                    
                    if masks[output_feature][i]:
                        
                        truths[output_feature].append(temp_truths[output_feature][i].item())
                    
                    else:
                        
                        truths[output_feature].append(math.nan)
                    
                    if temp_predictions[output_feature].dim() != 0:
                        
                        predictions[output_feature].append(temp_predictions[output_feature][i].item())
                        
                    else:
                        
                        predictions[output_feature].append(math.nan)
                        
    # Calculate the average test loss
    average_test_inference_loss = test_inference_loss / len(test_inference_loader)
    print(f"Inference Loss: {average_test_inference_loss}")

    # Create results dataframe
    results_df = pd.DataFrame({
        "domain": domains,
    })
    
    for output_feature in output_features:
    
        results_df[f"{output_feature}_predictions"] = predictions[output_feature]
        results_df[f"{output_feature}_truth"] = truths[output_feature]

    results_df.to_csv(results_path / "results.csv", index = False)

    return average_test_inference_loss, results_df

def run_inference_on_rnn(model, test_inference_loader, criterion, device: str, output_features: list, results_path):
    
    model.eval()
    test_inference_loss = 0.0

    predictions = {feature: [] for feature in output_features}
    truths = {feature: [] for feature in output_features}
    domains = []

    with torch.no_grad():
        
        for batch in test_inference_loader:
            
            inputs = batch["sequence_representation"].float().to(device)
            lengths = batch["length"].to(device)

            temp_predictions = {}
            temp_truths = {}
            masks = {}

            for output_feature in output_features:
                
                temp_truths[output_feature] = batch[f"{output_feature}_value"].float().to(device)
                masks[output_feature] = batch[f"{output_feature}_mask"].bool().to(device)

            outputs = model(inputs, lengths)

            for index, output_feature in enumerate(output_features):
                
                temp_predictions[output_feature] = outputs[:, index]

            losses = {}
            
            for output_feature in output_features:
                
                if temp_predictions[output_feature].masked_select(masks[output_feature]).nelement() > 0:
                    
                    losses[output_feature] = criterion(
                        temp_predictions[output_feature].masked_select(masks[output_feature]),
                        temp_truths[output_feature].masked_select(masks[output_feature])
                    ).item()
                    
                else:
                    
                    losses[output_feature] = 0.0

            total_loss = sum(losses.values())
            test_inference_loss += total_loss

            domains.extend(batch.get("domain_name", []))

            batch_size = inputs.size(0)
            
            for i in range(batch_size):
                
                for output_feature in output_features:
                    
                    if masks[output_feature][i]:
                        
                        truths[output_feature].append(temp_truths[output_feature][i].item())
                        predictions[output_feature].append(temp_predictions[output_feature][i].item())
                        
                    else:
                        
                        truths[output_feature].append(math.nan)
                        predictions[output_feature].append(math.nan)

    # Calculate the average test loss
    average_test_inference_loss = test_inference_loss / len(test_inference_loader)
    print(f"Inference Loss: {average_test_inference_loss}")

    # Create results dataframe
    results_df = pd.DataFrame({
        "domain": domains,
    })

    for output_feature in output_features:
        
        results_df[f"{output_feature}_predictions"] = predictions[output_feature]
        results_df[f"{output_feature}_truth"] = truths[output_feature]

    results_df.to_csv(results_path / "results.csv", index=False)

    return average_test_inference_loss, results_df