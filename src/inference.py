# Standard modules
import math

# Third party modules
import pandas as pd
import torch
import numpy as np

# Local modules
from visuals import plot_input_histogram, plot_predictions_vs_true
from helpers import get_results_path
from results import compute_metrics, compute_domain_specific_metrics, save_results, save_overall_metrics, save_domain_specific_metrics
from visuals import plot_domain_specific_metrics

def handle_inference(
    batch_size: int,
    downstream_models,
    model,
    dataloaders_dict: dict,
    criterion,
    device,
    output_features, 
    results_path,
    ) -> tuple[pd.DataFrame, dict, dict]:
    
    predictions_df = get_predictions(downstream_models, model, dataloaders_dict["TEST"], criterion, device, output_features, results_path, batch_size)
    save_results(predictions_df, results_path)
    overall_metrics = compute_metrics(results_path, output_features)
    save_overall_metrics(overall_metrics, results_path)
    domain_specific_metrics = compute_domain_specific_metrics(results_path, output_features)
    save_domain_specific_metrics(domain_specific_metrics, results_path)

    plot_input_histogram(predictions_df, output_features, results_path)
    plot_predictions_vs_true(predictions_df, output_features, results_path)
    plot_domain_specific_metrics(domain_specific_metrics, results_path)
    
    return predictions_df, overall_metrics, domain_specific_metrics

def get_predictions(
    downstream_models: list,
    trained_model,
    test_loader,
    criterion,
    device: str,
    output_features: list,
    results_path,
    batch_size: int
    ):

    match downstream_models[0]:
        
        case "FFNN":
            
            test_loss, predictions_df = run_inference_on_ffnn(trained_model, test_loader, criterion, device, output_features, results_path, batch_size)
    
        case "LSTM_UNIDIRECTIONAL" | "LSTM_BIDIRECTIONAL" | "GRU_UNIDIRECTIONAL" | "GRU_BIDIRECTIONAL":

            test_loss, predictions_df = run_inference_on_rnn(trained_model, test_loader, criterion, device, output_features, results_path)

    return predictions_df

def run_inference_on_ffnn(
    model,
    test_loader,
    criterion,
    device: str,
    output_features: list,
    results_path,
    batch_size: int
):

    model.eval()
    total_loss = 0.0

    predictions_dict = {feature: [] for feature in output_features}
    truths_dict = {feature: [] for feature in output_features}

    domains = []
    sequences = []

    with torch.no_grad():

        for batch in test_loader:

            inputs = batch["sequence_embedding"].float().to(device)

            outputs = model(inputs)

            batch_losses = []

            for idx, feature in enumerate(output_features):

                truths = batch[f"{feature}_value"].float().to(device)
                masks = batch[f"{feature}_mask"].bool().to(device)

                preds = outputs[:, idx]

                # Mask and compute loss
                valid_preds = preds[masks]
                valid_truths = truths[masks]

                if valid_preds.nelement() > 0:
                    
                    feature_loss = criterion(valid_preds, valid_truths)
                    batch_losses.append(feature_loss.item())
                    
                else:
                    
                    batch_losses.append(0.0)

                # Store predictions and truths (masked)
                cpu_preds = preds.cpu().numpy()
                cpu_truths = truths.cpu().numpy()
                cpu_masks = masks.cpu().numpy()

                predictions_dict[feature].append(np.where(cpu_masks, cpu_preds, np.nan))
                truths_dict[feature].append(np.where(cpu_masks, cpu_truths, np.nan))

            total_loss += sum(batch_losses)

            # Collect batch metadata
            domains.extend(batch["domain_name"])
            sequences.extend(batch["aa_seq"])

    # Average loss
    average_test_loss = total_loss / len(test_loader)
    print(f"Inference Loss: {average_test_loss}")

    # Concatenate results from all batches
    final_results = {
        "domain": domains,
        "sequences": sequences
    }

    for feature in output_features:
        
        final_results[f"{feature}_predictions"] = np.concatenate(predictions_dict[feature])
        final_results[f"{feature}_truth"] = np.concatenate(truths_dict[feature])

    results_df = pd.DataFrame(final_results)

    return average_test_loss, results_df

def old_run_inference_on_ffnn(model, test_loader, criterion, device: str, output_features: list, results_path, batch_size: int):

    model.eval()
    test_loss = 0.0
    
    predictions = {}
    truths = {}
    
    for output_feature in output_features:
        
        predictions[output_feature] = []
        truths[output_feature] = []
        
    domains = []
    sequences = []

    with torch.no_grad():
        
        for batch in test_loader:
            
            inputs = batch["sequence_embedding"].float().to(device)
            
            temp_predictions = {}
            temp_truths = {}
            masks = {}
            
            for output_feature in output_features:
                
                temp_truths[output_feature] = batch[f"{output_feature}_value"].float().to(device)
                masks[output_feature] = batch[f"{output_feature}_mask"].bool().to(device)

            outputs = model(inputs)

            if batch_size > 1:
                
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
            test_loss += loss.item()
            domains.extend(batch["domain_name"])
            sequences.extend(batch["aa_seq"])
            
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
    average_test_loss = test_loss / len(test_loader)
    print(f"Inference Loss: {average_test_loss}")

    # Create results dataframe
    results_df = pd.DataFrame({
        "domain": domains,
        "sequences": sequences
    })
    
    for output_feature in output_features:
    
        results_df[f"{output_feature}_predictions"] = predictions[output_feature]
        results_df[f"{output_feature}_truth"] = truths[output_feature]

    return average_test_loss, results_df

def run_inference_on_rnn(model, test_loader, criterion, device: str, output_features: list, results_path):
    
    model.eval()
    test_loss = 0.0

    predictions = {feature: [] for feature in output_features}
    truths = {feature: [] for feature in output_features}
    domains = []
    sequences = []

    with torch.no_grad():
        
        for batch in test_loader:
            
            inputs = batch["sequence_embedding"].float().to(device)
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
            test_loss += total_loss

            domains.extend(batch.get("domain_name", []))
            sequences.extend(batch.get("aa_seqs", []))

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
    average_test_loss = test_loss / len(test_loader)
    print(f"Inference Loss: {average_test_loss}")

    # Create results dataframe
    results_df = pd.DataFrame({
        "domain": domains,
        "sequences": sequences
    })

    for output_feature in output_features:
        
        results_df[f"{output_feature}_predictions"] = predictions[output_feature]
        results_df[f"{output_feature}_truth"] = truths[output_feature]

    results_df.to_csv(results_path / "results.csv", index=False)

    return average_test_loss, results_df