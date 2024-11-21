# Standard modules
import math

# Third party modules
import pandas as pd
import torch

# Local modules
from config_loader import config
from visuals import plot_input_histogram, plot_predictions_vs_true
from helpers import get_results_path

def get_predictions(trained_model, inference_loader, criterion, device: str, package_folder, results_path):

    test_loss, predictions_df = test_model(trained_model, inference_loader, criterion, device, results_path)
    torch.save(trained_model.state_dict(), results_path / "downstream_model.pt")
    plot_input_histogram(predictions_df, results_path)
    plot_predictions_vs_true(predictions_df, results_path)

    return predictions_df

def test_model(model, test_loader, criterion, device: str, results_path):

    model.eval()
    test_loss = 0.0
    
    energy_predictions_list = []
    fitness_predictions_list = []
    energy_truths_list = []
    fitness_truths_list = []
    domains = []

    with torch.no_grad():
        
        for batch in test_loader:
            
            inputs = batch['sequence_representation'].float().to(device)
            
            energy_values = batch["energy_value"].float().to(device)
            fitness_values = batch['fitness_value'].float().to(device)
            
            energy_mask = batch["energy_mask"].bool().to(device)
            fitness_mask = batch["fitness_mask"].bool().to(device)

            outputs = model(inputs)

            if config["TRAINING_PARAMETERS"]["BATCH_SIZE"] > 1:
                
                energy_predictions = outputs[:, 0].squeeze()
                fitness_predictions = outputs[:, 1].squeeze()
                
            else:
                
                energy_predictions = outputs[:, 0]
                fitness_predictions = outputs[:, 1]
            
            energy_loss = 0
            fitness_loss = 0
            
            if energy_predictions.masked_select(energy_mask).nelement() > 0:
                
                energy_loss = criterion(energy_predictions.masked_select(energy_mask), energy_values.masked_select(energy_mask))
                
            if fitness_predictions.masked_select(fitness_mask).nelement() > 0:
                
                fitness_loss = criterion(fitness_predictions.masked_select(fitness_mask), fitness_values.masked_select(fitness_mask))
            
            loss = energy_loss + fitness_loss
            test_loss += loss.item()

            domains.extend(batch['domain_name'])
            
            for i in range(len(batch['domain_name'])):

                if energy_mask[i]:
                    
                    energy_truths_list.append(energy_values[i].item())
                    
                else:

                    energy_truths_list.append(math.nan)
                
                if energy_predictions.dim() != 0:
                
                    energy_predictions_list.append(energy_predictions[i].item())
                
                else:
                    
                    energy_predictions_list.append(math.nan)

                if fitness_mask[i]:
                    
                    fitness_truths_list.append(fitness_values[i].item())
                    
                else:
                    
                    fitness_truths_list.append(math.nan)
                
                if fitness_predictions.dim() != 0:
                
                    fitness_predictions_list.append(fitness_predictions[i].item())
                
                else:
                    
                    fitness_predictions_list.append(math.nan)

    # Calculate the average test loss
    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss}")

    results_df = pd.DataFrame({
        'Domain': domains,
        'Predicted Energy': energy_predictions_list,
        'True Energy': energy_truths_list,
        'Predicted Fitness': fitness_predictions_list,
        'True Fitness': fitness_truths_list
    })

    results_df.to_csv(results_path / "results.csv", index = False)

    return avg_test_loss, results_df
