# Standard modules
import math

# Third party modules
import pandas as pd
import torch

# Local modules
from config_loader import config

def get_predictions(trained_model, inference_loader, criterion, DEVICE: str, results_path: str):

    test_loss, predictions_df = test_model(trained_model, inference_loader, criterion, DEVICE)
    torch.save(trained_model.state_dict(), results_path)

    return predictions_df

def test_model(model, test_loader, criterion, device: str):

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

            # Store predictions and ground truths
            
            #print("Batch energy predictions: ", energy_predictions)
            #print("Batch fitness predictions: ", fitness_predictions)
            #print("Batch energy truths: ", energy_values)
            #print("Batch fitness truths: ", fitness_values)
            #print("Batch energy masks: ", energy_mask)
            #print("Batch fitness masks", fitness_mask)
            
            #energy_predictions_list.extend(energy_predictions.masked_select(energy_mask).tolist())
            #fitness_predictions_list.extend(fitness_predictions.masked_select(fitness_mask).tolist())

            #energy_truths_list.extend(energy_values.masked_select(energy_mask).tolist())
            #fitness_truths_list.extend(fitness_values.masked_select(fitness_mask).tolist())

            domains.extend(batch['domain_name'])  # Assuming domain names are part of the batch
            
            # Store predictions and ground truths with masking
            for i in range(len(batch['domain_name'])):
                # Energy
                if energy_mask[i]:
                    
                    energy_truths_list.append(energy_values[i].item())
                    
                else:

                    energy_truths_list.append(math.nan)
                
                energy_predictions_list.append(energy_predictions[i].item())

                # Fitness
                if fitness_mask[i]:
                    
                    fitness_truths_list.append(fitness_values[i].item())
                    
                else:
                    
                    fitness_truths_list.append(math.nan)
                
                fitness_predictions_list.append(fitness_predictions[i].item())

    # Calculate the average test loss
    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss}")

    # Create a pandas DataFrame with predictions and ground truth
    results_df = pd.DataFrame({
        'Domain': domains,
        'Predicted Energy': energy_predictions_list,
        'True Energy': energy_truths_list,
        'Predicted Fitness': fitness_predictions_list,
        'True Fitness': fitness_truths_list
    })

    # Save the results to a CSV file
    results_df.to_csv("results/test_results.csv", index=False)

    return avg_test_loss, results_df
