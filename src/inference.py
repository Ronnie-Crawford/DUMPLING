# Third party modules
import pandas as pd
import torch

def get_predictions(trained_model, inference_loader, criterion, DEVICE: str, results_path: str):

    test_loss, predictions_df = test_model(trained_model, inference_loader, criterion, DEVICE)
    torch.save(trained_model.state_dict(), results_path)

    return predictions_df

def test_model(model, test_loader, criterion, device: str):

    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    predictions = []
    ground_truths = []
    domains = []  # To store domain names if needed

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['sequence_representation'].float().to(device)
            labels = batch['fitness_value'].float().to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            test_loss += loss.item()

            # Store predictions and ground truths
            predictions.extend(outputs.squeeze().tolist())
            ground_truths.extend(labels.tolist())
            domains.extend(batch['domain_name'])  # Assuming domain names are part of the batch

    # Calculate the average test loss
    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss}")

    # Create a pandas DataFrame with predictions and ground truth
    results_df = pd.DataFrame({
        'Domain': domains,
        'Predicted Fitness': predictions,
        'True Fitness': ground_truths
    })

    # Save the results to a CSV file
    results_df.to_csv("results/test_results.csv", index=False)

    return avg_test_loss, results_df
