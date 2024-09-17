# Third-party modules
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

def get_device(device: str):

    """
    Determines the best available device (GPU, MPS, or CPU).

    Returns:

        - torch.device: The best available device.
    """

    if device == None:

        if torch.cuda.is_available():

            print("Using device: CUDA.")
            return torch.device("cuda")

        elif torch.backends.mps.is_available():

            print("Using device: MPS.")
            return torch.device("mps")

        else:

            print("Using device: CPU.")
            return torch.device("cpu")

    else:

        device = device.lower()
        print(f"Manual overide, using device: {device.upper()}")
        return torch.device(device)

def truncate_domain(domain_name, domain_name_splitter):

    return domain_name.rsplit(domain_name_splitter, 1)[0] + ".pdb"

def is_valid_sequence(sequence: str, valid_alphabet: str) -> bool:

        """
        Check if the sequence contains only characters from the valid alphabet.

        Parameters:
            - sequence (str): The amino acid sequence to check.
            - valid_alphabet (str): A string of valid amino acids.

        Returns:
            - bool: True if the sequence is valid, False otherwise.
        """

        return all(residue in valid_alphabet for residue in sequence)

def is_floatable(value):

    """
    Helper method to check if a value can be converted to a float.
    """

    try:

        float(value)

        return True

    except ValueError:

        return False

def get_family_size(family, family_dict: dict):

        family_domains = family_dict[family]
        family_mask = [domain in family_domains for domain in dataset.domain_names]
        return sum(family_mask)

def compute_metrics(csv_path: str):

    """
    Reads a CSV file containing 'Predicted Fitness' and 'True Fitness' columns,
    and computes MSE, RMSE, R-squared, Pearson, and Spearman correlations.

    :param csv_file: Path to the CSV file with predictions and true values.
    :return: Dictionary with MSE, RMSE, R², Spearman correlation, and Pearson correlation.
    """

    # Read the CSV file into a Pandas DataFrame
    df = pd.read_csv(csv_path)

    # Ensure the DataFrame has the required columns
    if 'Predicted Fitness' not in df.columns or 'True Fitness' not in df.columns:
        raise ValueError("CSV file must contain 'Predicted Fitness' and 'True Fitness' columns.")

    # Extract the predicted and true fitness values as tensors
    predicted_values = torch.tensor(df['Predicted Fitness'].values, dtype=torch.float32)
    true_values = torch.tensor(df['True Fitness'].values, dtype=torch.float32)

    # Calculate Mean Squared Error (MSE)
    mse = torch.mean((predicted_values - true_values) ** 2).item()

    # Calculate Root Mean Squared Error (RMSE)
    rmse = torch.sqrt(torch.tensor(mse)).item()

    # Calculate R² (R-squared)
    ss_res = torch.sum((true_values - predicted_values) ** 2)
    ss_tot = torch.sum((true_values - torch.mean(true_values)) ** 2)
    r2 = 1 - (ss_res / ss_tot).item()

    # Convert tensors to NumPy arrays for Pearson and Spearman correlation
    predicted_values_np = predicted_values.numpy()
    true_values_np = true_values.numpy()

    # Calculate Pearson correlation
    pearson_corr, _ = pearsonr(predicted_values_np, true_values_np)

    # Calculate Spearman correlation
    spearman_corr, _ = spearmanr(predicted_values_np, true_values_np)

    # Print and return all metrics
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"R²: {r2}")
    print(f"Spearman Correlation: {spearman_corr}")
    print(f"Pearson Correlation: {pearson_corr}")

    return {
        'MSE': mse,
        'RMSE': rmse,
        'R²': r2,
        'Spearman': spearman_corr,
        'Pearson': pearson_corr
    }

def plot_predictions_vs_true(predictions_df: pd.DataFrame):

    """
    Plots a scatter plot of predicted vs true fitness values with a y = x reference line.

    predictions_df: A Pandas DataFrame with 'Predicted Fitness' and 'True Fitness' columns.
    """
    true_values = predictions_df['True Fitness']
    predicted_values = predictions_df['Predicted Fitness']

    plt.figure(figsize=(8, 6))

    # Scatter plot of predicted vs true values
    plt.scatter(true_values, predicted_values, color = 'blue', label = 'Predicted vs True', s = 0.1, alpha = 0.8)

    # Plot y=x line for perfect predictions
    min_val = min(true_values.min(), predicted_values.min())
    max_val = max(true_values.max(), predicted_values.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')

    # Labels and title
    plt.xlabel('True Fitness')
    plt.ylabel('Predicted Fitness')
    plt.title('Predicted vs True Fitness Values')
    plt.legend()

    # Show the plot
    plt.savefig("results/figures/predictedvstrue.png")
