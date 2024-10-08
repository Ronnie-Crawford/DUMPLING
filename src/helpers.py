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

    if domain_name_splitter != None:

        return domain_name.rsplit(domain_name_splitter, 1)[0] + ".pdb"
    
    else:
        
        return domain_name

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

def is_tensor_ready(value):

    """
    Helper method to check if a value can be converted to a float.
    """

    if isinstance(value, bool):
        
        return False

    try:

        float(value)

        return True

    except ValueError:

        return False
    
def make_tensor_ready(value):
    
    if value == False:
        
        return False

    else:
        
        return float(value)

def get_family_size(family, family_dict: dict):

        family_domains = family_dict[family]
        family_mask = [domain in family_domains for domain in dataset.domain_names]
        return sum(family_mask)

import pandas as pd
import torch
import math
from scipy.stats import pearsonr, spearmanr

def compute_metrics(csv_path: str, parameter: str, min_count: int = 10):
    """
    Reads a CSV file containing 'Predicted Energy' and 'True Energy' or 
    'Predicted Fitness' and 'True Fitness' columns, and computes MSE, RMSE, 
    R-squared, Pearson, and Spearman correlations.

    :param csv_path: Path to the CSV file with predictions and true values.
    :param parameter: 'energy' or 'fitness' to specify which metrics to compute.
    :param min_count: Minimum number of valid data points required to compute metrics.
    :return: Dictionary with MSE, RMSE, R², Spearman correlation, and Pearson correlation,
             or None values if the minimum count is not met.
    """

    # Validate the 'parameter' argument
    if parameter == "energy":
        title = "Energy Prediction Metrics"
        predicted_column = "Predicted Energy"
        truth_column = "True Energy"
    elif parameter == "fitness":
        title = "Fitness Prediction Metrics"
        predicted_column = "Predicted Fitness"
        truth_column = "True Fitness"
    else:
        raise ValueError("Parameter must be 'energy' or 'fitness'.")

    # Read the CSV file into a Pandas DataFrame
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {csv_path} was not found.")
    except pd.errors.EmptyDataError:
        raise ValueError(f"The file {csv_path} is empty.")
    except Exception as e:
        raise Exception(f"An error occurred while reading the file {csv_path}: {e}")

    # Ensure the DataFrame has the required columns
    if predicted_column not in df.columns or truth_column not in df.columns:
        raise ValueError(f"CSV file must contain '{predicted_column}' and '{truth_column}' columns.")

    # Drop rows with NaNs in either predicted or true columns
    filtered_df = df[[predicted_column, truth_column]].dropna()

    # Count the number of valid data points
    valid_count = len(filtered_df)

    # Check if the number of valid data points meets the minimum threshold
    if valid_count < min_count:
        print(f"Not enough valid data points for {parameter}. Required: {min_count}, Found: {valid_count}")
        return {
            'MSE': None,
            'RMSE': None,
            'R²': None,
            'Spearman': None,
            'Pearson': None
        }

    # Extract the predicted and true values as NumPy arrays
    try:
        predicted_values_np = filtered_df[predicted_column].astype(float).values
        true_values_np = filtered_df[truth_column].astype(float).values
    except ValueError as e:
        raise ValueError(f"Error converting columns to float: {e}")

    # Calculate Mean Squared Error (MSE)
    mse = ((predicted_values_np - true_values_np) ** 2).mean()

    # Calculate Root Mean Squared Error (RMSE)
    rmse = math.sqrt(mse)

    # Calculate R² (R-squared)
    ss_res = ((true_values_np - predicted_values_np) ** 2).sum()
    ss_tot = ((true_values_np - true_values_np.mean()) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else float('nan')

    # Calculate Pearson correlation
    try:
        pearson_corr, _ = pearsonr(predicted_values_np, true_values_np)
    except Exception as e:
        pearson_corr = float('nan')
        print(f"Pearson correlation calculation failed: {e}")

    # Calculate Spearman correlation
    try:
        spearman_corr, _ = spearmanr(predicted_values_np, true_values_np)
    except Exception as e:
        spearman_corr = float('nan')
        print(f"Spearman correlation calculation failed: {e}")

    # Print and return all metrics
    print(f"\n{title}")
    print(f"Number of valid data points: {valid_count}")
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
