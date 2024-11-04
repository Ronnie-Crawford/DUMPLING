# Standard modules
import os
import gc
import math

# Third-party modules
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

def get_device(device: str) -> torch.device:

    """
    Determines the best available device (GPU, MPS, or CPU).

    Returns:

        - torch.device: The best available device.
    """

    if device == None:

        if torch.cuda.is_available():

            print("Using device: CUDA.")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

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

def manage_memory():

    if torch.cuda.is_available():

        total_memory, free_memory = torch.cuda.mem_get_info()

        if total_memory / free_memory  < 0.1:

            print("GPU Memory Available: ", round(total_memory * 100 / free_memory, 3), "%.")
            print("Emptying CUDA cache.")
            torch.cuda.empty_cache()
            print("GPU Memory Available: ", round(total_memory * 100 / free_memory, 3), "%.")

def truncate_domain(domain_name: str, domain_name_splitter: str) -> str:

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

def is_tensor_ready(value) -> bool:

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

def compute_metrics(csv_path: str, parameter: str, min_count: int = 10):
    """
    Reads a CSV file containing Predicted Energy and True Energy or
    Predicted Fitness and True Fitness columns, and computes MSE, RMSE,
    R-squared, Pearson, and Spearman correlations.

    Parameters:
        - csv_path (str): Path to the CSV file with predictions and true values.
        - parameter (str): "energy" or "fitness" to specify which metrics to compute.
        - min_count (int): Minimum number of valid data points required to compute metrics.

    Returns:
        - (dict): Dictionary with MSE, RMSE, R², Spearman correlation, and Pearson correlation, or None values if the minimum count is not met.
    """

    if parameter == "energy":

        title = "Energy Prediction Metrics"
        predicted_column = "Predicted Energy"
        truth_column = "True Energy"

    elif parameter == "fitness":

        title = "Fitness Prediction Metrics"
        predicted_column = "Predicted Fitness"
        truth_column = "True Fitness"

    else:
        raise ValueError("""Parameter must be "energy" or "fitness.""")

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

def normalise_embeddings(embeddings_list):

    normalised_embeddings_list = []

    for embeddings in embeddings_list:

        normalised_embeddings = []

        for embedding in embeddings:

            normalised_embedding = normalise_tensor(embedding)
            normalised_embeddings.append(normalised_embedding)

        normalised_embeddings_list.append(normalised_embeddings)

    return normalised_embeddings_list

def normalise_tensor(tensor):

    vector = tensor.cpu().numpy()

    try:

        normalised_vector = (vector - np.mean(vector)) / np.std(vector)

    except Exception as e:

        print(f"Could not normalise vector: {e}")
        normalised_vector = vector

    return torch.tensor(normalised_vector)

def concatenate_embeddings(embeddings_list: list) -> list:

    concatenated_embeddings = []

    for variant_embeddings in list(zip(*embeddings_list)):

        concatenated_embeddings.append(torch.concat(variant_embeddings))

    return concatenated_embeddings

def fit_principal_components(embeddings, component_index: int, device: str = "cpu"):

    assert embeddings.dim() == 3, "Input embeddings must be a 3D tensor such that each slice is a point."

    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    with torch.no_grad():

        batch_size, n_sequences, n_residues = embeddings.shape
        flattened_embeddings = embeddings.reshape(-1, n_residues)
        centered_points = flattened_embeddings - flattened_embeddings.mean(dim = 1, keepdim = True)

        # Compute the covariance matrices for each sample
        centered_points = centered_points.reshape(batch_size, n_sequences, n_residues)
        covariance_matrices = torch.einsum("ijk,ijl->ikl", centered_points, centered_points) / (n_sequences - 1)

        # Compute eigenvalues and eigenvectors for each covariance matrix
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrices)

        # Sort eigenvalues and eigenvectors
        sorted_indices = torch.argsort(eigenvalues, dim=-1, descending=True)
        sorted_eigenvectors = eigenvectors.gather(2, sorted_indices.unsqueeze(-1).expand(-1, -1, eigenvectors.size(-1)))

    # Select principal component
    if component_index < 1 or component_index > sorted_eigenvectors.shape[1]:

        raise ValueError("Component_index must be between 1 and the number of components.")

    principal_components = sorted_eigenvectors[:, :, component_index - 1]

    # Try desperately to reduce memory usage and salvage back anything from the belly of the beast
    del embeddings, component_index, batch_size, n_sequences, n_residues, flattened_embeddings, centered_points, covariance_matrices, eigenvalues, eigenvectors, sorted_indices, sorted_eigenvectors
    gc.collect()
    torch.cuda.empty_cache()

    return principal_components
