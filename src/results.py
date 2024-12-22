# Standard modules
import math

# Third-party modules
import pandas as pd
from scipy.stats import pearsonr, spearmanr

def compute_metrics(results_path: str, output_features: list, min_count: int = 10) -> dict:
    
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

    metrics = {}
    csv_path = results_path / "results.csv"

    try:
            
        results = pd.read_csv(csv_path)

    except FileNotFoundError:

        raise FileNotFoundError(f"The file {csv_path} was not found.")

    except pd.errors.EmptyDataError:

        raise ValueError(f"The file {csv_path} is empty.")

    except Exception as e:

        raise Exception(f"An error occurred while reading the file {csv_path}: {e}")

    for output_feature in output_features:
        
        title = f"{output_feature.capitalize()} Prediction Metrics"
        predicted_column = f"{output_feature}_predictions"
        truth_column = f"{output_feature}_truth"
            
        if predicted_column not in results.columns or truth_column not in results.columns:

            raise ValueError(f"CSV file must contain '{predicted_column}' and '{truth_column}' columns.")
        
        # Drop rows with NaNs in either predicted or true columns
        filtered_df = results[[predicted_column, truth_column]].dropna()
        valid_count = len(filtered_df)
        
        # Check if the number of valid data points meets the minimum threshold
        if valid_count < min_count:
            
            print(f"Not enough valid data points for {output_feature}. Required: {min_count}, Found: {valid_count}")
            
            metrics[output_feature] = {
                'MSE': None,
                'RMSE': None,
                'R²': None,
                'Spearman': None,
                'Pearson': None
            }
            
            continue
        
        # Extract the predicted and true values as NumPy arrays
        try:
            
            predicted_values_np = filtered_df[predicted_column].astype(float).values
            true_values_np = filtered_df[truth_column].astype(float).values
            
        except ValueError as e:
            
            raise ValueError(f"Error converting columns to float: {e}")

        mse = ((predicted_values_np - true_values_np) ** 2).mean()
        rmse = math.sqrt(mse)
        ss_res = ((true_values_np - predicted_values_np) ** 2).sum()
        ss_tot = ((true_values_np - true_values_np.mean()) ** 2).sum()
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else float('nan')
        
        try:
        
            pearson_corr, _ = pearsonr(predicted_values_np, true_values_np)
        
        except Exception as e:
            
            pearson_corr = float('nan')
            print(f"Pearson correlation calculation failed: {e}")

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

        metrics[output_feature] = {
            'MSE': mse,
            'RMSE': rmse,
            'R²': r2,
            'Spearman': spearman_corr,
            'Pearson': pearson_corr
        }
    
    return metrics

def compute_metrics_old(results_path: str, parameter: str, min_count: int = 10):
    
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

    csv_path = results_path / "results.csv"

    if parameter == "energy":

        title = "Energy Prediction Metrics"
        predicted_column = "energy_predictions"
        truth_column = "energy_truth"

    elif parameter == "fitness":

        title = "Fitness Prediction Metrics"
        predicted_column = "fitness_predictions"
        truth_column = "fitness_truth"

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

    mse = ((predicted_values_np - true_values_np) ** 2).mean()
    rmse = math.sqrt(mse)
    ss_res = ((true_values_np - predicted_values_np) ** 2).sum()
    ss_tot = ((true_values_np - true_values_np.mean()) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else float('nan')

    try:
        
        pearson_corr, _ = pearsonr(predicted_values_np, true_values_np)
        
    except Exception as e:
        
        pearson_corr = float('nan')
        print(f"Pearson correlation calculation failed: {e}")

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