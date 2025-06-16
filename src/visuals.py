# Standard modules
import json
import re
from pathlib import Path

# Third-party modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import gaussian_kde

def plot_predictions_vs_true(predictions_df: pd.DataFrame, output_features: list, results_path):

    for output_feature in output_features:
        
        truth_column = f"{output_feature}_truth"
        predicted_column = f"{output_feature}_predictions"
        
        true_values = predictions_df[truth_column].to_numpy()
        predicted_values = predictions_df[predicted_column].to_numpy()
        keep_mask = np.isfinite(true_values) & np.isfinite(predicted_values)
        true_values = true_values[keep_mask]
        predicted_values = predicted_values[keep_mask]
        
        plt.scatter(true_values, predicted_values, color = "red", label = "Predicted vs True", s = 0.1, alpha = np.clip((1000 / len(true_values)), 0, 1)
        min_val = min(true_values.min(), predicted_values.min())
        max_val = max(true_values.max(), predicted_values.max())
        plt.plot([min_val, max_val], [min_val, max_val], "r--")
        
        # Checks if there is variance for contours
        if not np.allclose(predicted_values, predicted_values[0]):
            
            if len(true_values) > 5000:
            
                idx = np.random.choice(len(true_values), 5000, replace = False)
                true_values, predicted_values = true_values[idx], predicted_values[idx]

            density = gaussian_kde(np.vstack([true_values, predicted_values]))(np.vstack([true_values, predicted_values]))
            plt.tricontour(true_values, predicted_values, density, colors = "blue", alpha = 0.5)
        
        plt.xlabel(f"True {output_feature.capitalize()}")
        plt.ylabel(f"Predicted {output_feature.capitalize()}")
        plt.title(f"Predicted vs True {output_feature.capitalize()} Values")
        plt.legend()
        
        plt.savefig(results_path / f"{output_feature}_accuracy_scatter.png")
        plt.close()

def plot_benchmark_grid(results_base_path, output_feature, minimum_valid_datapoints = 10):
        
    train_folders = sorted([path for path in Path(results_base_path).iterdir() if path.is_dir()])
    train_dataset_names = [path.name.replace("trained_on_", "") for path in train_folders]
    train_roots = {name: (results_base_path / f"trained_on_{name}") for name in train_dataset_names}
    
    # Pick any one train folder to extract test_subset names; assume all use the same set of test subsets.
    sample_folder = train_folders[0]
    metrics_path  = sample_folder / "metrics_by_subset.json"

    with open(metrics_path, "r") as metrics_file:
        
        sample_metrics = json.load(metrics_file)
        
    test_dataset_names = sorted(sample_metrics.keys())
    
    # Create grid
    fig, axes = plt.subplots(
        len(test_dataset_names),
        len(train_dataset_names),
        figsize = (4 * len(train_dataset_names), 4 * len(test_dataset_names)),
        squeeze = False
    )
    axes = np.atleast_2d(axes)

    # Loop through axes
    for col_idx, train_subset in enumerate(train_dataset_names):
        
        current_folder = train_roots[train_subset]
        preds_df = pd.read_csv(current_folder / "results.csv", comment = "#")
        
        with open(current_folder / "metrics_by_subset.json", "r") as mf:
            
            metrics_by_subset = json.load(mf)

        for row_idx, test_subset in enumerate(test_dataset_names):
            
            ax = axes[row_idx, col_idx]
            ax.cla()

            # Filter only the rows for this particular test_subset
            cell_df = preds_df[preds_df["subset"] == test_subset].copy()
            cell_overall = metrics_by_subset[test_subset]["overall"]
            cell_domain  = metrics_by_subset[test_subset]["domain"]

            histogram_domain_metrics(cell_df, output_feature, cell_domain, cell_overall, ax)
            predictions_scatterplot(output_feature, cell_df, ax)
    
    for index, train_dataset in enumerate(train_dataset_names):
        
        axes[0, index].set_title(f"Train: {train_dataset}", fontsize = 8)

    for index, test_dataset in enumerate(test_dataset_names):
        
        axes[index, 0].set_ylabel(f"Test: {test_dataset}", fontsize = 8, rotation = 90, labelpad = 5)
    
    plt.tight_layout()
    output_file = results_base_path / f"{output_feature}_benchmark_grid.png"
    plt.savefig(output_file, dpi = 200)
    plt.close()

def old_plot_benchmark_grid(results_base_path, output_feature, minimum_valid_datapoints = 10):
    
    # Set up paths and train / test names
    results_base_path = Path(results_base_path)
    train_root = results_base_path / "train"
    test_root  = results_base_path / "test"
    train_dataset_names = sorted([folder.name for folder in train_root.iterdir() if folder.is_dir()])
    test_folder_names = [folder.name for folder in test_root.iterdir() if folder.is_dir()]
    test_dataset_set = set()

    for folder_name in test_folder_names:
        
        match = re.match(r"trained_on_(.+)_tested_on_(.+)", folder_name)
        
        if match:
            
            test_dataset_set.add(match.group(2))

    test_dataset_names = sorted(test_dataset_set)
    
    # Set up figure grid
    fig, axes = plt.subplots(len(test_dataset_names), len(train_dataset_names), figsize=(4 * len(train_dataset_names), 4 * len(test_dataset_names)))
    axes = np.atleast_2d(axes)
    
    for index, test_dataset in enumerate(test_dataset_names):
        
        for jndex, train_dataset in enumerate(train_dataset_names):
            
            predictions_df, overall_metrics, domain_metrics = set_up_axis_specific_data(test_root, train_dataset, test_dataset)
            ax = axes[index, jndex]
            ax.cla()
            
            if empty_data_check(output_feature, domain_metrics, predictions_df, ax): continue
            
            histogram_domain_metrics(predictions_df, output_feature, domain_metrics, overall_metrics, ax)
            predictions_scatterplot(output_feature, predictions_df, ax)
    
    for index, train_dataset in enumerate(train_dataset_names):
        
        axes[0, index].set_title(f"Train: {train_dataset}", fontsize = 8)

    for index, test_dataset in enumerate(test_dataset_names):
        
        axes[index, 0].set_ylabel(f"Test: {test_dataset}", fontsize = 8, rotation = 90, labelpad = 5)
    
    plt.tight_layout()
    output_file = results_base_path / f"{output_feature}_benchmark_grid.png"
    plt.savefig(output_file, dpi = 200)
    plt.close()
            
def set_up_axis_specific_data(test_root, train_dataset, test_dataset):
            
    current_path = test_root  / f"trained_on_{train_dataset}_tested_on_{test_dataset}"
    predictions_df = pd.read_csv(current_path / "results.csv", comment = "#")
    
    with open(current_path / "overall_metrics.json", "r") as overall_metrics_file:
        
        overall_metrics = json.load(overall_metrics_file)

    with open(current_path / "domain_specific_metrics.json", "r") as domain_specific_metrics_file:
        
        domain_metrics = json.load(domain_specific_metrics_file)
    
    return predictions_df, overall_metrics, domain_metrics
            
            
def empty_data_check(output_feature, domain_metrics, predictions_df, ax, minimum_valid_datapoints = 10):

    """
    Checks if there is data, skips if not
    """
    
    feature_metrics = domain_metrics.get(output_feature)
    
    if sum(~np.isnan(predictions_df[f"{output_feature}_truth"].values)) < minimum_valid_datapoints:
        
        ax.text(0.5, 0.5, "No data", ha = "center", va = "center", fontsize = 10, alpha = 0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        
        return True
    
    return False

def histogram_domain_metrics(predictions_df, output_feature, domain_metrics, overall_metrics, ax):

    spearman_vals = np.array(list(domain_metrics[output_feature]["Spearman"].values()))
    pearson_vals = np.array(list(domain_metrics[output_feature]["Pearson"].values()))
    bins = np.linspace(-1, 1, 21)
    ax.hist(spearman_vals, bins = bins, color = "blue", alpha = 0.5, label = "Spearman")
    ax.hist(pearson_vals, bins = bins, color = "green", alpha = 0.5, label = "Pearson")
    ax.legend(loc = "lower center", bbox_to_anchor = (0.5, 0.02), fontsize = 8, ncol = 2)
    ax.set_xlabel(
        f"Domains: {len(domain_metrics[output_feature]['Pearson'])}\n"
        f"Variants: {len(predictions_df[f'{output_feature}_truth'])}\n"
        f"Overall Spearman: {overall_metrics[output_feature]['Spearman']:.3f}\n"
        f"Overall Pearson: {overall_metrics[output_feature]['Pearson']:.3f}",
        fontsize = 8
    )

def predictions_scatterplot(output_feature, predictions_df, ax):

    inset_ax = ax.inset_axes([0.1, 0.45, 0.45, 0.45])
    x = predictions_df[f"{output_feature}_truth"].values
    y = predictions_df[f"{output_feature}_predictions"].values
    inset_ax.scatter(x, y, color = "red", s = 5, alpha = 0.1)
    min_point, max_point = min(x.min(), y.min()), max(x.max(), y.max())
    inset_ax.plot([min_point, max_point], [min_point, max_point], "k--", alpha = 0.1)
    inset_ax.xaxis.set_major_locator(MaxNLocator(integer = True))
    inset_ax.yaxis.set_major_locator(MaxNLocator(integer = True))

    # Checks if there is variance for contours
    if not np.allclose(y, y[0]):
        
        if len(x) > 5000:
        
            idx = np.random.choice(len(x), 5000, replace = False)
            x, y = x[idx], y[idx]

        density = gaussian_kde(np.vstack([x, y]))(np.vstack([x, y]))
        inset_ax.tricontour(x, y, density, colors = "blue", alpha = 0.5)