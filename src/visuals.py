# Standard modules
from itertools import combinations

# Third party modules
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from sklearn.decomposition import PCA
#from sklearn.preprocessing import LabelEncoder
#import umap.umap_ as umap

def plot_predictions_vs_true(predictions_df: pd.DataFrame, output_features: list, results_path):

    for output_feature in output_features:
        
        truth_column = f"{output_feature}_truth"
        predicted_column = f"{output_feature}_predictions"
        
        true_values = predictions_df[truth_column]
        predicted_values = predictions_df[predicted_column]
        
        plt.scatter(true_values, predicted_values, color = "blue", label = "Predicted vs True", s = 0.1, alpha = 0.8)
        min_val = min(true_values.min(), predicted_values.min())
        max_val = max(true_values.max(), predicted_values.max())
        plt.plot([min_val, max_val], [min_val, max_val], "r--")
        
        plt.xlabel(f"True {output_feature.capitalize()}")
        plt.ylabel(f"Predicted {output_feature.capitalize()}")
        plt.title(f"Predicted vs True {output_feature.capitalize()} Values")
        plt.legend()
        
        plt.savefig(results_path / f"{output_feature}_accuracy_scatter.png")
        plt.close()

def plot_input_histogram(predictions_df: pd.DataFrame, output_features: list, results_path):

    for output_feature in output_features:
        
        truth_column = f"{output_feature}_truth"
        true_values = predictions_df[truth_column]
        
        if true_values.notnull().values.any():

            plt.hist(true_values, bins = 100)
            plt.title(f"Histogram of True {output_feature.capitalize()}")
            plt.savefig(results_path / f"input_{output_feature}_histogram.png")
            plt.close()
    
def plot_embeddings(
    datasets: list,
    dim_reduction_method: str,
    n_components: int,
    output_attribute: str,
    device: str,
    results_path,
    label_type: str
):
    """
    Plots embeddings of variants in 2D space using PCA or UMAP, color-coded by fitness/energy or domains.

    Parameters:
        - datasets (list): List of PyTorch datasets.
        - dim_reduction_method (str): 'PCA' or 'UMAP'.
        - n_components (int): Number of dimensions to reduce to.
        - output_attribute (str): Attribute name for fitness or energy values in the dataset.
        - device (str): Device to use.
        - results_path: Path to save the results.
        - label_type (str): 'outputs' or 'domains' to specify how to color the points.
    """

    # Collect embeddings and outputs from datasets
    all_embeddings = []
    all_outputs = []
    all_domain_names = []

    for idx, dataset in enumerate(datasets):
        
        all_embeddings.append(torch.stack(dataset.sequence_representations))
        all_outputs.append(torch.tensor(getattr(dataset, output_attribute)))
        all_domain_names.extend(dataset.domain_names)

    # Concatenate all datasets
    embeddings_tensor = torch.cat(all_embeddings, dim=0)
    outputs_tensor = torch.cat(all_outputs, dim=0)

    # Prepare labels based on label_type
    if label_type == "outputs":
        
        labels = outputs_tensor.numpy()
        colorbar_label = output_attribute.capitalize()
        cmap = 'coolwarm'
        
    elif label_type == "domains":
        
        # Encode domain names as integers
        
        label_encoder = LabelEncoder()
        domain_labels = label_encoder.fit_transform(all_domain_names)
        labels = domain_labels
        colorbar_label = 'Domains'
        unique_domains = np.unique(domain_labels)
        n_domains = len(unique_domains)
        cmap = plt.cm.get_cmap('tab20', n_domains)
            
    else:
        
        labels = None
        colorbar_label = ''
        cmap = None

    # Apply dimensionality reduction
    if dim_reduction_method == "PCA":
        
        embeddings_np = embeddings_tensor.cpu().numpy()
        pca = PCA(n_components = n_components)
        reduced_embeddings = pca.fit_transform(embeddings_np)
        
    elif dim_reduction_method == "UMAP":
        
        reduced_embeddings = calculate_umap(embeddings_tensor, n_components=n_components)
        
    else:
        
        raise ValueError("dim_reduction_method must be 'PCA' or 'UMAP'")

    # Plotting: Create pairwise scatter plots of components
    component_indices = range(n_components)
    pairs = list(combinations(component_indices, 2))
    n_plots = len(pairs)

    # Determine grid size
    n_cols = int(np.ceil(np.sqrt(n_plots)))
    n_rows = int(np.ceil(n_plots / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False)

    for idx, (i, j) in enumerate(pairs):
        
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row][col]
        sc = ax.scatter(
            reduced_embeddings[:, i],
            reduced_embeddings[:, j],
            c = labels,
            cmap = cmap,
            s = 0.5,
            alpha = 0.8
        )
        ax.set_xlabel(f'Component {i+1}')
        ax.set_ylabel(f'Component {j+1}')
        ax.set_title(f'Components {i+1} vs {j+1}')
        
    # Remove any unused subplots
    for idx in range(n_plots, n_rows * n_cols):
        
        fig.delaxes(axes.flatten()[idx])

    # Adjust layout and add color bar
    plt.tight_layout()
    fig.subplots_adjust(right = 0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(sc, cax = cbar_ax, label = colorbar_label)

    plt.suptitle(f'Dimensionally Reduced Embeddings ({dim_reduction_method})', fontsize=16)
    plt.savefig(results_path / f"embedding_{output_attribute}_{dim_reduction_method}_{label_type}.png", dpi=300)
    plt.close()

    
def calculate_umap(embeddings, n_components = 2):
    
    """
    Reduces embeddings to a lower dimension using UMAP.

    Parameters:
        - embeddings: Tensor of shape (num_samples, embedding_dim).
        - n_components (int): Number of dimensions to reduce to.

    Returns:
        - reduced_embeddings: Numpy array of reduced embeddings.
    """
    
    embeddings_np = embeddings.cpu().numpy()
    reducer = umap.UMAP(n_components = n_components)
    reduced_embeddings = reducer.fit_transform(embeddings_np)
    
    return reduced_embeddings

def plot_loss(training_loss, validation_loss, results_path):
    
    """
    Plots the training and validation loss over epochs.

    Parameters:
    - training_loss (list): List of training loss values.
    - validation_loss (list): List of validation loss values.
    """
    
    epochs = range(1, len(training_loss) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, training_loss, 'b-', label = "Training Loss")
    
    if len(validation_loss) > 0:
        
        plt.plot(epochs, validation_loss, 'r-', label = "Validation Loss")
    
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig((results_path / "loss.png"))
    plt.close()
    
def handle_embedding_plots(all_datasets, device, results_path):
    
    plot_embeddings(
        datasets = all_datasets,
        dim_reduction_method = "PCA",
        n_components = 4,
        output_attribute = "fitness_values",
        device = device,
        results_path = results_path,
        label_type = "outputs"
        )
    plot_embeddings(
        datasets = all_datasets,
        dim_reduction_method = "PCA",
        n_components = 4,
        output_attribute = "fitness_values",
        device = device,
        results_path = results_path,
        label_type = "domains"
        )
    plot_embeddings(
        datasets = all_datasets,
        dim_reduction_method = "UMAP",
        n_components = 4,
        output_attribute = "fitness_values",
        device = device,
        results_path = results_path,
        label_type = "outputs"
        )
    plot_embeddings(
        datasets = all_datasets,
        dim_reduction_method = "UMAP",
        n_components = 4,
        output_attribute = "fitness_values",
        device = device,
        results_path = results_path,
        label_type = "domains"
        )
