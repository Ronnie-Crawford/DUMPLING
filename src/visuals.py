# Third party modules
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap.umap_ as umap

def plot_predictions_vs_true(predictions_df: pd.DataFrame):

    truth_column = "True Fitness"
    predicted_column = "Predicted Fitness"

    true_values = predictions_df[truth_column]
    predicted_values = predictions_df[predicted_column]

    plt.scatter(true_values, predicted_values, color = "blue", label = "Predicted vs True", s = 0.1, alpha = 0.8)
    min_val = min(true_values.min(), predicted_values.min())
    max_val = max(true_values.max(), predicted_values.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--")
    plt.xlabel("True Fitness")
    plt.ylabel("Predicted Fitness")
    plt.title("Predicted vs True Fitness Values")
    plt.legend()

    plt.savefig("./results/figures/accuracy_scatter.png")
    plt.close()

def plot_input_histogram(predictions_df: pd.DataFrame):

    truth_column = "True Fitness"
    true_values = predictions_df[truth_column]

    plt.hist(true_values, bins = 100)
    plt.title("Histogram of True Fitness")

    plt.savefig("./results/figures/input_histogram.png")
    plt.close()
    
def plot_embeddings(
    datasets: list,
    dim_reduction_method: str = "PCA",
    n_components: int = 6,
    output_attribute: str = "fitness_values",
    title: str = "Embeddings Visualization",
    device: str = "cpu"
):
    """
    Plots embeddings of variants in 2D space using PCA or UMAP, color-coded by fitness/energy.

    Parameters:
        - datasets (list): List of PyTorch datasets.
        - dim_reduction_method (str): 'PCA' or 'UMAP'.
        - n_components (int): Number of dimensions to reduce to.
        - fitness_attribute (str): Attribute name for fitness or energy values in the dataset.
        - title (str): Title of the plot.
    """
    
    # Collect embeddings and fitness values from datasets
    all_embeddings = []
    all_outputs = []
    dataset_labels = []

    for idx, dataset in enumerate(datasets):
        
        all_embeddings.append(torch.stack(dataset.sequence_representations))
        all_outputs.append(torch.tensor(getattr(dataset, output_attribute)))
        dataset_labels.extend([f'Dataset {idx+1}'] * len(dataset))

    # Concatenate all datasets
    embeddings_tensor = torch.cat(all_embeddings, dim = 0)
    outputs_tensor = torch.cat(all_outputs, dim = 0)

    # Apply dimensionality reduction
    if dim_reduction_method == "PCA":
        
        embeddings_tensor = embeddings_tensor.cpu()
        embeddings_np = embeddings_tensor.numpy()
        pca = PCA(n_components = n_components)
        reduced_embeddings = pca.fit_transform(embeddings_np)

    elif dim_reduction_method == "UMAP":
        
        reduced_embeddings = calculate_umap(embeddings_tensor, n_components = n_components)

    else:
        
        raise ValueError("dim_reduction_method must be 'PCA' or 'UMAP'")

    # Plot the embeddings
    if n_components == 2:
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            reduced_embeddings[:, 0],
            reduced_embeddings[:, 1],
            c = outputs_tensor.numpy(),
            cmap = "coolwarm",
            alpha = 0.95,
            s = 0.05
        )
        plt.colorbar(scatter, label = output_attribute.capitalize())
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title(title)
        plt.grid(True)
        plt.savefig("./results/figures/embedding_components.png", dpi = 600)
        plt.close()
        
    elif n_components > 2:
        
        component_indices = range(n_components)

        # Create a grid of plots
        fig, axes = plt.subplots(
            n_components,
            n_components,
            figsize=(4 * n_components, 4 * n_components),
            sharex = "col",
            sharey = False
        )

        # Ensure axes is a 2D array
        if n_components == 1:
            
            axes = np.array([[axes]])

        for i in component_indices:
            
            for j in component_indices:
                
                ax = axes[i, j]
                
                if i == j:
                    
                    # Plot histogram on the diagonal
                    ax.hist(reduced_embeddings[:, i], bins = 30, color = "gray", alpha = 0.1)
                    ax.set_xlabel(f"Component {j+1}")
                    ax.set_ylabel("Frequency")
                    
                else:
                    # Scatter plot for off-diagonal
                    scatter = ax.scatter(
                        reduced_embeddings[:, j],
                        reduced_embeddings[:, i],
                        c = outputs_tensor.numpy(),
                        cmap = "coolwarm",
                        alpha = 0.95,
                        s = 0.05
                    )
                if i == n_components - 1:
                    
                    ax.set_xlabel(f'Component {j+1}')
                    
                if j == 0:
                    
                    ax.set_ylabel(f'Component {i+1}')

        # Adjust layout and add color bar
        plt.tight_layout()
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.95, 0.1, 0.02, 0.8])
        fig.colorbar(scatter, cax = cbar_ax, label = output_attribute.capitalize())
        plt.suptitle(title, fontsize=16)
        plt.savefig("./results/figures/embedding_components.png", dpi = 500)
        plt.close()
    
def calculate_umap(embeddings, n_components = 2, random_state = 1):
    
    """
    Reduces embeddings to a lower dimension using UMAP.

    Parameters:
        - embeddings: Tensor of shape (num_samples, embedding_dim).
        - n_components (int): Number of dimensions to reduce to.

    Returns:
        - reduced_embeddings: Numpy array of reduced embeddings.
    """
    
    embeddings_np = embeddings.numpy()
    reducer = umap.UMAP(n_components=n_components, random_state=random_state)
    reduced_embeddings = reducer.fit_transform(embeddings_np)
    
    return reduced_embeddings
