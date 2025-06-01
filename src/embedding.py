#Standard modules
import gc
import itertools
import hashlib
import pickle
import os
from pathlib import Path

# Third-party modules
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, EsmModel, EsmForProteinFolding, EsmTokenizer

def handle_embeddings(
    dataset_dicts: list,
    embedding_model_names: list,
    batch_size: int,
    embedding_layers: list,
    embedding_types: list,
    dimensional_reduction_method: str,
    n_desired_dimensions: int,
    normalise_embeddings: bool,
    wildtype_concat: bool,
    wildtype_delta: bool,
    base_path,
    device: str,
    n_workers: int
    ):
    
    """
    The handler for embeddings, runs the upstream models and coordinates and post-processing on the extracted embeddings:
    concatination, dimensional reduction, etc.
    """
    
    dataset_dicts, embedding_size = load_embeddings(
        dataset_dicts,
        batch_size,
        embedding_model_names,
        embedding_layers,
        embedding_types,
        base_path,
        device,
        n_workers
        )
    
    dataset_dicts = post_process_embeddings(
        dataset_dicts,
        normalise_embeddings,
        wildtype_concat,
        wildtype_delta,
        dimensional_reduction_method,
        n_desired_dimensions
        )
    
    return dataset_dicts, embedding_size

def load_embeddings(
    dataset_dicts: dict,
    batch_size: int,
    model_selections: list,
    embedding_layers: list,
    embedding_types: list,
    package_folder: Path,
    device: str,
    n_workers: int
    ) -> tuple[list, int]:
    
    """
    Loads embeddings for each dataset, embeddings are saved for each dataset, not dataset-label subset,
    to save time generating embeddings and save space storing them.
    The trade-off is that embeddings must be loaded for the entire dataset of each subset - before it is filtered.
    The dataset hash matching is far from perfect, if the sequences are in a different order for example it will not recognise the embeddings,
    but it takes a lot of computation to regenerate each time, so I try to minimise changing the mechanisms.
    Generally there will be only one upstream model, hidden layer and post-processing type selected,
    but it is possible to load more and the results will be concatinated together,
    however I find that the increase in dimensionality quickly loses any benefit that increasing the available information might have had.
    """
    
    # Make list of all combinations of upstream models, hidden layer to extract embedding from, post-processing to apply to embeddings
    embedding_combinations = list(itertools.product(model_selections, embedding_layers, embedding_types))
    # Create dataframe to store all emebddings
    merged_embeddings_df = pd.DataFrame()
    
    for index, dataset_dict in enumerate(dataset_dicts):
        
        dataset_name = dataset_dict["dataset_name"]
        dataset = dataset_dict["dataset"]
        embeddings_list = []
        dataset_hash = compute_dataset_hash(dataset)
        
        for model_selection, embedding_layer, embedding_type in embedding_combinations:
            
            # Each embedding choice fills its own dataframe, before being merged later
            embedding_df = pd.DataFrame()
            embedding_sorted_path = package_folder / f"embeddings/dataset[{dataset_name}]" / f"model[{model_selection}]" / f"layer[{embedding_layer}]" / f"embedding_type[{embedding_type}]"
            embedding_tensor_path = embedding_sorted_path / "embeddings_tensor.pt"
            metadata_path = embedding_sorted_path / "metadata.pkl"
            
            # Attempt to load embeddings and metadata
            try:
                
                # Load embeddings on CPU for faster access
                embeddings = torch.load(embedding_tensor_path, map_location = "cpu")

                # Load metadata
                with open(metadata_path, "rb") as f:
                    
                    metadata = pickle.load(f)

                # Compare dataset hash
                if metadata.get("dataset_hash") != dataset_hash:
                    
                    raise ValueError("Dataset hash mismatch")

                # If everything is fine, append embeddings to the list
                embeddings_list.append(embeddings)
            
            except (FileNotFoundError, ValueError, EOFError, pickle.UnpicklingError) as e:
                
                print(f"Could not load embeddings for dataset '{dataset_name}' due to {e}. Generating embeddings.")
                
                # Ensure the directory exists
                if not os.path.exists(embedding_sorted_path):
                    
                    os.makedirs(embedding_sorted_path)

                model, embedding_size, tokeniser = setup_model(model_selection, device)
                embeddings = fetch_embeddings(dataset, model, tokeniser, batch_size, device, embedding_type, embedding_layer, embedding_size, n_workers)
                
                # Save embeddings and metadata
                torch.save(embeddings, embedding_tensor_path)
                metadata = {"dataset_hash": dataset_hash}
                
                with open(metadata_path, 'wb') as f:
                    
                    pickle.dump(metadata, f)
                
                # Append embeddings to the list
                embeddings_list.append(embeddings)
        
        dataset.sequence_embeddings = concatenate_embeddings(embeddings_list)
        dataset_dicts[index]["dataset"] = dataset
    
    # Fetch arbitrary dataset to check shape, should all be the same so order does not matter
    first_dataset = dataset_dicts[0]["dataset"]
    embedding_dimensions = first_dataset.sequence_embeddings[0].dim()
    embedding_size = first_dataset.sequence_embeddings[0].shape[embedding_dimensions - 1]

    return dataset_dicts, embedding_size

def compute_dataset_hash(dataset):
    
    sequence_str = ''.join(dataset.aa_seqs)
    
    return hashlib.md5(sequence_str.encode("utf-8")).hexdigest()

def setup_model(model_selection: str, device: str):
    
    """
    This feels like a hacky way to fetch the models but I can't think of a more pythonic way that doesn't break.
    """
    
    match model_selection:
        
        case "AMPLIFY_120M":

            model = AutoModel.from_pretrained("chandar-lab/AMPLIFY_120M", trust_remote_code = True)
            tokeniser = AutoTokenizer.from_pretrained("chandar-lab/AMPLIFY_120M", trust_remote_code = True)

        case "AMPLIFY_120M_base":

            model = AutoModel.from_pretrained("chandar-lab/AMPLIFY_120M_base", trust_remote_code = True)
            tokeniser = AutoTokenizer.from_pretrained("chandar-lab/AMPLIFY_120M_base", trust_remote_code = True)

        case "AMPLIFY_350M":

            model = AutoModel.from_pretrained("chandar-lab/AMPLIFY_350M", trust_remote_code = True)
            tokeniser = AutoTokenizer.from_pretrained("chandar-lab/AMPLIFY_350M", trust_remote_code = True)

        case "AMPLIFY_350M_base":

            model = AutoModel.from_pretrained("chandar-lab/AMPLIFY_350M_base", trust_remote_code = True)
            tokeniser = AutoTokenizer.from_pretrained("chandar-lab/AMPLIFY_350M_base", trust_remote_code = True)

        case "ESMFold":
            
            model = EsmForProteinFolding.from_pretrained("./models/upstream_models/esmfold_3B_v1", output_hidden_states = True)
            tokeniser = EsmTokenizer.from_pretrained("./models/upstream_models/esmfold_3B_v1")
        
        case "ESM2_T6_8M_UR50D":
            
            model = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
            tokeniser = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
        
        case "ESM2_T12_35M_UR50D":
            
            model = EsmModel.from_pretrained("facebook/esm2_t12_35M_UR50D")
            tokeniser = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
        
        case "ESM2_T30_150M_UR50D":
            
            model = EsmModel.from_pretrained("facebook/esm2_t30_150M_UR50D")
            tokeniser = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")
        
        case "ESM2_T33_650M_UR50D":
        
            model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
            tokeniser = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        
        case "ESM2_T36_3B_UR50D":

            model = EsmModel.from_pretrained("facebook/esm2_t36_3B_UR50D")
            tokeniser = AutoTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D")
        
        case "ESM2_T48_15B_UR50D":
            
            model = EsmModel.from_pretrained("facebook/esm2_t48_15B_UR50D")
            tokeniser = AutoTokenizer.from_pretrained("facebook/esm2_t48_15B_UR50D")
        
    model.eval()
    model.to(device)
    embedding_size = model.config.hidden_size
    
    return model, embedding_size, tokeniser

def fetch_embeddings(
    dataset: Dataset,
    model,
    tokeniser,
    batch_size: int,
    device: str,
    embedding_type: str,
    embedding_layer: int,
    embedding_size: int,
    n_workers: int
) -> torch.tensor:
    
    pooled_batch_embeddings = []
    full_embeddings = [None] * len(dataset)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = False, num_workers = n_workers, persistent_workers = True)

    with torch.no_grad():

        for batch_index, batch in enumerate(dataloader):

            sequences = batch["aa_seq"]
            batch_embeddings = fetch_batch_embeddings(sequences, model, tokeniser, device, embedding_layer)

            if embedding_type == "RAW":
                
                pooled_batch_embeddings = []
                
                for sequence_index, seq in enumerate(batch["aa_seq"]):
                    
                    sequence_embeddings = batch_embeddings[sequence_index, 1:len(seq) + 1, :].to(device)
                    pooled_batch_embeddings.append(sequence_embeddings)

            elif embedding_type == "MEAN":
                
                pooled_batch_embeddings = batch_embeddings.mean(dim = 1).float().to(device)

            elif embedding_type == "MAX":

                pooled_batch_embeddings = batch_embeddings.max(dim = 1).values.to(device)

            elif embedding_type == "MIN":

                pooled_batch_embeddings = batch_embeddings.min(dim = 1).values.to(device)

            elif embedding_type == "STD":

                pooled_batch_embeddings = batch_embeddings.std(dim = 1).float().to(device)

            elif embedding_type == "PC1":

                pooled_batch_embeddings = fit_principal_components(batch_embeddings, 1, device).to(device)

            elif embedding_type == "PC2":

                pooled_batch_embeddings = fit_principal_components(batch_embeddings, 2, device).to(device)

            elif embedding_type == "PC3":

                pooled_batch_embeddings = fit_principal_components(batch_embeddings, 3, device).to(device)

            else:

                raise Exception("Invalid embedding type.")

            for index in range(len(sequences)):

                full_embeddings[batch_index * batch_size + index] = pooled_batch_embeddings[index]

    del dataloader
    gc.collect()

    return full_embeddings

def post_process_embeddings(
    dataset_dicts: list[dict],
    normalise_embeddings: bool,
    wildtype_concat: bool,
    wildtype_delta: bool,
    dimensional_reduction_method: str,
    n_desired_dimensions: int
    ):
    
    for index, dataset_dict in enumerate(dataset_dicts):
        
        dataset = dataset_dict["dataset"]
        
        if normalise_embeddings:

            dataset.sequence_embeddings = normalise_embeddings(dataset.sequence_embeddings)
        
        if wildtype_concat:
            
            dataset = concat_wildtype_embeddings(dataset)
        
        elif wildtype_delta:
            
            dataset = find_wildtype_delta(dataset)
        
        dataset_dicts[index]["dataset"] = dataset
    
    if dimensional_reduction_method != "None":
    
        dataset_dicts = apply_dimensional_reduction(dataset_dicts, n_desired_dimensions)
        embedding_size = n_desired_dimensions
    
    return dataset_dicts

def fetch_batch_embeddings(sequences, model, tokeniser, device, embedding_layer):
    
    inputs = tokeniser(sequences, padding = True, truncation = True, return_tensors = "pt", max_length = 1024)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    model = model.to(device)
    output = model(input_ids, attention_mask = attention_mask, output_hidden_states = True)
    batch_embeddings = output.hidden_states[embedding_layer]
    
    return batch_embeddings

def concatenate_embeddings(embeddings_list: list) -> list:

    concatenated_embeddings = []

    for variant_embeddings in list(zip(*embeddings_list)):

        concatenated_embeddings.append(torch.concat(variant_embeddings))

    return concatenated_embeddings

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

def concat_wildtype_embeddings(dataset):
    
    domain_to_wt = {}
    new_embeddingss = []
    
    for index, domain in enumerate(dataset.domain_names):
        
        if dataset.wt_flags[index]:
            
            domain_to_wt[domain] = dataset.sequence_embeddings[index]
    
    for index, domain in enumerate(dataset.domain_names):
        
        concatinated_embedding = torch.cat((dataset.sequence_embeddings[index], domain_to_wt.get(domain)), dim = 0)
        new_embeddings.append(concatinated_embedding)
    
    dataset.sequence_embeddings = new_embeddings
    
    return dataset

def find_wildtype_delta(dataset):
    
    domain_to_wt = {}
    new_embeddings = []
    
    for index, domain in enumerate(dataset.domain_names):
        
        if dataset.wt_flags[index]:
            
            domain_to_wt[domain] = dataset.sequence_embeddings[index]
    
    for index, domain in enumerate(dataset.domain_names):
        
        delta_embedding = dataset.sequence_embeddings[index] - domain_to_wt.get(domain)
        new_embeddings.append(delta_embedding)
    
    dataset.sequence_embeddings = new_embeddings
    
    return dataset