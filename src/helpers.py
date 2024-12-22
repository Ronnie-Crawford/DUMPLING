# Standard modules
import os
import gc
import math
from pathlib import Path
import datetime
import shutil
import hashlib

# Third-party modules
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, ConcatDataset

# Local modules
from splits import read_homology_file

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

def setup_folders() -> Path:
    
    package_folder = Path(__file__).resolve().parent.parent
    directories = ["embeddings", "homology", "models", "results", "splits"]
    
    for directory in directories:
        
        directory_path = package_folder / directory
        directory_path.mkdir(parents = True, exist_ok = True)
    
    print("Directories set up okay.")
    
    return package_folder

def get_results_path(package_folder):
    
    timestamp = datetime.datetime.now()
    results_path = package_folder / "results" / (str(timestamp.year) + "-" + str(timestamp.month) + "-" + str(timestamp.day)) / (str(timestamp.hour) + ":" + str(timestamp.minute) + ":" + str(timestamp.second))
    results_path.mkdir(parents = True, exist_ok = True)
    shutil.copy((package_folder / "config.json"), (results_path / "config.json"))
    
    return results_path

def get_homology_path(package_folder, all_dataset_names):
    
    datasets_key = "-".join(sorted(all_dataset_names))
    homology_folder_path = package_folder / "homology" / f"homology[{datasets_key}]"
    
    return homology_folder_path

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

def compute_dataset_hash(dataset):
    
    sequence_str = ''.join(dataset.variant_aa_seqs)
    
    return hashlib.md5(sequence_str.encode('utf-8')).hexdigest()

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

def collate_fn(batch):
    
    # Extract sequences and compute lengths
    sequences = [item["variant_aa_seq"] for item in batch]
    sequence_representations = [item["sequence_representation"] for item in batch]
    lengths = torch.tensor([len(sequence) for sequence in sequences])
    
    # Pad sequences to the same length
    padded_sequences = torch.nn.utils.rnn.pad_sequence(sequence_representations, batch_first = True)
    
    # Collect other batch elements
    batch_dict = {
        "sequence_representation": padded_sequences,
        "length": lengths,
    }
    
    for key in batch[0]:
        
        if key not in ['sequence_representation']:
            
            values = [item[key] for item in batch]
            
            if isinstance(values[0], torch.Tensor):
                
                batch_dict[key] = torch.stack(values)
                
            elif isinstance(values[0], (float, int, bool)):
                
                batch_dict[key] = torch.tensor(values)
                
            elif isinstance(values[0], str):
                
                batch_dict[key] = values  # Keep as list of strings
                
            else:
                
                # Handle other data types if necessary
                batch_dict[key] = values  # Keep as is or implement custom handling
                
    return batch_dict

def old_collate_fn(batch):
    
    # Extract sequences and compute lengths
    sequences = [item["sequence_representation"] for item in batch]
    lengths = torch.tensor([seq.size(0) for seq in sequences])  # Assuming sequences are tensors of shape (seq_length, embedding_dim)
    
    # Pad sequences to the same length
    padded_sequences = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    
    # Collect other batch elements
    batch_dict = {
        'sequence_representation': padded_sequences,
        'length': lengths,
    }
    
    for key in batch[0]:
        
        if key not in ['sequence_representation']:
            
            values = [item[key] for item in batch]
            
            if isinstance(values[0], torch.Tensor):
                
                batch_dict[key] = torch.stack(values)
                
            elif isinstance(values[0], (float, int, bool)):
                
                batch_dict[key] = torch.tensor(values)
                
            elif isinstance(values[0], str):
                
                batch_dict[key] = values  # Keep as list of strings
                
            else:
                
                # Handle other data types if necessary
                batch_dict[key] = values  # Keep as is or implement custom handling
                
    return batch_dict

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

def remove_homologous_sequences_from_inference(datasets_dict, homology_path):
    
    homology_family_dict = read_homology_file(homology_path / "sequence_families.tsv")
    sequence_to_family_dict = {}

    for family_key, sequence_list in homology_family_dict.items():
        
        for sequence in sequence_list:
            
            sequence_to_family_dict[sequence] = family_key
    
    used_domain_families = []
    
    for dataset in datasets_dict["train"].values():
    
        for sequence in dataset.variant_aa_seqs:
            
            family = sequence_to_family_dict.get(sequence)
            
            if family:
                
                used_domain_families.append(family)
    
    filtered_datasets = []
    
    for dataset in datasets_dict["inference_only"].values():
        
        keep_indices = []
        
        for index in range(len(dataset)):
            
            protein = dataset[index]
            sequence = protein["variant_aa_seq"]
            inference_sequence_family = sequence_to_family_dict.get(sequence)
            
            if inference_sequence_family not in used_domain_families:
                
                keep_indices.append(index)
        
        filtered_dataset = dataset.filter_by_indices(keep_indices)
        filtered_datasets.append(filtered_dataset)
    
    return filtered_datasets

def concat_splits(splits: dict) -> dict:
    
    if splits["train"]:
        
        splits["train"] = ConcatDataset(splits["train"])
        
    else:
        
        splits["train"] = None

    if splits["validation"]:
        
        splits["validation"] = ConcatDataset(splits["validation"])
        
    else:
        
        splits["validation"] = None

    if splits["test"] and splits["inference"]:
        
        splits["testing_inference"] = ConcatDataset(splits["test"] + splits["inference"])
    
    elif splits["test"] and not splits["inference"]:
        
        splits["test_inference"] = ConcatDataset(splits["test"])
        
    elif not splits["test"] and splits["inference"]:
        
        splits["test_inference"] = ConcatDataset(splits["inference"])
    
    else:
        
        splits["test_inference"] = None
    
    splits["test"] = None
    splits["inference"] = None
    
    return splits

def splits_to_loaders(splits: dict, batch_size: int, n_workers: int) -> dict:
    
    if splits["train"] is not None:
        
        training_loader = DataLoader(
            splits["train"],
            batch_size = batch_size,
            shuffle = True,
            num_workers = n_workers,
            collate_fn=collate_fn
            )
        
    else:
        
        training_loader = None
        
    if len(splits["validation"]) > 0:
        
        validation_loader = DataLoader(
            splits["validation"],
            batch_size = batch_size,
            shuffle = True,
            num_workers = n_workers,
            collate_fn=collate_fn
            )
    
    else:
        
        validation_loader = None
    
    if splits["test_inference"] is not None:
        
        testing_inference_loader = DataLoader(
            splits["test_inference"],
            batch_size = batch_size,
            shuffle = False,
            num_workers = n_workers,
            collate_fn=collate_fn
            )

    else:
        
        testing_loader = None

    dataloaders = {
        "train": training_loader,
        "validation": validation_loader,
        "test_inference": testing_inference_loader
    }

    return dataloaders