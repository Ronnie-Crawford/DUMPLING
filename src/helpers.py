# Standard modules
import os
import gc
import math
from pathlib import Path
import datetime
import shutil
import hashlib
import random
import json
import os
import multiprocessing

# Third-party modules
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, ConcatDataset

# Local modules
#from splits import read_homology_file

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

def get_n_workers():
    
    if os.environ.get("SLURM_JOB_ID") is not None:
        
        n_workers = 0
        
    else:
        
        n_workers = max(1, multiprocessing.cpu_count() - 1)

    return n_workers

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

def get_benchmarking_results_path(package_folder, train_datasets, test_datasets, scale_size, benchmarking_timestamp, config, base_flag):
    
    if base_flag == True:
        
        results_path = package_folder / "results" / (str(benchmarking_timestamp.year) + "-" + str(benchmarking_timestamp.month) + "-" + str(benchmarking_timestamp.day)) / (str(benchmarking_timestamp.hour) + ":" + str(benchmarking_timestamp.minute) + ":" + str(benchmarking_timestamp.second))
        results_path.mkdir(parents = True, exist_ok = True)
        
        return results_path

    else:
    
        results_path = package_folder / "results" / (str(benchmarking_timestamp.year) + "-" + str(benchmarking_timestamp.month) + "-" + str(benchmarking_timestamp.day)) / (str(benchmarking_timestamp.hour) + ":" + str(benchmarking_timestamp.minute) + ":" + str(benchmarking_timestamp.second)) / str(train_datasets) / str(scale_size) / str(test_datasets)
        results_path.mkdir(parents = True, exist_ok = True)
        new_config_filepath = results_path / "config.json"
        
        with open(new_config_filepath, "w") as new_config_file:
            
            json.dump(config, new_config_file)
        
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
    
    sequence_str = ''.join(dataset.aa_seqs)
    
    return hashlib.md5(sequence_str.encode("utf-8")).hexdigest()

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

#@lru_cache(maxsize = None)
def read_sequence_to_family(homology_tsv_path):
    
    """
    Cache the sequence→family map so we only parse the TSV once.
    """
    
    homology_df = pd.read_csv(homology_tsv_path, sep="\t")
    sequence_to_family_dict = {}
    
    for _, row in homology_df.iterrows():
        
        sequence_to_family_dict[row["sequence"]] = row["sequence_family"]
        
    return sequence_to_family_dict

def remove_homologous_sequences_from_inference(
    dataset_dicts: dict,
    homology_path: str
) -> dict:
    
    """
    Given a dict of ProteinDataset instances—one entry must be
    "spoof_training_dataset" holding the training sequences—this
    function removes from every other dataset any sequences that
    share a homology family with the spoof training set.

    Returns a new dict of filtered ProteinDataset (excluding the spoof).
    """
    
    # Load the mapping sequence → family
    sequence_to_family = read_sequence_to_family(homology_path / "sequence_families.tsv")

    # Identify all families used by the spoof training dataset
    spoof = datasets_dict["spoof_training_dataset"]
    used_families = {sequence_to_family.get(sequence) for sequence in spoof.aa_seqs if sequence_to_family.get(sequence) is not None}

    # Filter each real dataset
    filtered_datasets = {}
    
    for index, dataset_dict in enumerate(dataset_dicts):
        
        name = dataset_dict["unique_key"]
        dataset = dataset_dict["dataset"]
        
        if name == "spoof_training_dataset":
            
            continue

        keep_indices = [index for index, sequence in enumerate(dataset.aa_seqs) if sequence_to_family.get(sequence) not in used_families]
        dataset_dicts[index]["dataset"] = dataset.filter_by_indices(keep_indices)

    return dataset_dicts

def concat_splits(splits: dict, include_test_in_inference: bool) -> dict:
        
    if splits["train"]:
        
        splits["train"] = ConcatDataset(splits["train"])
        
    else:
        
        splits["train"] = None

    if splits["validation"]:
        
        splits["validation"] = ConcatDataset(splits["validation"])
        
    else:
        
        splits["validation"] = None

    if include_test_in_inference == True:

        if splits["test"] and splits["inference"]:
            
            splits["test_inference"] = ConcatDataset(splits["test"] + splits["inference"])
        
        elif splits["test"] and not splits["inference"]:
            
            splits["test_inference"] = ConcatDataset(splits["test"])
            
        elif not splits["test"] and splits["inference"]:
            
            splits["test_inference"] = ConcatDataset(splits["inference"])
        
        else:
            
            splits["test_inference"] = None
    
    else:
        
        if splits["inference"]:
            
            splits["test_inference"] = ConcatDataset(splits["inference"])
        
        else:
            
            splits["test_inference"] = None
    
    splits["test"] = None
    splits["inference"] = None
    
    return splits

def get_mutants(sequence, vocab, search_breadth, search_depth = 1):

    mutant_sequences = [sequence]

    for _search in range(search_depth):

        mutants_found_this_layer = []

        for sequence in mutant_sequences:

            mutants_found_this_layer.extend(get_adjacent_mutants(sequence, vocab, search_breadth))

        mutant_sequences.extend(mutants_found_this_layer)

    return mutant_sequences

def get_adjacent_mutants(sequence, vocab, search_breadth):

    mutant_sequences = []

    for _search in range(search_breadth):

        mutant_sequences.append(str(mutate_sequence(str(sequence), list(vocab.keys()))))

    return mutant_sequences

def mutate_sequence(sequence, vocab):

    index = random.randint(0, len(sequence) - 1)
    mutation = random.choice(["deletion", "substitution", "insertion"])

    if mutation == "deletion":

        sequence = sequence[:index] + sequence[index + 1:]

    elif mutation == "substitution":

        new_token = random.choice(vocab)
        sequence = sequence[:index] + new_token + sequence[index + 1:]

    elif mutation == "insertion":

        new_token = random.choice(vocab)
        sequence = sequence[:index] + new_token + sequence[index:]

    return sequence

def check_one_wt_per_domain(dataset):
    
    for domain_name in set(dataset.domain_names):
        
        wts_in_domain = 0
        
        for index, query_domain in enumerate(dataset.domain_names):
            
            if query_domain == domain_name and dataset.wt_flags[index]:

                wts_in_domain += 1
        
        if wts_in_domain != 1:
            
            raise ValueError(f"Domain '{domain_name}' has {wts_in_domain} WT sequences.")
    
    print("Checked WTs successfully.")

def filter_domains_with_one_wt(dataset):
    
    domain_total = {}
    domain_wt = {}
    
    for i, domain in enumerate(dataset.domain_names):
        
        domain_total[domain] = domain_total.get(domain, 0) + 1
        
        if dataset.wt_flags[i]:
            
            domain_wt[domain] = domain_wt.get(domain, 0) + 1

    valid_domains = {domain for domain in domain_total if domain_wt.get(domain, 0) == 1}
    dropped_domains = set(domain_total.keys()) - valid_domains

    print(f"Total domains: {len(domain_total)}")
    print(f"Valid domains (exactly one WT): {len(valid_domains)}")
    print(f"Dropped domains: {len(dropped_domains)}")

    # Build a list of indices to keep (only rows belonging to valid domains)
    keep_indices = [i for i, domain in enumerate(dataset.domain_names) if domain in valid_domains]
    
    # Use the dataset's filtering method to return a new, filtered dataset.
    return dataset.filter_by_indices(keep_indices)

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

def handle_setup(
    downstream_models_dict: dict,
    ) -> tuple[list, list, list, list, str, bool]:
    
    # Fetch settings from config
    rnn_type = downstream_models[0].split("_")[0]
    bidirectional = False
    
    if "_" in downstream_models[0]:
        
        parts = downstream_models[0].split("_")
        bidirectional = (len(parts) > 1 and parts[1] == "BIDIRECTIONAL")
    
    return rnn_type, bidirectional
    
    