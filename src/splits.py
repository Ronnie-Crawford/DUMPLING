# Standard modules
import copy
import pickle
import os
from pathlib import Path

# Third-party modules
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset, ConcatDataset, random_split, DataLoader

def handle_splits(
    splits_priority_choice: str,
    splits_method_choice: str,
    exclude_wildtype_from_inference: bool,
    batch_size: int,
    n_workers: int,
    dataset_dicts: list,
    datasets_splits_dict: dict,
    homology_path,
    results_path
    ):
    
    # Read homology families
    print("Reading homology families")
    homology_family_file_path = homology_path / "sequence_families.tsv"
    homology_family_dict = read_homology_file(homology_family_file_path)
    
    # Prepare global split tracking
    family_to_split_assignment = {} # {family_id: "train"/"validation"/"test"}

    # Compute desired split sizes for each dataset
    print("Calculating desired split sizes")
    desired_split_sizes = calculate_desired_split_sizes(dataset_dicts, datasets_splits_dict)

    # Assign families globally
    print("Assigning families")
    family_to_split_assignment = assign_families_to_splits(
        dataset_dicts,
        homology_family_dict,
        desired_split_sizes,
        splits_priority_choice,
        datasets_splits_dict
    )
    
    # Get indices per dataset and per split based on assignment
    print("Getting indicies per dataset")
    datasets_split_indices = get_datasets_split_indices(
        dataset_dicts,
        homology_family_dict,
        family_to_split_assignment
    )
    
    # Filter datasets
    print("Filtering datasets")
    split_datasets = construct_filtered_datasets(dataset_dicts, datasets_split_indices, datasets_splits_dict)

    # Remove wildtype sequences from inference splits
    if exclude_wildtype_from_inference:
        
        split_datasets["VALIDATION"] = remove_wt_from_split(split_datasets["VALIDATION"])
        split_datasets["TEST"] = remove_wt_from_split(split_datasets["TEST"])

    # Save splits
    save_training_sequences(results_path, family_to_split_assignment, datasets_split_indices, dataset_dicts)

    # Concatenate datasets into splits
    final_splits = {
        split: ConcatDataset(split_datasets[split])
        for split in get_nonzero_splits(datasets_splits_dict)
    }

    # Load into dataloaders
    dataloaders_dict = splits_to_loaders(final_splits, batch_size, n_workers)

    return dataloaders_dict

def read_homology_file(homology_file: str) -> dict:
    
    """
    Reads the homology file and creates a dictionary mapping family IDs to sequences.

    Parameters:
        - homology_file (str): Path to the homology file.

    Returns:
        - family_dict (dict): Dictionary mapping family IDs to lists of sequences.
    """
    
    homology_df = pd.read_csv(homology_file, sep="\t")
    family_dict = {}

    for _, row in homology_df.iterrows():
        
        family_id = row["sequence_family"]
        sequence = row["sequence"]
        
        if family_id not in family_dict:
            
            family_dict[family_id] = []
            
        family_dict[family_id].append(sequence)

    return family_dict

def get_nonzero_splits(datasets_splits_dict):
    
    splits = {"TRAIN", "VALIDATION", "TEST"}
    nonzero_splits = set()
    
    for split in splits:
        
        if any(splits_dict.get(split, 0) > 0 for splits_dict in datasets_splits_dict.values()):
            
            nonzero_splits.add(split)

    return list(nonzero_splits)


def calculate_desired_split_sizes(dataset_dicts, datasets_splits_dict):
    
    desired_split_sizes = {}

    for dataset_dict in dataset_dicts:
        
        total_sequences = len(dataset_dict["dataset"])
        splits = datasets_splits_dict[dataset_dict["unique_key"]]
        train_count = int(total_sequences * splits.get("TRAIN", 0))
        validation_count = int(total_sequences * splits.get("VALIDATION", 0))
        test_count = total_sequences - (train_count + validation_count)

        desired_split_sizes[dataset_dict["unique_key"]] = {
            "TRAIN": train_count,
            "VALIDATION": validation_count,
            "TEST": test_count
        }
        
    return desired_split_sizes

def assign_families_to_splits(
    dataset_dicts, 
    homology_family_dict, 
    desired_split_sizes, 
    splits_priority_choice, 
    datasets_splits_dict
):

    family_to_split_assignment = {}
    split_options = get_nonzero_splits(datasets_splits_dict)
    dataset_names = [dataset_dict["unique_key"] for dataset_dict in dataset_dicts]
    split_counts_per_dataset = {name: {split: 0 for split in split_options} for name in dataset_names}

    # Precompute fast sequence lookup sets
    #datasets_sequence_sets = {f"{dataset_dict["dataset_name"]}-{dataset_dict["label"]}": set(dataset_dict["dataset"].aa_seqs) for dataset_dict in dataset_dicts}

    # Precompute family-to-dataset sequence counts
    #family_dataset_counts = {
    #    family_id: {
    #        dataset_name: len(set(family_seqs) & datasets_sequence_sets[dataset_name])
    #        for dataset_name in datasets_dict
    #    }
    #    for family_id, family_seqs in homology_family_dict.items()
    #}
    
    # Precompute fast sequence lookup sets
    sequence_sets = []
    
    for dataset_dict in dataset_dicts:
        
        key = dataset_dict["unique_key"]
        sequence_sets.append((key, set(dataset_dict["dataset"].aa_seqs)))

    # Precompute family-to-dataset-label-group sequence counts
    family_dataset_counts = {}
    
    for family_id, family_seqs in homology_family_dict.items():
        
        family_set = set(family_seqs)
        counts = {}
        
        for key, sequence_set in sequence_sets:
            
            counts[key] = len(family_set & sequence_set)
            
        family_dataset_counts[family_id] = counts

    all_family_ids = list(homology_family_dict.keys())
    np.random.shuffle(all_family_ids)

    # Assign at least one family globally to each nonzero split
    for split in split_options:
        
        for family_id in all_family_ids:
            
            if family_id in family_to_split_assignment:
                
                continue
            
            family_to_split_assignment[family_id] = split
            update_split_counts(family_id, family_dataset_counts, split_counts_per_dataset, split)
            break

    # Assign remaining families efficiently
    for family_id in all_family_ids:
        
        if family_id in family_to_split_assignment:
            
            continue

        best_split = select_best_split(
            family_id,
            family_dataset_counts,
            split_counts_per_dataset,
            desired_split_sizes,
            splits_priority_choice,
            dataset_dicts,
            datasets_splits_dict
        )

        family_to_split_assignment[family_id] = best_split
        update_split_counts(family_id, family_dataset_counts, split_counts_per_dataset, best_split)

    return family_to_split_assignment

def update_split_counts(family_id, family_dataset_counts, split_counts_per_dataset, split):
    
    for dataset_name in split_counts_per_dataset.keys():
        
        split_counts_per_dataset[dataset_name][split] += family_dataset_counts[family_id][dataset_name]

def select_best_split(
    family_id: int,
    family_dataset_counts,
    split_counts_per_dataset,
    desired_split_sizes,
    splits_priority_choice,
    dataset_dicts: list,
    datasets_splits_dict
):
    min_total_normalised_deviation = float("inf")
    best_split = None

    for split in get_nonzero_splits(datasets_splits_dict):
        
        total_normalised_deviation = 0

        for dataset_dict in dataset_dicts:
            
            dataset_name = dataset_dict["unique_key"]
            dataset = dataset_dict["dataset"]
            
            dataset_size = len(dataset)
            num_sequences_in_family = family_dataset_counts[family_id][dataset_name]
            projected_count = split_counts_per_dataset[dataset_name][split] + num_sequences_in_family
            desired_count = desired_split_sizes[dataset_name][split]

            deviation = abs(desired_count - projected_count)
            normalised_deviation = deviation / dataset_size
            total_normalised_deviation += normalised_deviation

        if total_normalised_deviation < min_total_normalised_deviation:
            
            min_total_normalised_deviation = total_normalised_deviation
            best_split = split

    return best_split

def get_datasets_split_indices(
    dataset_dicts: list,
    homology_family_dict: dict,
    family_to_split_assignment
    ):
    
    datasets_split_indices = {dataset_dict["unique_key"]: {"TRAIN": [], "VALIDATION": [], "TEST": []} for dataset_dict in dataset_dicts}
    sequence_to_family = {sequence: family_id for family_id, sequences in homology_family_dict.items() for sequence in sequences}

    for dataset_dict in dataset_dicts:
        
        dataset_name = dataset_dict["unique_key"]
        dataset = dataset_dict["dataset"]
        
        for index, sequence in enumerate(dataset.aa_seqs):
            
            family_id = sequence_to_family.get(sequence)
            assigned_split = family_to_split_assignment.get(family_id)
            
            if assigned_split:
                
                datasets_split_indices[dataset_name][assigned_split].append(index)

    return datasets_split_indices

def construct_filtered_datasets(
    dataset_dicts: list,
    datasets_split_indices,
    datasets_splits_dict: dict
    ):
    
    split_datasets = {"TRAIN": [], "VALIDATION": [], "TEST": []}
    
    for dataset_name, indices_dict in datasets_split_indices.items():
        
        for split in get_nonzero_splits(datasets_splits_dict):
            
            target_dataset_dict = [dataset_dict for dataset_dict in dataset_dicts if dataset_dict["unique_key"] == dataset_name][0]
            dataset_subset = target_dataset_dict["dataset"].filter_by_indices(indices_dict[split])
            split_datasets[split].append(dataset_subset)
            
    return split_datasets

def remove_wt_from_split(split: list) -> list:
    
    if split is not []:
        
        filtered_split = []
        
        for dataset in split:
    
            keep_indices = [index for index, wt_flag in enumerate(dataset.wt_flags) if not wt_flag]
            filtered_dataset = dataset.filter_by_indices(keep_indices)
            filtered_split.append(filtered_dataset)
        
        return filtered_split
    
    else:
        
        return []

def splits_to_loaders(
    splits: dict,
    batch_size: int,
    n_workers: int
) -> dict:
    
    """
    Convert a dict of dataset splits into PyTorch DataLoader objects.
    """
    
    loaders = {}
            
    for split_name, dataset in splits.items():
        
        if dataset is None or len(dataset) == 0:

            loaders[split_name] = None
            continue

        shuffle_flag = split_name in ["TRAIN", "VALIDATION"]
        drop_last_flag = False

        if split_name == "TRAIN":
            
            # only drop the last batch if it's size 1
            drop_last_flag = (len(dataset) % batch_size == 1)

        loaders[split_name] = DataLoader(
            dataset,
            batch_size = batch_size,
            shuffle = shuffle_flag,
            drop_last = drop_last_flag,
            num_workers = n_workers,
            collate_fn = collate_fn
        )

    return loaders

def collate_fn(batch):
    
    # Extract sequences and compute lengths
    sequences = [item["aa_seq"] for item in batch]
    sequence_embeddings = [item["sequence_embedding"] for item in batch]
    lengths = torch.tensor([len(sequence) for sequence in sequences])
    
    # Pad sequences to the same length
    padded_sequences = torch.nn.utils.rnn.pad_sequence(sequence_embeddings, batch_first = True)
    
    # Collect other batch elements
    batch_dict = {
        "sequence_embedding": padded_sequences,
        "length": lengths,
    }
    
    for key in batch[0]:
        
        if key not in ["sequence_embedding"]:
            
            values = [item[key] for item in batch]
            
            if isinstance(values[0], torch.Tensor):
                
                batch_dict[key] = torch.stack(values)
                
            elif isinstance(values[0], (float, int, bool)):
                
                batch_dict[key] = torch.tensor(values)
                
            elif isinstance(values[0], str):
                
                batch_dict[key] = values
                
            else:
                
                # Handle other data types if necessary
                batch_dict[key] = values
                
    return batch_dict

def save_training_sequences(
    save_directory: str,
    family_to_split_assignment: dict,
    datasets_split_indices: dict,
    dataset_dicts: list
) -> None:
    
    """
    Save splits as pickle files.

    - family_to_split_assignment: { family_id: split_name }
    - datasets_split_indices:     { dataset_name: { 'TRAIN': [...], ... } }
    - datasets_dict:              { dataset_name: ProteinDataset }

    This writes:
      - save_directory/family_to_split_assignment.pkl
      - save_directory/train_sequence_keys.pkl
    """

    # Build and save list of training sequence keys
    training_keys = set()
    
    for dataset_name, split_indices in datasets_split_indices.items():
        
        for idx in split_indices.get("TRAIN", []):
            
            # Collect the sequence at that index
            target_dataset_dict = [dataset_dict for dataset_dict in dataset_dicts if dataset_dict["unique_key"] == dataset_name][0]
            training_keys.add(target_dataset_dict["dataset"].aa_seqs[idx])

    keys_path = save_directory / "train_sequences.pkl"
    
    with open(keys_path, "wb") as f:
        
        pickle.dump(list(training_keys), f)

def load_training_sequences(save_directory: str) -> list[str]:
    
    """
    Loads the pickled list of training sequences from disk.

    Parameters:
      - save_directory (str): Path to the folder containing train_sequences.pkl

    Returns:
      - List[str]: The list of amino acid sequence strings saved earlier.
    """
    
    path = Path(save_directory) / "train_sequences.pkl"
    
    with open(path, "rb") as f:
        
        training_sequences = pickle.load(f)
        
    return training_sequences