# Standard modules
import copy
import pickle
import os

# Third-party modules
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import Subset, ConcatDataset, random_split

# Local modules
from config_loader import config

def handle_splits_flag(splits: str, datasets_dict: dict, homology_path) -> tuple[dict, dict, dict]:

    """
    Decides how to split data based on the value passed via the splits flag.

    Parameters:
        - splits (str): The value of the splits flag, valid values are [homologous-aware] or [random].
        - datasets (list): A list of the datasets for which splits will be assigned.

    Returns:
        - training_split (dict): A dictionary containing the domain families assigned to the training split.
        - validation_split (dict): A dictionary containing the domain families assigned to the validation split.
        - testing_split (dict): A dictionary containing the domain families assigned to the testing split.
    """

    training_split, validation_split, testing_split = {}, {}, {}

    if splits == "homologous-aware":

        splits = get_splits(datasets_dict, homology_path)

    elif splits == "random":

        training_split, validation_split, testing_split = random_split(datasets[0], [config["SPLITS"]["TRAINING_SIZE"], config["SPLITS"]["VALIDATION_SIZE"], config["SPLITS"]["TESTING_SIZE"]])

    elif splits == "thermompnn":
        
        splits = get_thermompnn_splits(datasets_dict["train"].values(), package_folder)

    else:
        raise ValueError("Value for splits flag not recognized.")

    return splits

def get_splits(datasets_dict: dict, homology_path) -> tuple[Dataset, Dataset, Dataset]:
    
    # Read homology families
    homology_families_file_path = homology_path / "sequence_families.tsv"
    homology_family_dict = read_homology_file(homology_families_file_path)
    
    # Assign splits globally
    split_ratios = {
        'train': config["SPLITS"]["TRAINING_SIZE"],
        'validation': config["SPLITS"]["VALIDATION_SIZE"],
        'test': config["SPLITS"]["TESTING_SIZE"],
    }
    homology_splits = assign_splits(datasets_dict["train"], homology_family_dict, split_ratios)
    
    # Subset datasets based on assigned splits
    splits = {"train": [], "validation": [], "test": []}
    
    for name, dataset in datasets_dict["train"].items():
        
        dataset_splits = subset_dataset(dataset, homology_splits)
        splits["train"].append(dataset_splits["train"])
        splits["validation"].append(dataset_splits["validation"])
        splits["test"].append(dataset_splits["test"])
    
    return splits

def assign_splits(
    datasets: dict,
    homology_family_dict: dict,
    split_ratios: dict,
    random_state: int = config["TRAINING_PARAMETERS"]["RANDOM_STATE"]
    ) -> dict:
    
    """
    Assign homology families to splits globally across all datasets.

    Parameters:
        - datasets (dict): Dictionary mapping dataset names to dataset objects.
        - homology_family_dict (dict): Dictionary mapping homology family IDs to sequences.
        - split_ratios (dict): Desired split ratios with keys 'train', 'validation', 'test'.
        - random_state (int): Random seed for reproducibility.

    Returns:
        - homology_splits (dict): Dictionary mapping sequences to assigned splits.
    """
    
    np.random.seed(random_state)

    dataset_names = list(datasets.keys())
    assigned_sequences = {name: {'train': 0, 'validation': 0, 'test': 0} for name in dataset_names}
    desired_sequences = calculate_desired_sequences(datasets, split_ratios)
    homology_splits = {}

    # Shuffle homology families
    homology_family_ids = list(homology_family_dict.keys())
    np.random.shuffle(homology_family_ids)
    splits_needing_assignment = [split for split, ratio in split_ratios.items() if ratio > 0]

    if len(homology_family_ids) < len(splits_needing_assignment):
        
        raise Exception("Not enough homology families to assign at least one to each split.")

    # Assign one homology family to each split that needs assignment
    initial_family_assignments = {}
    
    for split in splits_needing_assignment:
        
        # Pop a homology family from the list
        family_id = homology_family_ids.pop()
        sequences = homology_family_dict[family_id]
        sequences_per_dataset, sequence_to_dataset = count_sequences_per_dataset(sequences, datasets)

        # Assign the family to the split
        for seq in sequences:
            
            homology_splits[seq] = split

        for name in dataset_names:
            
            assigned_sequences[name][split] += sequences_per_dataset[name]

        # Keep track of assigned families
        initial_family_assignments[family_id] = split

    # Now proceed with the remaining homology families
    for family_id in homology_family_ids:
        
        sequences = homology_family_dict[family_id]
        sequences_per_dataset, sequence_to_dataset = count_sequences_per_dataset(sequences, datasets)

        # Evaluate assigning to each split
        best_split = select_best_split(assigned_sequences, desired_sequences, sequences_per_dataset)

        # Assign the family to the best split
        for seq in sequences:
            
            homology_splits[seq] = best_split

        for name in dataset_names:
            
            assigned_sequences[name][best_split] += sequences_per_dataset[name]

    return homology_splits

def calculate_desired_sequences(datasets: dict, split_ratios: dict) -> dict:
    
    """
    Calculates the desired number of sequences per split per dataset.

    Parameters:
        - datasets (dict): Dictionary mapping dataset names to dataset objects.
        - split_ratios (dict): Desired split ratios.

    Returns:
        - desired_sequences (dict): Desired sequence counts per split per dataset.
    """
    
    desired_sequences = {}
    
    for name, dataset in datasets.items():
        
        total = len(dataset)
        n_train = int(total * split_ratios['train'])
        n_val = int(total * split_ratios['validation'])
        n_test = total - n_train - n_val
        desired_sequences[name] = {'train': n_train, 'validation': n_val, 'test': n_test}
        
    return desired_sequences

def count_sequences_per_dataset(sequences: list, datasets: dict) -> tuple[dict, dict]:
    
    """
    Counts the number of sequences per dataset for a given homology family.

    Parameters:
        - sequences (list): List of sequences in the homology family.
        - datasets (dict): Dictionary mapping dataset names to dataset objects.

    Returns:
        - sequences_per_dataset (dict): Counts per dataset.
        - sequence_to_dataset (dict): Mapping of sequence to dataset name.
    """
    
    sequences_per_dataset = {name: 0 for name in datasets.keys()}
    sequence_to_dataset = {}
    
    for seq in sequences:
        
        for name, dataset in datasets.items():
            
            if seq in dataset.variant_aa_seqs:
                
                sequences_per_dataset[name] += 1
                sequence_to_dataset[seq] = name
                
                break
            
    return sequences_per_dataset, sequence_to_dataset

def select_best_split(assigned_sequences: dict, desired_sequences: dict, sequences_per_dataset: dict) -> str:
    
    """
    Selects the best split for a homology family based on minimizing total deviation.

    Parameters:
        - assigned_sequences (dict): Current assigned sequence counts per dataset and split.
        - desired_sequences (dict): Desired sequence counts per dataset and split.
        - sequences_per_dataset (dict): Number of sequences per dataset for the homology family.

    Returns:
        - best_split (str): The selected split ('train', 'validation', or 'test').
    """
    
    min_total_deviation = float('inf')
    best_split = None
    
    for split in ['train', 'validation', 'test']:
        
        projected_assigned_sequences = copy.deepcopy(assigned_sequences)
        
        for name in assigned_sequences.keys():
            
            projected_assigned_sequences[name][split] += sequences_per_dataset[name]
            
        # Calculate total deviation from desired sequences
        total_deviation = sum(
            abs(projected_assigned_sequences[name][s] - desired_sequences[name][s])
            for name in assigned_sequences.keys() for s in ['train', 'validation', 'test']
        )
        
        # Select split with minimum total deviation
        if total_deviation < min_total_deviation:
            
            min_total_deviation = total_deviation
            best_split = split
            
    return best_split

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

def subset_dataset(dataset: Dataset, homology_splits: dict) -> dict:
    
    """
    Subsets the dataset into train, validation, and test subsets based on homology splits.

    Parameters:
        - dataset (Dataset): The dataset to be split.
        - homology_splits (dict): Dictionary mapping sequences to assigned splits.

    Returns:
        - splits_dict (dict): Dictionary with keys 'train', 'validation', 'test' containing Subsets.
    """
    
    split_indices = {'train': [], 'validation': [], 'test': []}

    for idx, item in enumerate(dataset):
        
        sequence = item["variant_aa_seq"]
        assigned_split = homology_splits.get(sequence, None)
        split_indices[assigned_split].append(idx)
    
    splits_dict = {
        "train":  dataset.filter_by_indices(split_indices["train"]),
        "validation":  dataset.filter_by_indices(split_indices["validation"]),
        "test": dataset.filter_by_indices(split_indices["test"])
    }

    return splits_dict

def get_thermompnn_splits(training_datasets: list, package_folder) -> dict:
    
    """
    Reads the ThermoMPNN splits and assigns sequences to splits.

    Parameters:
        - training_datasets (list): List containing the Rocklin dataset.
        - package_folder: Path to the package folder.

    Returns:
        - splits (dict): Dictionary with keys 'train', 'validation', 'test' containing lists of dataset subsets.
    """

    # Assuming the splits file is located in a known path within the package folder
    splits_file_path = package_folder / "splits" / "mega_splits.pkl"

    with open(splits_file_path, "rb") as f:
        
        splits_data = pickle.load(f)

    # Extract the splits
    train_domains = splits_data.get("train", [])
    val_domains = splits_data.get("val", [])
    test_domains = splits_data.get("test", [])

    # Prepare sets for quick lookup
    train_set = set(train_domains)
    val_set = set(val_domains)
    test_set = set(test_domains)

    # Initialize split indices
    training_indices = []
    validation_indices = []
    testing_indices = []

    # Assuming only Rocklin dataset is used
    rocklin_dataset = training_datasets[0]

    # Map domain names in your dataset to splits
    for idx in range(len(rocklin_dataset)):
        
        sample = rocklin_dataset[idx]
        domain_name = sample["domain_name"]
        domain_id = extract_domain_id(domain_name)

        if domain_id in train_set:
            
            training_indices.append(idx)
            
        elif domain_id in val_set:
            
            validation_indices.append(idx)
            
        elif domain_id in test_set:
            
            testing_indices.append(idx)
            
        else:
            
            print(f"Domain ID: {domain_id} not found in train, val, or test splits.")

    # Create splits_dict similar to subset_dataset
    splits = {"train": [], "validation": [], "test": []}

    # Assuming your dataset has a method filter_by_indices
    splits["train"].append(rocklin_dataset.filter_by_indices(training_indices))
    splits["validation"].append(rocklin_dataset.filter_by_indices(validation_indices))
    splits["test"].append(rocklin_dataset.filter_by_indices(testing_indices))

    return splits
