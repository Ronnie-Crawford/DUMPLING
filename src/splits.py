# Third-party modules
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import Subset, ConcatDataset, random_split

# Local modules
from config_loader import config

def handle_splits_flag(splits: str, datasets: list) -> tuple[dict, dict, dict]:

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

        training_split, validation_split, testing_split = get_splits(datasets)

    elif (splits == "random") or (splits == None):

        training_split, validation_split, testing_split = random_split(datasets[0], [config["SPLITS"]["TRAINING_SIZE"], config["SPLITS"]["VALIDATION_SIZE"], config["SPLITS"]["TESTING_SIZE"]])

    else:

        raise Exception("Value for splits flag not recognised.")

    return training_split, validation_split, testing_split

def get_splits(datasets: list) -> tuple[Dataset, Dataset, Dataset]:

    combined_training_splits = []
    combined_validation_splits = []
    combined_testing_splits = []

    for index, dataset_name in enumerate(config["DATASETS_IN_USE"]):

        homology_family_dict = read_homology_file(config["DATASETS"][dataset_name]["HOMOLOGOUS_FAMILIES_PATH"])
        splits_dict = assign_splits(datasets[index], homology_family_dict)
        splits = subset_dataset(datasets[index], splits_dict)

        combined_training_splits.append(splits["training_split"])
        combined_validation_splits.append(splits["validation_split"])
        combined_testing_splits.append(splits["testing_split"])

    return (
            ConcatDataset(combined_training_splits),
            ConcatDataset(combined_validation_splits),
            ConcatDataset(combined_testing_splits)
        )

def read_homology_file(homology_file: str) -> dict:

    """
    Reads the homology file and assigns domains to a dictionary of domain families.

    Parameters:
        - homology_file (str): Path to the homology file.

    Returns:
        - family_dict (dict): Dictionary mapping domains to their domain families.
    """

    homology_df = pd.read_csv(homology_file, sep="\t")
    family_dict = {}

    for _index, row in homology_df.iterrows():

        if row["domain_family"] not in family_dict:

            family_dict[row["domain_family"]] = []

        family_dict[row["domain_family"]].append(row["name"])

    return family_dict

def assign_splits(dataset: Dataset, family_dict: dict, random_state: int = config["TRAINING_PARAMETERS"]["RANDOM_STATE"]) -> dict:

    """
    Assigns domains to splits while ensuring homologous domains are in the same split.
    Attempts to balance splits based on number of variants in each domain family.

    Parameters:
        - dataset (DataFrame): The input dataset.
        - family_dict (dict): Dictionary mapping families to a list of domains.
        - random_state (int): Seed for the randomness.

    Returns:
        - train_domains (list): Lists of domains for training.
        - val_domains (list): Lists of domains for validation.
        - test_domains (list): Lists of domains for testing.
    """

    all_families = list(family_dict.keys())
    np.random.seed(random_state)
    np.random.shuffle(all_families)

    train_domains = []
    val_domains = []
    test_domains = []

    n_total_variants = len(dataset)
    n_train_variants = int(n_total_variants * config["SPLITS"]["TRAINING_SIZE"])
    n_val_variants = int(n_total_variants * config["SPLITS"]["VALIDATION_SIZE"])
    n_test_variants = n_total_variants - (n_train_variants + n_val_variants)

    current_train_size = 0
    current_val_size = 0
    current_test_size = 0

    # Ensure each split has at least one domain family assigned
    family = all_families[0]
    family_domains = family_dict[family]
    family_mask = [domain in family_domains for domain in list(dataset.domain_names)]
    family_variants = [dataset[i] for i in range(len(dataset)) if family_mask[i]]
    family_size = len(family_variants)
    train_domains.extend(family_domains)
    current_train_size += family_size

    family = all_families[1]
    family_domains = family_dict[family]
    family_mask = [domain in family_domains for domain in list(dataset.domain_names)]
    family_variants = [dataset[i] for i in range(len(dataset)) if family_mask[i]]
    family_size = len(family_variants)
    val_domains.extend(family_domains)
    current_val_size += family_size

    family = all_families[2]
    family_domains = family_dict[family]
    family_mask = [domain in family_domains for domain in list(dataset.domain_names)]
    family_variants = [dataset[i] for i in range(len(dataset)) if family_mask[i]]
    family_size = len(family_variants)
    test_domains.extend(family_domains)
    current_test_size += family_size

    splits_dict = {}

    for family in all_families[3:]:

        family_domains = family_dict[family]
        family_mask = [domain in family_domains for domain in list(dataset.domain_names)]
        family_variants = [dataset[i] for i in range(len(dataset)) if family_mask[i]]
        family_size = len(family_variants)

        if current_train_size + family_size <= n_train_variants:

            train_domains.extend(family_domains)
            current_train_size += family_size

        elif current_val_size + family_size <= n_val_variants:

            val_domains.extend(family_domains)
            current_val_size += family_size

        else:

            test_domains.extend(family_domains)
            current_test_size += family_size

        splits_dict = {"train" : train_domains, "validation" : val_domains, "test" : test_domains}

    return splits_dict

def subset_dataset(dataset: Dataset, splits_dict: dict) -> dict:

    train_indices = []
    validation_indices = []
    test_indices = []

    for idx, item in enumerate(dataset):
        domain = item["domain_name"]
        if domain in splits_dict["train"]: train_indices.append(idx)
        elif domain in splits_dict["validation"]: validation_indices.append(idx)
        else: test_indices.append(idx)

    # Create Subset objects
    training_subset = Subset(dataset, train_indices)
    validation_subset = Subset(dataset, validation_indices)
    testing_subset = Subset(dataset, test_indices)

    return {"training_split": training_subset, "validation_split": validation_subset, "testing_split": testing_subset}
