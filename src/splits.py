# Standard modules
import pickle
from pathlib import Path
import random

# Third-party modules
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader, WeightedRandomSampler

# Local modules
from sampling import apply_sampling

def handle_splits(
    dataset_dicts: list[dict],
    datasets_splits_dict: dict,
    exclude_wildtype_from_inference: bool,
    predicted_features: list,
    sampling_flag: bool,
    subset_weighting_flag: bool,
    severity_weighting_flag: bool,
    reliability_weighting_flag: bool,
    priority_split: str,
    training_assign_bias: int,
    validation_assign_bias: int,
    testing_assign_bias: int,
    batch_size: int,
    n_workers: int,
    homology_path,
    splits_path,
    save_splits: bool,
    load_splits: bool
    ):

    if load_splits == True:

        split_datasets = load_sequence_splits_from_file(splits_path, dataset_dicts)

    else:

        # Read homology families
        homology_family_file_path = homology_path / "sequence_families.tsv"
        homology_family_dict = read_homology_file(homology_family_file_path)

        # Prepare global split tracking - Family ID: Train/Validation/Test
        family_to_split_assignment = {}

        # Compute desired split sizes for each dataset
        desired_split_sizes = calculate_desired_split_sizes(
            dataset_dicts,
            datasets_splits_dict
            )

        # Assign families globally
        family_to_split_assignment = assign_families_to_splits(
            dataset_dicts,
            homology_family_dict,
            desired_split_sizes,
            datasets_splits_dict,
            priority_split,
            training_assign_bias,
            validation_assign_bias,
            testing_assign_bias,
        )

        # Get indices per dataset and per split based on assignment
        datasets_split_indices = get_datasets_split_indices(
            dataset_dicts,
            homology_family_dict,
            family_to_split_assignment,
            datasets_splits_dict
        )

        print_datasets(dataset_dicts, datasets_split_indices)

        # Save / load splits
        if save_splits == True:

            save_splits_to_file(datasets_split_indices, dataset_dicts, splits_path)

        # Filter datasets
        split_datasets = construct_filtered_datasets(
            dataset_dicts,
            datasets_split_indices
            )

    # Remove wildtype sequences from inference splits
    if exclude_wildtype_from_inference:

        split_datasets["VALIDATION"] = remove_wt_from_split(split_datasets["VALIDATION"])
        split_datasets["TEST"] = remove_wt_from_split(split_datasets["TEST"])

    # Save training sequences (depreciated)
    #save_training_sequences(results_path, datasets_split_indices, dataset_dicts)
    #test_subset_to_sequence_dict = create_subset_to_sequence_dict(split_datasets["TEST"])

    final_splits = {
        split: ConcatDataset(split_datasets[split].values())
        for split, dict_check in split_datasets.items()
        if split in ["TRAIN","VALIDATION","TEST"] and dict_check
    }

    # Use a sampler if desired
    train_sampler = None

    if sampling_flag == True:

        train_sampler = apply_sampling(split_datasets["TRAIN"], predicted_features, subset_weighting_flag, severity_weighting_flag, reliability_weighting_flag)

    # Load into dataloaders
    dataloaders_dict = splits_to_loaders(final_splits, batch_size, n_workers, train_sampler)

    return dataloaders_dict

def read_homology_file(homology_file: str) -> dict:

    """
    Reads the homology file and creates a dictionary mapping family IDs to sequences.

    Parameters:
        - homology_file (str): Path to the homology file.

    Returns:
        - family_dict (dict): Dictionary mapping family IDs to lists of sequences.
    """

    homology_df = pd.read_csv(homology_file, sep = "\t")
    family_dict = {}

    for _index, row in homology_df.iterrows():

        family_id = row["sequence_family"]
        sequence = row["sequence"]

        if family_id not in family_dict:

            family_dict[family_id] = []

        family_dict[family_id].append(sequence)

    return family_dict

def calculate_desired_split_sizes(dataset_dicts, datasets_splits_dict):

    """
    Find the ideal amount of sequences from each subset to put into each split, including an unassigned split.
    """

    desired_split_sizes = {}

    for dataset_dict in dataset_dicts:

        total_sequences = len(dataset_dict["dataset"])
        splits = datasets_splits_dict[dataset_dict["unique_key"]]
        train_count = int(round(total_sequences * splits.get("TRAIN", 0)))
        validation_count = int(round(total_sequences * splits.get("VALIDATION", 0)))
        test_count = int(round(total_sequences * splits.get("TEST", 0)))
        unassigned_count = total_sequences - (train_count + validation_count + test_count)

        desired_split_sizes[dataset_dict["unique_key"]] = {
            "TRAIN": train_count,
            "VALIDATION": validation_count,
            "TEST": test_count,
            "UNASSIGNED": unassigned_count
        }

    return desired_split_sizes

def assign_families_to_splits(
    dataset_dicts,
    homology_family_dict,
    desired_split_sizes,
    datasets_splits_dict,
    priority_split,
    training_assign_bias,
    validation_assign_bias,
    testing_assign_bias
    ):

    """
    This function is now a lot more permissive, rather than trying to match splits exactly;
    instead it tries to ensure that every subset has its desired splits filled,
    this leaves a lot of subset-splits overfilled, but this is now left to the next function to trim.
    """

    family_to_split_assignment = {}
    split_options = ["TRAIN", "VALIDATION", "TEST", "UNASSIGNED"]
    subset_names = [dataset_dict["unique_key"] for dataset_dict in dataset_dicts]
    split_counts_per_subset = {name: {split: 0 for split in split_options} for name in subset_names}

    # Precompute fast sequence lookup sets - to help find which subset a sequence is in
    sequence_sets = []

    for dataset_dict in dataset_dicts:

        sequence_sets.append((dataset_dict["unique_key"], set(dataset_dict["dataset"].aa_seqs)))

    # Precompute family-to-subset sequence counts - for each subset, how many sequences from each homology family does it contain
    family_subset_counts = {}

    for family_id, family_seqs in homology_family_dict.items():

        counts = {}

        for subset_name, sequence_set in sequence_sets:

            counts[subset_name] = len(set(family_seqs) & sequence_set)

        family_subset_counts[family_id] = counts

    # Shuffle the family IDs
    all_family_ids = list(homology_family_dict.keys())
    np.random.shuffle(all_family_ids)

    # Assign one family to each non-zero split
    # Currently unimplimented to aid exact splits being created,
    # however might be useful in the future.

    #family_to_split_assignment, split_counts_per_subset = assign_at_least_one_family_to_required_splits(
    #    family_subset_counts,
    #    dataset_dicts,
    #    desired_split_sizes,
    #    family_to_split_assignment,
    #    split_counts_per_subset
    #    )

    # If there is a priority split
    if priority_split != None:

        family_to_split_assignment = assign_families_to_priority_split(
            priority_split,
            family_subset_counts,
            dataset_dicts,
            desired_split_sizes,
            family_to_split_assignment,
            split_counts_per_subset
            )

    # Assign families using heuristics to find best way to fill up each split
    for family_id in all_family_ids:

        if family_id in family_to_split_assignment:

            continue

        best_split = select_best_split(
            family_id,
            family_subset_counts,
            split_counts_per_subset,
            desired_split_sizes,
            dataset_dicts,
            datasets_splits_dict,
            training_assign_bias,
            validation_assign_bias,
            testing_assign_bias
        )

        family_to_split_assignment[family_id] = best_split
        split_counts_per_subset = update_split_counts(family_id, family_subset_counts, split_counts_per_subset, best_split)

    return family_to_split_assignment

def assign_families_to_priority_split(
    priority_split: str,
    family_subset_counts: dict[str, dict[str, int]],
    dataset_dicts: list[dict],
    desired_split_sizes: dict[str, dict[str, int]],
    family_to_split_assignment: dict[str, str],
    split_counts_per_subset: dict[str, dict[str, int]],
):
    """
    Greedily fill each subset's priority_split quota by choosing the
    family that adds the most sequences for that subset until it
    reaches its desired count.
    """

    # All families not yet assigned
    unassigned = set(family_subset_counts)

    # Work subset-by-subset
    for subset_dict in dataset_dicts:

        subset = subset_dict["unique_key"]
        target = desired_split_sizes[subset][priority_split]

        # Keep adding until we hit or exceed the target, or run out of families
        while split_counts_per_subset[subset][priority_split] < target and unassigned:
            # pick family that contributes most to *this* subset
            best = max(unassigned, key=lambda f: family_subset_counts[f][subset])
            contrib = family_subset_counts[best][subset]
            if contrib == 0:
                # no remaining family can help this subset
                break

            # assign it
            family_to_split_assignment[best] = priority_split
            unassigned.remove(best)

            # update *all* subsets’ counts
            for sub in split_counts_per_subset:
                split_counts_per_subset[sub][priority_split] += family_subset_counts[best][sub]

    return family_to_split_assignment

def assign_at_least_one_family_to_required_splits(
    family_subset_counts: dict[str, dict[str, int]],
    dataset_dicts: list[dict],
    desired_split_sizes: dict[str, dict[str, int]],
    family_to_split_assignment,
    split_counts_per_subset
    ):

    """
    Randomly assigning families can lead to edge-cases where one dataset has its train-test assigned correctly,
    but this causes another dataset to have no families available to assign to the splits it requires.
    This function aims to assign at least one family to each subset-split,
    before the others are assigned randomly, and throws an error if this is not possible.
    """

    for dataset_dict in dataset_dicts:

        for split in ["TRAIN", "VALIDATION", "TEST"]:

            if desired_split_sizes[dataset_dict["unique_key"]][split] > 0 and split_counts_per_subset[dataset_dict["unique_key"]][split] == 0:

                eligible_families = [
                    family for family, counts in family_subset_counts.items()
                    if family not in family_to_split_assignment and counts[dataset_dict["unique_key"]] > 0
                    ]

                if not eligible_families:

                    raise RuntimeError(f"No available family can fill at least one sequence for "f"{dataset_dict['unique_key']} {split}.")

                reserved_family = random.choice(eligible_families)
                family_to_split_assignment[reserved_family] = split

                for subset in split_counts_per_subset:

                    split_counts_per_subset[subset][split] += (family_subset_counts[reserved_family][subset])

    return family_to_split_assignment, split_counts_per_subset

def select_best_split(
    family_id: str,
    family_subset_counts: dict[str, dict[str, int]],
    split_counts_per_subset: dict[str, dict[str, int]],
    desired_split_sizes: dict[str, dict[str, int]],
    dataset_dicts: list[dict],
    datasets_splits_dict: dict[str, dict],
    training_assign_bias = 1,
    validation_assign_bias = 1,
    testing_assign_bias = 1
) -> str:

    """
    Each family can be assigned to training, validation, testing, or left unassigned.
    For each of these possibilities we check how "hungry" each subset is for it to be assigned,
    and also how much that particular family would satisify. The current metric simply mutiplies these variables,
    and can be biased, incase you really want the a particular split to be full.
    The split sizes are the goals, the biases are how prioritised each is, a larger number is a higher priority.
    """

    best_split = "UNASSIGNED"
    min_total_deviation = float("inf")

    # Check each split, for each subset, to see what the deviation from the desired split is.
    for split in ["TRAIN", "VALIDATION", "TEST"]:

        split_metric = 0
        accept_split = False

        for dataset_dict in dataset_dicts:

            family_sequences_in_subset_count = family_subset_counts[family_id][dataset_dict["unique_key"]]
            subset_sequences_in_split_count = split_counts_per_subset[dataset_dict["unique_key"]][split]
            projected_subset_sequence_in_split_count = subset_sequences_in_split_count + family_sequences_in_subset_count
            desired_subset_sequence_in_split_count = desired_split_sizes[dataset_dict["unique_key"]][split]

            # Our metric is a combination of "How hungry is this split?" and "How much will this family fill it up?"
            subset_deviation = projected_subset_sequence_in_split_count - desired_subset_sequence_in_split_count
            family_contribution = family_sequences_in_subset_count * subset_deviation

            if split == "TRAIN": family_contribution *= training_assign_bias
            if split == "VALIDATION": family_contribution *= validation_assign_bias
            if split == "TEST": family_contribution *= testing_assign_bias

            # If the projected split once the family is added to it is
            if family_contribution < 0:

                accept_split = True
                split_metric += family_contribution

        if split_metric < min_total_deviation and accept_split == True:

            min_total_deviation = split_metric
            best_split = split

    return best_split

def update_split_counts(family_id, family_subset_counts, split_counts_per_subset, split):

    for subset_name in split_counts_per_subset.keys():

        split_counts_per_subset[subset_name][split] += family_subset_counts[family_id][subset_name]

    return split_counts_per_subset

def get_datasets_split_indices(
    dataset_dicts: list,
    homology_family_dict: dict,
    family_to_split_assignment,
    datasets_splits_dict
    ):

    """
    This function now accepts the potentially overfilled subset-splits from the assign families function
    and then trims out individual sequences randomly, until each subset-split is as close as it can get to its
    desired number of sequences.
    """

    # Build a reverse map: sequence -> assigned split
    sequence_to_split: dict[str, str] = {}

    for family_id, split in family_to_split_assignment.items():

        for sequence in homology_family_dict.get(family_id, []):

            sequence_to_split[sequence] = split

    datasets_split_indices = {dataset_dict["unique_key"]: {"TRAIN": [], "VALIDATION": [], "TEST": [], "UNASSIGNED": []} for dataset_dict in dataset_dicts}

    # Raw assignment by reading each dataset’s sequences
    for dataset_dict in dataset_dicts:

        subset = dataset_dict["unique_key"]

        for index, sequence in enumerate(dataset_dict["dataset"].aa_seqs):

            assigned_split = sequence_to_split.get(sequence, "UNASSIGNED")
            datasets_split_indices[subset][assigned_split].append(index)

    for dataset_dict in dataset_dicts:

        subset_name = dataset_dict["unique_key"]
        subset = dataset_dict["dataset"]
        total_sequences_count = len(subset.aa_seqs)

        # For each real split, calculate allowed count and prune if needed
        for split in ["TRAIN", "VALIDATION", "TEST"]:

            desired_sequences_count = int(total_sequences_count * datasets_splits_dict[subset_name][split])
            current_sequences = datasets_split_indices[subset_name][split]

            if len(current_sequences) > desired_sequences_count:

                kept_sequences = random.sample(current_sequences, desired_sequences_count)
                datasets_split_indices[subset_name][split] = kept_sequences

    return datasets_split_indices

def construct_filtered_datasets(
    dataset_dicts: list,
    datasets_split_indices
    ):

    split_datasets = { "TRAIN": {}, "VALIDATION": {}, "TEST": {}, "UNASSIGNED": {}}

    for dataset_name, indices_dict in datasets_split_indices.items():

        for split in ["TRAIN", "VALIDATION", "TEST"]:

            target_dataset_dict = [dataset_dict for dataset_dict in dataset_dicts if dataset_dict["unique_key"] == dataset_name][0]
            dataset_subset = target_dataset_dict["dataset"].filter_by_indices(indices_dict[split])
            split_datasets[split][dataset_name] = dataset_subset

    return split_datasets

def print_datasets(dataset_dicts, datasets_split_indices):

    print("\n---------- Dataset-Split Assignments ----------")

    for dataset_dict in dataset_dicts:

        total_length = len(dataset_dict["dataset"])

        if total_length > 0:

            train_count = len(datasets_split_indices[dataset_dict["unique_key"]]["TRAIN"])
            validation_count = len(datasets_split_indices[dataset_dict["unique_key"]]["VALIDATION"])
            test_count = len(datasets_split_indices[dataset_dict["unique_key"]]["TEST"])
            print(f"[{dataset_dict['unique_key']}] [Train: {train_count}/{total_length} {100 * train_count / total_length:.1f}% / Validation: {validation_count}/{total_length} {100 * validation_count / total_length:.1f}%/ Test: {test_count}/{total_length} {100 * test_count / total_length:.1f}%] ")

    print("-----------------------------------------------")

def remove_wt_from_split(split: dict) -> dict:

    if not split:

        return split

    filtered_split = {}

    for dataset_name, dataset in split.items():

        keep_indices = [index for index, wt_flag in enumerate(dataset.wt_flags) if not wt_flag]
        filtered_dataset = dataset.filter_by_indices(keep_indices)
        filtered_split[dataset_name] = filtered_dataset

    return filtered_split

def save_training_sequences(
    save_directory: Path,
    datasets_split_indices: dict,
    dataset_dicts: list
) -> None:

    """
    Save splits as pickle files.

    - datasets_split_indices:     { dataset_name: { 'TRAIN': [...], ... } }
    - datasets_dict:              { dataset_name: ProteinDataset }

    This writes:
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

def create_subset_to_sequence_dict(split_subsets):

    subset_to_sequence_dict = {}

    for subset_name, dataset in split_subsets.items():

        subset_to_sequence_dict[subset_name] = dataset.aa_seqs

    return subset_to_sequence_dict

def splits_to_loaders(
    splits: dict,
    batch_size: int,
    n_workers: int,
    train_sampler
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

        if split_name == "TRAIN" and train_sampler != None:

            loaders[split_name] = DataLoader(
            dataset,
            batch_size = batch_size,
            sampler = train_sampler,
            drop_last = drop_last_flag,
            num_workers = n_workers,
            persistent_workers = True,
            collate_fn = collate_fn
            )

        else:

            loaders[split_name] = DataLoader(
                dataset,
                batch_size = batch_size,
                shuffle = shuffle_flag,
                drop_last = drop_last_flag,
                num_workers = n_workers,
                persistent_workers = True,
                collate_fn = collate_fn
            )

    return loaders

def collate_fn(batch):

    # Extract data to keep and batch it
    domains = [item["domain_name"] for item in batch]
    subsets = [item["subset"] for item in batch]
    aa_seqs = [item["aa_seq"] for item in batch]
    sequence_embeddings = torch.stack([item["sequence_embedding"] for item in batch], dim = 0)

    # Collect other batch elements
    batch_dict = {
        "domain_name": domains,
        "subset": subsets,
        "aa_seq": aa_seqs,
        "sequence_embedding": sequence_embeddings
    }

    for key in batch[0]:

        if key.endswith("_value") or key.endswith("_mask"):

            values = [item[key] for item in batch]

            if isinstance(values[0], torch.Tensor):

                batch_dict[key] = torch.stack(values)

            elif isinstance(values[0], (float, int, bool)):

                batch_dict[key] = torch.tensor(values)

            else:

                # Handle other data types if necessary - it should not be necessary
                batch_dict[key] = values

    return batch_dict

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

def remove_homologous_sequences_from_inference(
    dataset_dicts: list,
    homology_path: Path
) -> list:

    """
    Given a dict of ProteinDataset instances—one entry must be
    "spoof_training_dataset" holding the training sequences—this
    function removes from every other dataset any sequences that
    share a homology family with the spoof training set.

    Returns a new dict of filtered ProteinDataset (excluding the spoof).
    """

    # Load the mapping sequence → family
    sequence_to_family = read_sequence_to_family(homology_path / "sequence_families.tsv")
    spoof = None

    # Identify all families used by the spoof training dataset
    for index, dataset_dict in enumerate(dataset_dicts):

        if dataset_dict["dataset_name"] == "spoof_training_dataset":

            spoof = dataset_dict["dataset"]
            break

    assert isinstance(spoof, Dataset), "Could not correctly fetch spoof training dataset."
    used_families = {sequence_to_family.get(sequence) for sequence in spoof.aa_seqs if sequence_to_family.get(sequence) is not None}

    # Only keep indices in remaining data if it is not in a family used by the spoof data
    for index, dataset_dict in enumerate(dataset_dicts):

        name = dataset_dict["unique_key"]
        dataset = dataset_dict["dataset"]

        if name == "spoof_training_dataset":

            continue

        # === DEBUG STEP 3: homology filter effect on this subset ===
        pre_filter = len(dataset.aa_seqs)
        keep_indices = [index for index, sequence in enumerate(dataset.aa_seqs) if sequence_to_family.get(sequence) not in used_families]
        post_filter = len(keep_indices)
        print(
            f"DEBUG STEP 3 – {name}: "
            f"{post_filter}/{pre_filter} "
            f"({post_filter/pre_filter*100:.1f}%) variants kept"
        )
        dataset_dicts[index]["dataset"] = dataset.filter_by_indices(keep_indices)

    return dataset_dicts

#@lru_cache(maxsize = None)
def read_sequence_to_family(homology_tsv_path):

    """
    Cache the sequence-family map so we only parse the TSV once.
    """

    homology_df = pd.read_csv(homology_tsv_path, sep = "\t")
    sequence_to_family_dict = {}

    for _, row in homology_df.iterrows():

        sequence_to_family_dict[row["sequence"]] = row["sequence_family"]

    return sequence_to_family_dict

def save_splits_to_file(
    datasets_split_indices,
    dataset_dicts,
    splits_path
    ):

    # Sequence to split mapping
    sequence_splits = {"TRAIN": [], "VALIDATION": [], "TEST": []}
    # for each subset-dataset
    for dataset_dict in dataset_dicts:

        key = dataset_dict["unique_key"]
        aa_seqs = dataset_dict["dataset"].aa_seqs

        for split in ("TRAIN", "VALIDATION", "TEST"):

            for index in datasets_split_indices[key][split]:

                sequence_splits[split].append(aa_seqs[index])

    # Duplicate sequences should be in the same splits anyway, but just to be sure nothing breaks downstream
    for split, sequences in sequence_splits.items():

        seen = set()
        deduped = []

        for sequence in sequences:

            if sequence not in seen:

                seen.add(sequence)
                deduped.append(sequence)

        sequence_splits[split] = deduped

    with open(splits_path, "wb") as pickle_file:

        pickle.dump(sequence_splits, pickle_file)

def load_sequence_splits_from_file(splits_file_path, dataset_dicts):

    with open(splits_file_path, "rb") as splits_file:

        sequence_splits = pickle.load(splits_file)

    train_set = set(sequence_splits.get("TRAIN", []))
    val_set   = set(sequence_splits.get("VALIDATION", []))
    test_set  = set(sequence_splits.get("TEST", []))
    datasets_split_indices = {}

    datasets_split_indices = {dataset_dict["unique_key"]: {"TRAIN": [], "VALIDATION": [], "TEST": [], "UNASSIGNED": []} for dataset_dict in dataset_dicts}

    for dataset_dict in dataset_dicts:

        key = dataset_dict["unique_key"]
        data = dataset_dict["dataset"]

        train_idx = []
        val_idx   = []
        test_idx  = []
        unass_idx = []

        for i, sequence in enumerate(data.aa_seqs):

            if sequence in train_set:

                train_idx.append(i)

            elif sequence in val_set:

                val_idx.append(i)

            elif sequence in test_set:

                test_idx.append(i)

            else:

                unass_idx.append(i)

        datasets_split_indices[key] = {
            "TRAIN": train_idx,
            "VALIDATION": val_idx,
            "TEST": test_idx,
            "UNASSIGNED": unass_idx
        }

    split_datasets = construct_filtered_datasets(
            dataset_dicts,
            datasets_split_indices
            )

    return split_datasets

def load_domain_splits_from_file(
    splits_file_path,
    dataset_dicts: list,
    exclude_wildtype_from_inference: bool,
    batch_size: int,
    n_workers: int,
    ):

    # Read in splits
    loaded_splits = pd.read_pickle(splits_file_path)

    if loaded_splits["train"][0].endswith(".pdb"):

        loaded_train_split_domains = [domain.removesuffix(".pdb") for domain in loaded_splits["train"]]
        loaded_test_split_domains = [domain.removesuffix(".pdb") for domain in loaded_splits["test"]]

    else:

        loaded_train_split_domains = [domain for domain in loaded_splits["train"]]
        loaded_test_split_domains = [domain for domain in loaded_splits["test"]]

    train_datasets = {}
    test_datasets  = {}

    # Iterate through each dataset, sequences which were not part of the original splits file will not be used
    for dataset_dict in dataset_dicts:

        dataset_name = dataset_dict["unique_key"]
        dataset = dataset_dict["dataset"]
        train_idx = [index for index, domain in enumerate(dataset.domain_names) if domain in loaded_train_split_domains]
        test_idx  = [index for index, domain in enumerate(dataset.domain_names) if domain in loaded_test_split_domains]

        train_dataset = dataset.filter_by_indices(train_idx)
        test_dataset  = dataset.filter_by_indices(test_idx)

        train_datasets[dataset_name] = train_dataset
        test_datasets[dataset_name] = test_dataset

    if exclude_wildtype_from_inference:

        test_datasets = remove_wt_from_split(test_datasets)

    final_train = ConcatDataset(list(train_datasets.values()))
    final_test  = ConcatDataset(list(test_datasets.values()))

    loaders = splits_to_loaders(
        splits = {"TRAIN": final_train, "TEST": final_test},
        batch_size = batch_size,
        n_workers = n_workers,
        train_sampler = None
    )

    test_subset_to_sequence_dict = {
      subset_name: ds.aa_seqs
      for subset_name, ds in test_datasets.items()
    }

    return loaders, test_subset_to_sequence_dict
