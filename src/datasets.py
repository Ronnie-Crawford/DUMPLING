# Standard modules
import csv
from distutils.util import strtobool
from functools import partial

# Third-party modules
import pandas as pd
import torch
from torch.utils.data import Dataset

# Local modules
from splits import load_training_sequences

MISSING_COLUMN = object()   # Used as sentinel

class ProteinDataset(Dataset):
    
    def __init__(
        self,
        domain_names: list,
        wt_flags: list,
        aa_seqs: list,
        sequence_embeddings: list,
        predicted_features: list,
        feature_values: dict,
        feature_masks: dict,
        labels: dict
        ):
        
        self.domain_names = domain_names
        self.wt_flags = wt_flags
        self.aa_seqs = aa_seqs
        self.sequence_embeddings = sequence_embeddings
        self.predicted_features = predicted_features
        self.feature_values = feature_values
        self.feature_masks = feature_masks
        self.labels = labels
    
    def __len__(self):
        
        return len(self.aa_seqs)

    def __getitem__(self, idx: int) -> dict:
        
        item = {
            "domain_name": self.domain_names[idx],
            "wt_flag": self.wt_flags[idx],
            "aa_seq": self.aa_seqs[idx].replace('*', '<unk>'),
            "sequence_embedding": self.sequence_embeddings[idx]
        }
        
        for feat in self.predicted_features:
            
            item[f"{feat}_value"] = torch.tensor(self.feature_values[feat][idx])
            item[f"{feat}_mask"]  = torch.tensor(self.feature_masks [feat][idx])
        
        for label_name in self.labels:
            
            item[label_name] = torch.tensor(self.labels[label_name][idx])
            
        return item
    
    @classmethod
    def from_file(
        cls,
        path: str,
        domain_name_column: str,
        wt_flag_column: str,
        aa_seq_column: str,
        predicted_feature_columns: dict,
        label_columns: dict,
        amino_acids: str,
        ):
        
        """
        Formats dataset file before constructing dataset object, also checks row values are valid.
        """
        
        dataset_rows = read_dataset_file(path)
        
        # Ensure every row has a boolean in "wt_flag"
        for row in dataset_rows:
            
            row["wt_flag"] = bool(strtobool(row.get(wt_flag_column, "False").strip()))
            
        # If not WT flad column passed, assume all rows are non-WT
        if not wt_flag_column:
            
            for row in dataset_rows:
                
                row["wt_flag"] = False
                
            wt_flag_column = "wt_flag"
        
        # Initialise empty lists for dataset attributes
        domain_names, wt_flags, aa_seqs = [], [], []
        feature_values = {feature: [] for feature in predicted_feature_columns}
        feature_masks  = {feature: [] for feature in predicted_feature_columns}
        labels = {label: [] for label in label_columns if label != "ALL"}

        # Collect and check row attributes
        for row in dataset_rows:

            aa_seq = row[aa_seq_column]
            
            # If the aa sequence contains unexpected values, it is not added to dataset
            if not is_valid_sequence(aa_seq, amino_acids):

                continue
            
            row, values, masks, keep_row = set_feature_values_and_masks(row, predicted_feature_columns)
            
            # Drop row if it has no valid value for any feature
            if not keep_row:

                continue
            
            # Add row's attributes to dataset attributes
            domain_names.append(row[domain_name_column])
            wt_flags.append(bool(strtobool(row[wt_flag_column].strip())))
            aa_seqs.append(aa_seq)
            
            for feature in predicted_feature_columns:
                
                feature_values[feature].append(values[feature])
                feature_masks[feature].append(masks[feature])

            for label, label_config in label_columns.items():
                
                if label == "ALL":
                    
                    continue

                column_name = label_config["COLUMN_NAME"]
                labels[label].append(bool(strtobool(row[column_name].strip())) if column_name else False)
        
        # Initialise zero embeddings in one group for efficiency - zero tensor is placeholder for later
        sequence_embeddings = [torch.zeros(0) for _ in domain_names]
        
        return cls(
            domain_names,
            wt_flags,
            aa_seqs,
            sequence_embeddings,
            list(predicted_feature_columns.keys()),
            feature_values,
            feature_masks,
            labels
        )
    
    def filter_by_indices(self, indices: list) -> "ProteinDataset":
        
        """
        Takes dataset and list of indices to keep and returns the filtered dataset.
        """
        
        kwargs = {
            "domain_names": [self.domain_names[i] for i in indices],
            "wt_flags": [self.wt_flags[i] for i in indices],
            "aa_seqs": [self.aa_seqs[i] for i in indices],
            "sequence_embeddings": [self.sequence_embeddings[i] for i in indices],
            "predicted_features": self.predicted_features,
            "feature_values": {feat: [self.feature_values[feat][i] for i in indices] for feat in self.predicted_features},
            "feature_masks":  {feat: [self.feature_masks [feat][i] for i in indices] for feat in self.predicted_features},
            "labels": {label: [self.labels[label][i] for i in indices] for label in self.labels}
        }
        
        return type(self)(**kwargs)
    
    def filter_by_label(self, label_name: str, keep_wts: bool = True) -> "ProteinDataset":
        
        """
        Looks at supplied label and filters the dataset based on true/false values,
        can also keep wildtypes independent of whether they have the label or not.
        With keep_wts, it will still only keep wildtypes for which there is at least one non-WT being kept in the same domain.
        """
        
        non_wt_indices = [index for index, label_keep in enumerate(self.labels.get(label_name, [])) if label_keep]
        domains_with_non_wt_keeps = {self.domain_names[index] for index in non_wt_indices}
        keep_indices = list(non_wt_indices)
        
        if keep_wts:
            
            wt_indices = [index for index, wt in enumerate(self.wt_flags) if wt and self.domain_names[index] in domains_with_non_wt_keeps]
            keep_indices += wt_indices
            
        keep_indices = sorted(set(keep_indices))
                
        return self.filter_by_indices(keep_indices)
      
def handle_data(
    base_path: str,
    subsets_in_use: list[tuple],
    datasets_config_dict: dict,
    is_filter_one_wildtype_per_domain: bool,
    predicted_features: list,
    feature_remapping_dict: dict,
    amino_acids: str
    ) -> list:
    
    """
    The handler function for preparing all datasets.
    It creates the full length dataset for each subset, and does not yet filter by label,
    although this means increased memory usage storing multiple versions of the same full-length dataset,
    its a compromise so that we only need to find embeddings once per dataset, rather than per subset.
    Perhaps in the future label filtering could occur here,
    and more complex embedding loading will mean no extra embeddings needed.
    """
    
    # Set up datasets
    dataset_dicts = []
    
    for dataset_name, label in subsets_in_use:
        
        # For each requested predicted feature, check if the dataset has a column for it,
        # if it does, pass it to constructor, if not, pass sentinel object MISSING_COLUMN, which is handled by constructor
        feature_column_mapping = {
            feature: datasets_config_dict[dataset_name]["PREDICTED_FEATURE_COLUMNS"].get(feature, MISSING_COLUMN)
            for feature in predicted_features
        }
        label_column_mapping = datasets_config_dict[dataset_name]["LABEL_COLUMNS"]
            
        dataset = ProteinDataset.from_file(
            base_path / datasets_config_dict[dataset_name]["PATH"],
            datasets_config_dict[dataset_name]["DOMAIN_NAME_COLUMN"],
            datasets_config_dict[dataset_name]["WT_FLAG_COLUMN"],
            datasets_config_dict[dataset_name]["AA_SEQ_COLUMN"],
            feature_column_mapping,
            label_column_mapping,
            amino_acids
            )
    
        if is_filter_one_wildtype_per_domain:
    
            dataset = filter_domains_with_one_wt(dataset)
        
        dataset_dicts.append({
            "dataset_name": dataset_name,
            "dataset": dataset,
            "label": label,
            "unique_key": f"{dataset_name}-{label}"
            })
        
    # Apply and feature remapping so that, for example,
    # activity can be treated as fitness for inference
    dataset_dicts = apply_feature_remapping(dataset_dicts, feature_remapping_dict)
    
    return dataset_dicts      
        
def read_dataset_file(file_path: str) -> list:
    
    # Detect delimiter
    with open(file_path, "r") as dataset_file:
        
        first = dataset_file.readline()
        delimiter = "," if first.count(",") > first.count("\t") else "\t"

    # Read data
    with open(file_path, "r") as dataset_file:
        
        reader = csv.DictReader(dataset_file, delimiter = delimiter)
        rows = list(reader)
    
    return rows

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

def set_feature_values_and_masks(row, predicted_feature_columns):
    
    # Determine masks and values
    keep_row = False    # keep_row checks if the row has valid value for at least one feature
    values = {}
    masks = {}

    for feature, column in predicted_feature_columns.items():
        
        row_feature_value = row.get(column, MISSING_COLUMN)
        
        # If a specific column name wasn't assigned, it will default to the sentinel object (MISSING_COLUMN)
        if row_feature_value != MISSING_COLUMN and is_tensor_ready(row_feature_value):
            
            value = float(row_feature_value)
            mask = True
            
        else:
            
            value = 0.0
            mask = False
                    
        values[feature] = value
        masks[feature] = mask
                    
        if mask:
            
            keep_row = True
        
        # Ensures we keep all WTs, downstream logic in label filtering later filters them to be 1 per domain,
        # but this ensures even those without valid data get an embedding, but are masked out to do not contribute
        # to training and inference
        if row["wt_flag"]:
            
            keep_row = True
    
    return row, values, masks, keep_row

def filter_domains_with_one_wt(dataset):
    
    """
    Filters a dataset to only keep domains which contain exactly one WT sequence.
    """
    
    domain_wt_df = pd.DataFrame({
        "domain_name": dataset.domain_names,
        "is_wt": dataset.wt_flags
    })
    
    # Find domains where exactly one wt_flag is True
    valid = domain_wt_df.groupby("domain_name")["is_wt"].sum().eq(1)
    valid_domains = valid[valid].index
    keep_indicies = domain_wt_df.index[domain_wt_df["domain_name"].isin(valid_domains)].tolist()
    filtered_dataset = dataset.filter_by_indices(keep_indicies)
    
    return filtered_dataset

def add_spoof_train_dataset(dataset_dicts, training_sequences_path, predicted_features_list):
    
    dataset_dicts.append({
        "dataset_name": "spoof_training_dataset",
        "dataset": make_spoof_train_dataset(load_training_sequences(training_sequences_path), predicted_features_list),
        "label": "spoof_training_dataset",
        "unique_key": "spoof_training_dataset"
        })
    
    return dataset_dicts

def make_spoof_train_dataset(
    train_sequence_list: list[str],
    predicted_features: list[str]
) -> ProteinDataset:
    
    """
    Build a ProteinDataset containing only the given sequences,
    with dummy values for all other fields.
    Predicted feature columns generated, but empty, for compatability,
    no label columns generated.
    """
    
    # Generate dummy attributes
    domain_names = ["" for _ in range(len(train_sequence_list))]
    wt_flags = [False] * len(train_sequence_list)
    sequence_embeddings = [torch.zeros(0) for _ in range(len(train_sequence_list))]
    feature_values = {feature: [0.0] * len(train_sequence_list) for feature in predicted_features}
    feature_masks  = {feature: [False] * len(train_sequence_list) for feature in predicted_features}
    labels = {}

    return ProteinDataset(
        domain_names,
        wt_flags,
        train_sequence_list,
        sequence_embeddings,
        predicted_features,
        feature_values,
        feature_masks,
        labels
    )

def handle_filtering(dataset_dicts):

    """
    Handler function for filtering, current only filtering by labels.
    """

    for index, dataset_dict in enumerate(dataset_dicts):
        
        if dataset_dict["label"] != "ALL":
            
            dataset_dict["dataset"] = dataset_dict["dataset"].filter_by_label(dataset_dict["label"])
    
        dataset_dicts[index] = dataset_dict

    return dataset_dicts

def apply_feature_remapping(dataset_dicts, remap_dict):
    
    """
    Uses dict from config to remap features,
    copies the values over and sets new feature mask to true
    """
    
    for dataset_dict in dataset_dicts:
        
        for source, target in remap_dict.items():
            
            # Skip if not in dataset
            if target not in dataset_dict["dataset"].feature_masks or source not in dataset_dict["dataset"].feature_masks:
                
                continue

            for index in range(len(dataset_dict["dataset"].aa_seqs)):
                
                if dataset_dict["dataset"].feature_masks[source][index] and not dataset_dict["dataset"].feature_masks[target][index]:
                    
                    dataset_dict["dataset"].feature_masks[target][index]  = True
                    dataset_dict["dataset"].feature_values[target][index] = dataset_dict["dataset"].feature_values[source][index]
                    
    return dataset_dicts