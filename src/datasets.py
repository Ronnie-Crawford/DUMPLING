# Standard modules
import csv
from distutils.util import strtobool
from functools import partial

# Third-party modules
import torch
from torch.utils.data import Dataset

# Local modules
from helpers import truncate_domain, is_valid_sequence, is_tensor_ready, filter_domains_with_one_wt, get_homology_path
from homology import handle_homology

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
        print(f"Initialized dataset with {len(self)} sequences and features: {self.predicted_features}")

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
        # Detect delimiter
        with open(path, 'r') as f:
            
            first = f.readline()
            delim = ',' if first.count(',') > first.count('\t') else '\t'

        with open(path, 'r') as f:
            
            reader = csv.DictReader(f, delimiter=delim)
            rows = list(reader)

        # Default wt flags
        if not wt_flag_column:
            
            for r in rows:
                r['wt_flag'] = False
            wt_flag_column = 'wt_flag'

        # Build value and mask dicts
        feature_values = {feat: [] for feat in predicted_feature_columns}
        feature_masks  = {feat: [] for feat in predicted_feature_columns}
        
        labels = {label: [] for label in label_columns}

        domain_names, wt_flags, aa_seqs = [], [], []
        
        for row in rows:
            
            seq = row[aa_seq_column]
            
            if not is_valid_sequence(seq, amino_acids):
                
                continue

            # Determine masks and values
            
            keep = False
            values = {}
            masks = {}
            
            for feature, column in predicted_feature_columns.items():
                
                # Explicitly ignore empty column names
                if column == "":
                    
                    value = 0.0
                    mask = False
                    
                else:
                    
                    raw = row.get(column, "")
                    
                    if raw != "" and is_tensor_ready(raw):
                        
                        value = float(raw)
                        mask = True
                        
                    else:
                        
                        value = 0.0
                        mask = False
                            
                values[feature] = value
                masks[feature] = mask
                            
                if mask:
                    
                    keep = True

            if not keep:
                
                continue

            # collect
            domain_names.append(row[domain_name_column])
            wt_flags.append(bool(strtobool(row[wt_flag_column].strip())))
            aa_seqs.append(seq)
            
            for feature in predicted_feature_columns:
                
                feature_values[feature].append(values[feature])
                feature_masks[feature].append(masks[feature])
            
            for label, label_config in label_columns.items():
                
                column_name = label_config["COLUMN_NAME"]
                labels[label].append(bool(strtobool(row[column_name].strip())) if column_name else False)

        # Initialise zero embeddings - placeholder for later
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

    def filter_by_indices(self, indices: list) -> "ProteinDataset":
        
        print("Starting filter by indices")
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
        
        indices = [index for index, value in enumerate(self.labels[label_name]) if value]
        
        indices = []
        
        for i, label_flag in enumerate(self.labels.get(label_name, [])):
            
            if label_flag or (keep_wts and self.wt_flags[i]):
                
                indices.append(i)
                
        return self.filter_by_indices(indices)

def handle_data(
    base_path: str,
    datasets_in_use: list[tuple],
    datasets_config_dict: dict,
    is_filter_one_wildtype_per_domain: bool,
    predicted_features: list,
    amino_acids: str
    ) -> dict:
    
    # Set up datasets
    dataset_dicts = []
    
    for dataset_name, label in datasets_in_use:
            
        feature_column_mapping = {
            feature: datasets_config_dict[dataset_name]["PREDICTED_FEATURE_COLUMNS"].get(feature, "")
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
        
    return dataset_dicts

def make_spoof_train_dataset(
    train_sequence_list: list[str],
    predicted_features: list[str]
) -> ProteinDataset:
    
    """
    Build a ProteinDataset containing only the given sequences,
    with dummy values for all other fields.
    """
    
    # Dummy domain names (we won't use them in filtering)
    domain_names = ["" for _ in range(len(train_sequence_list))]
    wt_flags = [False] * len(train_sequence_list)
    sequence_embeddings = [torch.zeros(0) for _ in range(len(train_sequence_list))]

    # One dict entry per predicted feature, all zeros
    feature_values = { feat: [0.0] * len(train_sequence_list) for feat in predicted_features }
    # All masks False (so no real values; doesn’t matter for filter)
    feature_masks  = { feat: [False] * len(train_sequence_list) for feat in predicted_features }
    labels = {label: [False] * len(train_sequence_list) for label in labels_list or []}


    return ProteinDataset(
        domain_names,
        wt_flags,
        train_sequence_list,
        sequence_embeddings,
        predicted_features,
        feature_values,
        feature_masks
    )

def handle_filtering(dataset_dicts):

    for index, dataset_dict in enumerate(dataset_dicts):
        
        if dataset_dict["label"] != "ALL":
            
            dataset_dict["dataset"] = dataset_dict["dataset"].filter_by_label(dataset_dict["label"])
    
        dataset_dicts[index] = dataset_dict

    return dataset_dicts