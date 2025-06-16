# Standard modules
import json
from pathlib import Path
import os

def read_config(file_path: str) -> dict:

    """
    Reads a JSON configuration file and returns a dictionary of configuration parameters.

    Parameters:
    - file_path (str): The path to the JSON configuration file.

    Returns:
    - config (dict): A dictionary containing the configuration parameters.
    """

    with open(file_path, "r") as file:
        
        config = json.load(file)

    return config

def preflight_checks(config):
    
    """
    Hopefully this will catch most invalid configs before they get a chance to run, and explain what needs changing.
    
    To do:
    - Keep expanding with more tests
    """
    
    # Data
    for dataset in config["DATA"]["DATASETS"].keys():
        
        assert isinstance(config["DATA"]["DATASETS"][dataset]["PATH"], str), f"Config path for dataset: {dataset} must be a string."
        assert os.path.exists(config["DATA"]["DATASETS"][dataset]["PATH"]), f"Could not find path for dataset: {dataset}."
        #assert all(isinstance(split, float) for split in config["DATA"]["DATASETS"][dataset]["SPLITS"].values()), f"Config splits for dataset: {dataset} must be floats."
        #assert sum(config["DATA"]["DATASETS"][dataset]["SPLITS"].values()) <= 1.0, f"Train, validation and test split for dataset: {dataset} must total to less than 1.0."
    
    assert isinstance(config["DATA"]["PREDICTED_FEATURES"]["APCA_FITNESS"], bool), "Config predicted feature: aPCA fitness option must be a bool."
    assert isinstance(config["DATA"]["PREDICTED_FEATURES"]["CDNAPD_ENERGY"], bool), "Config predicted feature: cDNA-PD energy option must be a bool."
    assert sum(config["DATA"]["PREDICTED_FEATURES"].values()) > 0, ""
    assert isinstance(config["DATA"]["FILTERS"]["FILTER_ONE_WILDTYPE_PER_DOMAIN"], bool), "Config filters: filter one wildtype per domain option must be a bool."
    assert isinstance(config["DATA"]["FILTERS"]["EXCLUDE_WILDTYPE_INFERENCE"], bool), "Config filters: exclude wildtype from inference option must be a bool."
    
    # Upstream models
    assert all(isinstance(model, bool) for model in config["UPSTREAM_MODELS"]["MODELS"].values()), ""
    assert sum(config["UPSTREAM_MODELS"]["MODELS"].values()) > 0, "At least one model in upstream models must be true."
    assert all(isinstance(layer, int) for layer in config["UPSTREAM_MODELS"]["EMBEDDING_LAYERS"])
    assert all(isinstance(model, bool) for model in config["UPSTREAM_MODELS"]["EMBEDDING_POOL_TYPES"].values()), ""
    assert sum(config["UPSTREAM_MODELS"]["EMBEDDING_POOL_TYPES"].values()) > 0, "At least one embedding pool type must be true."
    assert isinstance(config["UPSTREAM_MODELS"]["POSTPROCESSING"]["NORMALISE_EMBEDDINGS"], bool), ""
    assert all(isinstance(method, bool) for method in config["UPSTREAM_MODELS"]["POSTPROCESSING"]["WILDTYPE_EMBEDDING"].values()), ""
    assert all(isinstance(method, bool) for method in config["UPSTREAM_MODELS"]["POSTPROCESSING"]["DIMENSIONAL_REDUCTION"].values()), ""
    assert isinstance(config["UPSTREAM_MODELS"]["POSTPROCESSING"]["N_DESIRED_DIMENSIONS"], int), ""
    
    # Downstream models
    assert all(isinstance(layer, int) for layer in config["DOWNSTREAM_MODELS"]["MODEL_TYPE"].values()), ""
    
def format_config(config):
    
    """
    Takes some of the more complex config settings and parses them into a more managable state for the rest of the package.
    """
    
    config["SUBSETS_IN_USE"] = [
        (dataset_name, label_name)
        for dataset_name, labels in config["DATA"]["DATASET_GROUPS"].items()
        for label_name, include in labels.items() if include
        ]
    config["SUBSETS_FOR_BENCHMARK_TRAINING"] = [
        (dataset_name, label_name)
        for dataset_name, labels in config["DATA"]["BENCHMARK_TRAINING_SUBSETS"].items()
        for label_name, include in labels.items() if include
        ]
    subsets_splits_dict = {}
    
    for dataset_name, label_name in config["SUBSETS_IN_USE"]:
        
        unique_key = f"{dataset_name}-{label_name}"
        label_config = config["DATA"]["DATASETS"][dataset_name]["LABEL_COLUMNS"][label_name]
        subsets_splits_dict[unique_key] = {
            "TRAIN": label_config["SPLITS"]["TRAIN"],
            "VALIDATION": label_config["SPLITS"]["VALIDATION"],
            "TEST": label_config["SPLITS"]["TEST"],
        }
    
    config["SUBSETS_SPLITS_DICT"] = subsets_splits_dict
    predicted_features = set()
    
    for dataset_name, label_name in config["SUBSETS_IN_USE"]:
        
        for feature in config["DATA"]["DATASETS"][dataset_name]["PREDICTED_FEATURE_COLUMNS"].keys():
                
            predicted_features.add(feature)

    config["PREDICTED_FEATURES_LIST"] = sorted(predicted_features)
    split_priorities = [key for key, value in config["DATA"]["SPLITS_PRIORITY"].items() if value]
    priority_split = split_priorities[0] if len(split_priorities) == 1 else None
    config["PRIORITY_SPLIT"] = priority_split
    config["UPSTREAM_MODELS_LIST"] = [key for key, value in config["UPSTREAM_MODELS"]["MODELS"].items() if value]
    config["EMBEDDING_POOL_TYPE_LIST"] = [key for key, value in config["UPSTREAM_MODELS"]["EMBEDDING_POOL_TYPES"].items() if value]
    config["DIMENSIONAL_REDUCTION_CHOICE"] = [key for key, value in config["UPSTREAM_MODELS"]["POSTPROCESSING"]["DIMENSIONAL_REDUCTION"].items() if value][0]
    config["DOWNSTREAM_MODELS_LIST"] = [key for key, value in config["DOWNSTREAM_MODELS"]["MODEL_TYPE"].items() if value]
    config["ACTIVATION_FUNCTIONS_LIST"] = [key for key, value in config["DOWNSTREAM_MODELS"]["MODEL_ARCHITECTURE"]["ACTIVATION_FUNCTIONS"].items() if value]
    config["LOSS_FUNCTIONS"] = [key for key, value in config["DOWNSTREAM_MODELS"]["MODEL_ARCHITECTURE"]["LOSS_FUNCTIONS"].items() if value]
    config["OPTIMISERS"] = [key for key, value in config["DOWNSTREAM_MODELS"]["MODEL_ARCHITECTURE"]["OPTIMISERS"].items() if value]
    
    assert len(config["PREDICTED_FEATURES_LIST"]) > 0, "Must have non-zero amount of predicted features."
    assert len(config["UPSTREAM_MODELS_LIST"]) > 0, "Must have non-zero amount of upstream models."
    assert len(config["EMBEDDING_POOL_TYPE_LIST"]) > 0, "Must have non-zero amount of embedding pool types."
    assert len(config["DOWNSTREAM_MODELS_LIST"]) > 0, "Must have non-zero amount of downstream models."
    assert len(config["ACTIVATION_FUNCTIONS_LIST"]) > 0, "Must have non-zero amount of activation functions."
    
    return config

config_path = Path(__file__).resolve().parent.parent / "config.json"
config = read_config(config_path)
preflight_checks(config)
config = format_config(config)