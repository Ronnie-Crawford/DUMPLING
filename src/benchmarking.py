# Standard modules
import datetime
import copy

# Local modules
from runner import setup_folders, train, test

def all_against_all_benchmarking(config: dict):
    
    """
    Runs benchmarking by training models on each dataset and evaluating them
    against every dataset. Results are stored clearly separated into training
    and testing directories.
    """

    # Setup
    base_results_folder = setup_folders()
    benchmarking_directory = create_benchmarking_directory(base_results_folder)
    trained_model_paths = train_phase(benchmarking_directory, config)
    test_phase(benchmarking_directory, config, trained_model_paths)
    
    return benchmarking_directory

def create_benchmarking_directory(package_folder):
    
    """
    Creates and returns a timestamped benchmarking directory under:
    package_folder/results/YYYY-M-D/HH:MM:SS
    """

    timestamp = datetime.datetime.now()
    date_folder = f"{timestamp.year}-{timestamp.month}-{timestamp.day}"
    time_folder = f"{timestamp.hour}:{timestamp.minute}:{timestamp.second}"
    benchmarking_root = (package_folder / "results" / date_folder / time_folder)
    benchmarking_root.mkdir(parents = True, exist_ok = True)

    return benchmarking_root

def train_phase(benchmarking_directory, config):
    
    """
    We train a model on each subset in subsets to use.
    We assign splits to each subset, if it is the subset we want to train on,
    we allow 80% of the data to be used to train a model,
    if it is not the training subset, we set all splits to 0%.
    This produces one config per subset, which is then used to run the train function.
    """
    
    training_results_folder = benchmarking_directory / "train"
    training_results_folder.mkdir(parents = True, exist_ok = True)
    trained_model_paths = {}

    #for dataset_name, label_name in config["SUBSETS_IN_USE"]:
    for dataset_name, label_name in [("APCA_WITHOUT_NEW_DATA", "ALL"), ("CDNA-DP", "ALL")]:

        training_subset_key = f"{dataset_name}-{label_name}"
        # Copy and modify training config
        training_config = copy.deepcopy(config)
        new_splits = {}

        for query_dataset_name, query_label_name in config["SUBSETS_IN_USE"]:

            query_subset_key = f"{query_dataset_name}-{query_label_name}"

            if query_subset_key == training_subset_key:

                new_splits[query_subset_key] = {
                    "TRAIN": 0.8,
                    "VALIDATION": 0.0,
                    "TEST": 0.0
                }

            else:

                new_splits[query_subset_key] = {
                    "TRAIN": 0.0,
                    "VALIDATION": 0.0,
                    "TEST": 0.0
                }
    
        # We end up with one config per subset to train on
        training_config["SUBSETS_SPLITS_DICT"] = new_splits
        training_dataset_folder = training_results_folder / training_subset_key
        training_dataset_folder.mkdir(parents = True, exist_ok = True)
        training_config["DOWNSTREAM_MODELS"]["PATH"] = training_dataset_folder
        train(training_config, results_path_override = training_dataset_folder)
        trained_model_paths[training_subset_key] = training_dataset_folder
    
    return trained_model_paths

def test_phase(benchmarking_directory, config, trained_model_paths):
    
    testing_results_folder = benchmarking_directory / "test"
    testing_results_folder.mkdir(parents = True, exist_ok = True)

    for training_subset_key, model_folder_path in trained_model_paths.items():

        for dataset_name, label_name in config["SUBSETS_IN_USE"]:
            
            testing_subset_key = f"{dataset_name}-{label_name}"
            
            # Copy and modify testing config
            testing_config = copy.deepcopy(config)
            new_splits = {}
            
            for query_dataset_name, query_label_name in config["SUBSETS_IN_USE"]:
                
                query_subset_key = f"{query_dataset_name}-{query_label_name}"

                if query_subset_key == testing_subset_key:

                    new_splits[query_subset_key] = {
                        "TRAIN": 0.0,
                        "VALIDATION": 0.0,
                        "TEST": 1.0
                    }

                else:

                    new_splits[query_subset_key] = {
                        "TRAIN": 0.0,
                        "VALIDATION": 0.0,
                        "TEST": 0.0
                    }
            
            testing_config["SUBSETS_SPLITS_DICT"] = new_splits
            testing_pair_folder_name = f"trained_on_{training_subset_key}_tested_on_{testing_subset_key}"
            testing_dataset_folder = testing_results_folder / testing_pair_folder_name
            testing_dataset_folder.mkdir(parents = True, exist_ok = True)
            testing_config["DOWNSTREAM_MODELS"]["PATH"] = model_folder_path
            test(testing_config, results_path_override = testing_dataset_folder)