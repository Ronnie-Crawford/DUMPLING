# Standard modules
import copy
import datetime

# Local modules
from config_loader import config
from runner import train, test
from helpers import setup_folders, get_benchmarking_results_path

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

def all_against_all_benchmarking(config: dict, test_label_groups: bool = True):
    
    """
    Runs benchmarking by training models on each dataset and evaluating them
    against every dataset. Results are stored clearly separated into training
    and testing directories.
    """

    # Setup
    base_results_folder = setup_folders()
    benchmarking_directory = create_benchmarking_directory(base_results_folder)
    trained_model_paths = {}
    
    if test_label_groups:
        
        datagroups_to_use = [f"{dataset}-{label}" for dataset, label in config["DATASETS_IN_USE"]]
        
    else:

        datagroups_to_use = [dataset for dataset, label in config["DATASETS_IN_USE"] if label == "ALL"]

    # Training phase
    training_results_folder = benchmarking_directory / "train"
    training_results_folder.mkdir(parents = True, exist_ok = True)

    for training_datagroup_key in datagroups_to_use:

        training_dataset_folder = training_results_folder / training_datagroup_key
        training_dataset_folder.mkdir(parents = True, exist_ok = True)

        # Copy and modify training config
        training_config = copy.deepcopy(config)

        for query_datagroup_key in datagroups_to_use:

            if query_datagroup_key == training_datagroup_key:

                splits = {
                    "TRAIN": 0.8,
                    "VALIDATION": 0.0,
                    "TEST": 0.2
                }

            else:

                splits = {
                    "TRAIN": 0.0,
                    "VALIDATION": 0.0,
                    "TEST": 0.0
                }

            dataset_name, label = query_datagroup_key.rsplit("-", 1)
            
            if label == "ALL":
                # Overwrite the base splits on the dataset itself
                training_config["DATA"]["DATASETS"][dataset_name]["SPLITS"] = splits
                
            else:
                # Overwrite the splits of that LABEL in LABEL_COLUMNS
                training_config["DATA"]["DATASETS"][dataset_name]["LABEL_COLUMNS"][label]["SPLITS"] = splits

        # Set explicit training results path and train
        training_config["DOWNSTREAM_MODELS"]["PATH"] = training_dataset_folder
        train(training_config, results_path_override = training_dataset_folder)
        trained_model_paths[training_datagroup_key] = training_dataset_folder

    # Testing phase
    testing_results_folder = benchmarking_directory / "test"
    testing_results_folder.mkdir(parents = True, exist_ok = True)

    for training_datagroup_key, model_folder_path in trained_model_paths.items():

        for testing_datagroup_key in datasets_to_use:

            testing_pair_folder_name = f"trained_on_{training_datagroup_key}_tested_on_{testing_datagroup_key}"
            testing_dataset_folder = testing_results_folder / testing_pair_folder_name
            testing_dataset_folder.mkdir(parents = True, exist_ok = True)
            
            # Copy and modify testing config
            testing_config = copy.deepcopy(config)

            for query_datagroup_key in datasets_to_use:

                if query_datagroup_key == testing_datagroup_key:

                    testing_split_fraction = 0.2 if testing_datagroup_key == training_datagroup_key else 1.0
                    splits = {
                        "TRAIN": 0.0,
                        "VALIDATION": 0.0,
                        "TEST": testing_split_fraction
                    }

                else:

                    splits = {
                        "TRAIN": 0.0,
                        "VALIDATION": 0.0,
                        "TEST": 0.0
                    }

                dataset_name, label = training_datagroup_key.rsplit("-", 1)
                
                if label == "ALL":
                    
                    testing_config["DATA"]["DATASETS"][dataset_name]["SPLITS"] = splits
                    
                else:
                
                    testing_config["DATA"]["DATASETS"][dataset_name]["LABEL_COLUMNS"][label]["SPLITS"] = splits

            # Set the trained model's path to load and test
            testing_config["DOWNSTREAM_MODELS"]["PATH"] = model_folder_path
            test(testing_config, results_path_override = testing_dataset_folder)

    return benchmarking_directory