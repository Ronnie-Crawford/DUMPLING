# Standard modules
import datetime
import copy
from pathlib import Path

# Local modules
from runner import setup_folders, train_and_test, train, test

def all_against_all_benchmarking(config: dict) -> Path:
    
    # Setup
    base_results_folder = setup_folders()
    benchmarking_directory = create_benchmarking_directory(base_results_folder)
    
    #for train_dataset_name, train_label_name in config["SUBSETS_IN_USE"]:
    #for train_dataset_name, train_label_name in [("APCA_WITHOUT_NEW_DATA", "ALL"), ("CDNA-DP", "ALL")]:
    for train_dataset_name, train_label_name in [("APCA_WITHOUT_NEW_DATA", "ALL")]:
        
        training_key = f"{train_dataset_name}-{train_label_name}"
        # Copy and modify training config
        run_config = copy.deepcopy(config)
        new_splits = {}
        
        for query_dataset_name, query_label_name in config["SUBSETS_IN_USE"]:
            
            query_subset_key = f"{query_dataset_name}-{query_label_name}"
            
            if query_subset_key == training_key:
                
                new_splits[query_subset_key] = {"TRAIN": 0.8, "VALIDATION": 0.0, "TEST": 0.2}
                
            else:
                
                new_splits[query_subset_key] = {"TRAIN": 0.0, "VALIDATION": 0.0, "TEST": 1.0}
        
        run_config["SUBSETS_SPLITS_DICT"] = new_splits
        run_folder = benchmarking_directory / f"trained_on_{training_key}"
        run_folder.mkdir(parents = True, exist_ok = True)
        run_config["DOWNSTREAM_MODELS"]["PATH"] = run_folder
        train_and_test(run_config, results_path_override = run_folder)
        
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