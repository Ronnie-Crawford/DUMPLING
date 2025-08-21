# Standard modules
import os
import multiprocessing
import datetime
import shutil
from pathlib import Path
import gc

# Third-party modules
import torch

# Local modules
from datasets import handle_data, handle_filtering, add_spoof_train_dataset
from embedding import handle_embeddings
from homology import handle_homology
from splits import handle_splits, remove_homologous_sequences_from_inference, load_domain_splits_from_file
from models import handle_models
from training import handle_training_models, load_trained_model
from inference import handle_inference
from metrics import handle_metrics, compute_metrics_per_subset
from visuals import handle_visuals

# Global variables
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
MAX_FILENAME_LENGTH = 255
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def train_and_test(config, results_path_override = None):

    device = get_device()
    n_workers = get_n_workers()
    paths_dict = {"base": setup_folders()}

    if results_path_override == None:

        paths_dict["results"] = get_results_path(paths_dict["base"])

    else:

        paths_dict["results"] = results_path_override

    paths_dict["splits"] = config["DATA"]["SPLITS_FILE"]["PATH"]

    print("Data")
    unique_datasets_dict = handle_data(
        paths_dict["base"],
        config["SUBSETS_IN_USE"],
        config["DATA"]["DATASETS"],
        config["DATA"]["FILTERS"]["FILTER_ONE_WILDTYPE_PER_DOMAIN"],
        config["PREDICTED_FEATURES_LIST"],
        config["DATA"]["FEATURE_REMAPPING"],
        AMINO_ACIDS
        )

    print("Embeddings")
    unique_datasets_dict, embedding_size = handle_embeddings(
        unique_datasets_dict,
        config["UPSTREAM_MODELS_LIST"],
        config["DOWNSTREAM_MODELS"]["TRAINING_PARAMETERS"]["BATCH_SIZE"],
        config["UPSTREAM_MODELS"]["TOKENISATION"]["SPECIAL_TOKENS_IN_CONTEXT"],
        config["UPSTREAM_MODELS"]["TOKENISATION"]["SPECIAL_TOKENS_IN_POOLING"],
        config["UPSTREAM_MODELS"]["EMBEDDING_LAYERS"],
        config["EMBEDDING_POOL_TYPE_LIST"],
        config["DIMENSIONAL_REDUCTION_CHOICE"],
        config["UPSTREAM_MODELS"]["POSTPROCESSING"]["N_DESIRED_DIMENSIONS"],
        config["UPSTREAM_MODELS"]["POSTPROCESSING"]["NORMALISE_EMBEDDINGS"],
        config["UPSTREAM_MODELS"]["POSTPROCESSING"]["WILDTYPE_EMBEDDING"]["CONCAT"],
        config["UPSTREAM_MODELS"]["POSTPROCESSING"]["WILDTYPE_EMBEDDING"]["DELTA"],
        paths_dict["base"],
        device,
        n_workers,
        AMINO_ACIDS
        )

    print("Filtering")
    dataset_dicts = handle_filtering(unique_datasets_dict, config["SUBSETS_IN_USE"])

    print("Homology")
    paths_dict["homology"] = handle_homology(
        dataset_dicts,
        paths_dict["base"]
        )

    print("Splits")
    dataloaders_dict = handle_splits(
        dataset_dicts,
        config["SUBSETS_SPLITS_DICT"],
        config["DATA"]["FILTERS"]["EXCLUDE_WILDTYPE_INFERENCE"],
        config["PREDICTED_FEATURES_LIST"],
        config["DATA"]["SAMPLING"]["SAMPLING_FLAG"],
        config["DATA"]["SAMPLING"]["SUBSET_WEIGHTING_FLAG"],
        config["DATA"]["SAMPLING"]["SEVERITY_WEIGHTING_FLAG"],
        config["DATA"]["SAMPLING"]["RELIABILITY_WEIGHTING_FLAG"],
        config["PRIORITY_SPLIT"],
        config["DATA"]["SPLITS_BIASES"]["TRAIN_BIAS"],
        config["DATA"]["SPLITS_BIASES"]["VALIDATION_BIAS"],
        config["DATA"]["SPLITS_BIASES"]["TEST_BIAS"],
        config["DOWNSTREAM_MODELS"]["TRAINING_PARAMETERS"]["BATCH_SIZE"],
        n_workers,
        paths_dict["homology"],
        paths_dict["splits"],
        config["DATA"]["SPLITS_FILE"]["SAVE_SPLITS"],
        config["DATA"]["SPLITS_FILE"]["LOAD_SPLITS"]
        )

    print("Models")
    model, criterion, optimiser = handle_models(
        dataloaders_dict,
        config["DOWNSTREAM_MODELS_LIST"],
        embedding_size,
        config["PREDICTED_FEATURES_LIST"],
        config["DOWNSTREAM_MODELS"]["MODEL_ARCHITECTURE"]["HIDDEN_LAYERS"],
        config["DOWNSTREAM_MODELS"]["MODEL_ARCHITECTURE"]["HEAD_LAYERS"],
        config["DOWNSTREAM_MODELS"]["MODEL_ARCHITECTURE"]["DROPOUT_LAYERS"],
        config["DOWNSTREAM_MODELS"]["TRAINING_PARAMETERS"]["LEARNING_RATE"],
        config["DOWNSTREAM_MODELS"]["MODEL_ARCHITECTURE"]["WEIGHT_DECAY"],
        config["ACTIVATION_FUNCTIONS_LIST"],
        config["LOSS_FUNCTIONS"],
        config["OPTIMISERS"],
        paths_dict["results"],
        device
        )

    print("Training")
    trained_model = handle_training_models(
        dataloaders_dict,
        model,
        criterion,
        optimiser,
        config["PREDICTED_FEATURES_LIST"],
        embedding_size,
        config["DOWNSTREAM_MODELS_LIST"],
        config["DOWNSTREAM_MODELS"]["TRAINING_PARAMETERS"]["MIN_EPOCHS"],
        config["DOWNSTREAM_MODELS"]["TRAINING_PARAMETERS"]["MAX_EPOCHS"],
        config["DOWNSTREAM_MODELS"]["TRAINING_PARAMETERS"]["PATIENCE"],
        device,
        paths_dict["results"]
        )

    print("Inference")
    predictions_df = handle_inference(
        dataloaders_dict,
        config["DOWNSTREAM_MODELS_LIST"],
        config["PREDICTED_FEATURES_LIST"],
        trained_model,
        criterion,
        config["DOWNSTREAM_MODELS"]["TRAINING_PARAMETERS"]["BATCH_SIZE"],
        device,
        paths_dict["results"]
        )

    print("Metrics")
    overall_metrics, domain_specific_metrics = handle_metrics(
       config["PREDICTED_FEATURES_LIST"],
       paths_dict["results"]
       )
    metrics_by_subset = compute_metrics_per_subset(
        config["PREDICTED_FEATURES_LIST"],
        paths_dict["results"]
        )

    print("Visuals")
    handle_visuals(
        predictions_df,
        metrics_by_subset,
        config["SUBSETS_IN_USE"],
        config["PREDICTED_FEATURES_LIST"],
        paths_dict["results"]
        )

def train(config, results_path_override = None):

    device = get_device()
    n_workers = get_n_workers()
    paths_dict = {"base": setup_folders()}

    if results_path_override == None:

        paths_dict["results"] = get_results_path(paths_dict["base"])

    else:

        paths_dict["results"] = results_path_override

    print("Data")
    dataset_dicts = handle_data(
        paths_dict["base"],
        config["SUBSETS_IN_USE"],
        config["DATA"]["DATASETS"],
        config["DATA"]["FILTERS"]["FILTER_ONE_WILDTYPE_PER_DOMAIN"],
        config["PREDICTED_FEATURES_LIST"],
        config["DATA"]["FEATURE_REMAPPING"],
        AMINO_ACIDS
        )

    print("Embeddings")
    dataset_dicts, embedding_size = handle_embeddings(
        dataset_dicts,
        config["UPSTREAM_MODELS_LIST"],
        config["DOWNSTREAM_MODELS"]["TRAINING_PARAMETERS"]["BATCH_SIZE"],
        config["UPSTREAM_MODELS"]["EMBEDDING_LAYERS"],
        config["EMBEDDING_POOL_TYPE_LIST"],
        config["DIMENSIONAL_REDUCTION_CHOICE"],
        config["UPSTREAM_MODELS"]["POSTPROCESSING"]["N_DESIRED_DIMENSIONS"],
        config["UPSTREAM_MODELS"]["POSTPROCESSING"]["NORMALISE_EMBEDDINGS"],
        config["UPSTREAM_MODELS"]["POSTPROCESSING"]["WILDTYPE_EMBEDDING"]["CONCAT"],
        config["UPSTREAM_MODELS"]["POSTPROCESSING"]["WILDTYPE_EMBEDDING"]["DELTA"],
        paths_dict["base"],
        device,
        n_workers
        )

    print("Filtering")
    dataset_dicts = handle_filtering(dataset_dicts)

    print("Homology")
    paths_dict["homology"] = handle_homology(
        dataset_dicts,
        paths_dict["base"],
        force_regeneration = True
        )

    print("Splits")
    dataloaders_dict, test_subset_to_sequence_dict = handle_splits(
        dataset_dicts,
        config["SUBSETS_SPLITS_DICT"],
        config["DATA"]["FILTERS"]["EXCLUDE_WILDTYPE_INFERENCE"],
        config["DATA"]["FILTERS"]["UPSAMPLE_TRAINING_SUBSETS"],
        config["PRIORITY_SPLIT"],
        config["DATA"]["SPLITS_BIASES"]["TRAIN_BIAS"],
        config["DATA"]["SPLITS_BIASES"]["VALIDATION_BIAS"],
        config["DATA"]["SPLITS_BIASES"]["TEST_BIAS"],
        config["DOWNSTREAM_MODELS"]["TRAINING_PARAMETERS"]["BATCH_SIZE"],
        n_workers,
        paths_dict["homology"],
        paths_dict["splits"],
        config["DATA"]["SPLITS_FILE"]["SAVE_SPLITS"],
        config["DATA"]["SPLITS_FILE"]["LOAD_SPLITS"]
        )

    print("Models")
    model, criterion, optimiser = handle_models(
        dataloaders_dict,
        config["DOWNSTREAM_MODELS_LIST"],
        embedding_size,
        config["PREDICTED_FEATURES_LIST"],
        config["DOWNSTREAM_MODELS"]["MODEL_ARCHITECTURE"]["HIDDEN_LAYERS"],
        config["DOWNSTREAM_MODELS"]["MODEL_ARCHITECTURE"]["HEAD_LAYERS"],
        config["DOWNSTREAM_MODELS"]["MODEL_ARCHITECTURE"]["DROPOUT_LAYERS"],
        config["DOWNSTREAM_MODELS"]["TRAINING_PARAMETERS"]["LEARNING_RATE"],
        config["DOWNSTREAM_MODELS"]["MODEL_ARCHITECTURE"]["WEIGHT_DECAY"],
        config["ACTIVATION_FUNCTIONS_LIST"],
        config["LOSS_FUNCTIONS"],
        config["OPTIMISERS"],
        paths_dict["results"],
        device
        )

    print("Training")
    trained_model = handle_training_models(
        dataloaders_dict,
        model,
        criterion,
        optimiser,
        config["PREDICTED_FEATURES_LIST"],
        embedding_size,
        config["DOWNSTREAM_MODELS_LIST"],
        config["DOWNSTREAM_MODELS"]["TRAINING_PARAMETERS"]["MIN_EPOCHS"],
        config["DOWNSTREAM_MODELS"]["TRAINING_PARAMETERS"]["MAX_EPOCHS"],
        config["DOWNSTREAM_MODELS"]["TRAINING_PARAMETERS"]["PATIENCE"],
        device,
        paths_dict["results"]
        )

    gc.collect()

def test(config, results_path_override = None):

    device = get_device()
    n_workers = get_n_workers()
    paths_dict = {"base": setup_folders()}

    if results_path_override == None:

        paths_dict["results"] = get_results_path(paths_dict["base"])

    else:

        paths_dict["results"] = results_path_override

    print("Data")
    dataset_dicts = handle_data(
        paths_dict["base"],
        config["SUBSETS_IN_USE"],
        config["DATA"]["DATASETS"],
        config["DATA"]["FILTERS"]["FILTER_ONE_WILDTYPE_PER_DOMAIN"],
        config["PREDICTED_FEATURES_LIST"],
        config["DATA"]["FEATURE_REMAPPING"],
        AMINO_ACIDS
        )

    print("Embeddings")
    dataset_dicts, embedding_size = handle_embeddings(
        dataset_dicts,
        config["UPSTREAM_MODELS_LIST"],
        config["DOWNSTREAM_MODELS"]["TRAINING_PARAMETERS"]["BATCH_SIZE"],
        config["UPSTREAM_MODELS"]["EMBEDDING_LAYERS"],
        config["EMBEDDING_POOL_TYPE_LIST"],
        config["DIMENSIONAL_REDUCTION_CHOICE"],
        config["UPSTREAM_MODELS"]["POSTPROCESSING"]["N_DESIRED_DIMENSIONS"],
        config["UPSTREAM_MODELS"]["POSTPROCESSING"]["NORMALISE_EMBEDDINGS"],
        config["UPSTREAM_MODELS"]["POSTPROCESSING"]["WILDTYPE_EMBEDDING"]["CONCAT"],
        config["UPSTREAM_MODELS"]["POSTPROCESSING"]["WILDTYPE_EMBEDDING"]["DELTA"],
        paths_dict["base"],
        device,
        n_workers
        )

    print("Filtering")
    dataset_dicts = handle_filtering(dataset_dicts)

    print("Adding spoof train dataset")
    dataset_dicts = add_spoof_train_dataset(dataset_dicts, config["DOWNSTREAM_MODELS"]["PATH"], config["PREDICTED_FEATURES_LIST"])

    print("Homology")
    paths_dict["homology"] = handle_homology(
        dataset_dicts,
        paths_dict["base"],
        force_regeneration = True
        )

    print("Remove leakage from inference")
    dataset_dicts = remove_homologous_sequences_from_inference(
        dataset_dicts,
        paths_dict["homology"]
        )

    print("Remove spoof training data")
    dataset_dicts = [dataset_dict for dataset_dict in dataset_dicts if dataset_dict["unique_key"] != "spoof_training_dataset"]

    print("Splits")
    dataloaders_dict, test_subset_to_sequence_dict = handle_splits(
        dataset_dicts,
        config["SUBSETS_SPLITS_DICT"],
        config["DATA"]["FILTERS"]["EXCLUDE_WILDTYPE_INFERENCE"],
        config["DATA"]["FILTERS"]["UPSAMPLE_TRAINING_SUBSETS"],
        config["PRIORITY_SPLIT"],
        config["DATA"]["SPLITS_BIASES"]["TRAIN_BIAS"],
        config["DATA"]["SPLITS_BIASES"]["VALIDATION_BIAS"],
        config["DATA"]["SPLITS_BIASES"]["TEST_BIAS"],
        config["DOWNSTREAM_MODELS"]["TRAINING_PARAMETERS"]["BATCH_SIZE"],
        n_workers,
        paths_dict["homology"],
        paths_dict["splits"],
        config["DATA"]["SPLITS_FILE"]["SAVE_SPLITS"],
        config["DATA"]["SPLITS_FILE"]["LOAD_SPLITS"]
        )

    print("Models")
    model, criterion, optimiser = handle_models(
        dataloaders_dict,
        config["DOWNSTREAM_MODELS_LIST"],
        embedding_size,
        config["PREDICTED_FEATURES_LIST"],
        config["DOWNSTREAM_MODELS"]["MODEL_ARCHITECTURE"]["HIDDEN_LAYERS"],
        config["DOWNSTREAM_MODELS"]["MODEL_ARCHITECTURE"]["HEAD_LAYERS"],
        config["DOWNSTREAM_MODELS"]["MODEL_ARCHITECTURE"]["DROPOUT_LAYERS"],
        config["DOWNSTREAM_MODELS"]["TRAINING_PARAMETERS"]["LEARNING_RATE"],
        config["DOWNSTREAM_MODELS"]["MODEL_ARCHITECTURE"]["WEIGHT_DECAY"],
        config["ACTIVATION_FUNCTIONS_LIST"],
        config["LOSS_FUNCTIONS"],
        config["OPTIMISERS"],
        paths_dict["results"],
        device
        )

    trained_model = load_trained_model(
        model,
        config["DOWNSTREAM_MODELS"]["PATH"],
        device
        )

    print("Inference")
    predictions_df = handle_inference(
        dataloaders_dict,
        config["DOWNSTREAM_MODELS_LIST"],
        config["PREDICTED_FEATURES_LIST"],
        trained_model,
        criterion,
        config["DOWNSTREAM_MODELS"]["TRAINING_PARAMETERS"]["BATCH_SIZE"],
        test_subset_to_sequence_dict,
        device,
        paths_dict["results"]
        )

    print("Metrics")
    overall_metrics, domain_specific_metrics = handle_metrics(
        config["PREDICTED_FEATURES_LIST"],
        paths_dict["results"]
        )

    gc.collect()

def get_device() -> torch.device:

    """
    Determines the best available device (GPU, MPS, or CPU).
    Want to add XPU/NPU support but drivers are a nightmare so far.

    Returns:

        - torch.device: The best available device.
    """

    if torch.cuda.is_available():

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        return torch.device("cuda")

    elif torch.backends.mps.is_available():

        return torch.device("mps")

    else:

        return torch.device("cpu")

def get_n_workers():

    """
    Bit of a hacky attempt to judge the number of CPU cores available,
    looks for "SLURM_JOB_ID" to see if its running on HPC, which might not enjoy having number of cores set,
    otherwise checks max number of cores on device and takes 1 lower, or 1.
    """

    if os.environ.get("SLURM_JOB_ID") is not None:

        n_workers = 0

    else:

        n_workers = max(1, multiprocessing.cpu_count() // 2)

    return n_workers

def setup_folders() -> Path:

    """

    """

    package_folder = Path(__file__).resolve().parent.parent
    directories = ["embeddings", "homology", "models", "results", "splits"]

    for directory in directories:

        directory_path = package_folder / directory
        directory_path.mkdir(parents = True, exist_ok = True)

    return package_folder

def get_results_path(package_folder):

    timestamp = datetime.datetime.now()
    results_path = package_folder / "results" / (str(timestamp.year) + "-" + str(timestamp.month) + "-" + str(timestamp.day)) / (str(timestamp.hour) + ":" + str(timestamp.minute) + ":" + str(timestamp.second))
    results_path.mkdir(parents = True, exist_ok = True)
    shutil.copy((package_folder / "config.json"), (results_path / "config.json"))

    return Path(safe_filename(results_path))

def safe_filename(path):

    dirpath, name = os.path.split(path)
    stem, ext = os.path.splitext(name)

    if len(name) <= MAX_FILENAME_LENGTH:

        return path

    # Remove vowels from the stem first
    vowels = set("aeiouAEIOU")
    no_vowel = "".join(ch for ch in stem if ch not in vowels)

    # If still too long, truncate
    truncated = no_vowel[: MAX_FILENAME_LENGTH - len(ext)]

    safe_name = truncated + ext
    safe_name = os.path.join(dirpath, safe_name)

    return safe_name
