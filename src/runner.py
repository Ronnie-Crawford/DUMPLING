# Local modules
from datasets import handle_data, make_spoof_train_dataset, handle_filtering
from homology import handle_homology
from embedding import handle_embeddings
from splits import handle_splits, load_training_sequences
from models import handle_models
from training import handle_training_models, load_trained_model
from inference import handle_inference
from helpers import get_device, setup_folders, get_results_path, handle_setup, remove_homologous_sequences_from_inference, get_n_workers

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

"""
Idea is that this module contains the building-blocks of pipelines that can be called from main.
This module sees config, other modules should be config agnostic.

To do:
- Add handle_visualisation
- Move handle_tuning from main to runner
"""

def train_and_test(config):
    
    device = get_device(None)
    n_workers = get_n_workers()
    paths_dict = {"base": setup_folders()}
    paths_dict["results"] = get_results_path(paths_dict["base"])
    
    dataset_dicts = handle_data(
        paths_dict["base"],
        config["DATASETS_IN_USE"],
        config["DATA"]["DATASETS"],
        config["DATA"]["FILTERS"]["FILTER_ONE_WILDTYPE_PER_DOMAIN"],
        config["PREDICTED_FEATURES_LIST"],
        AMINO_ACIDS
        )

    dataset_dicts, embedding_size = handle_embeddings(
        config["UPSTREAM_MODELS_LIST"],
        config["DOWNSTREAM_MODELS"]["TRAINING_PARAMETERS"]["BATCH_SIZE"],
        config["UPSTREAM_MODELS"]["EMBEDDING_LAYERS"],
        config["EMBEDDING_POOL_TYPE_LIST"],
        config["DIMENSIONAL_REDUCTION_CHOICE"],
        config["UPSTREAM_MODELS"]["POSTPROCESSING"]["N_DESIRED_DIMENSIONS"],
        config["UPSTREAM_MODELS"]["POSTPROCESSING"]["WILDTYPE_EMBEDDING"]["CONCAT"],
        config["UPSTREAM_MODELS"]["POSTPROCESSING"]["WILDTYPE_EMBEDDING"]["DELTA"],
        config["DOWNSTREAM_MODELS_LIST"],
        dataset_dicts,
        device,
        paths_dict["base"],
        n_workers
        )
    
    dataset_dicts = handle_filtering(
        dataset_dicts
        )

    paths_dict["homology"] = handle_homology(
        dataset_dicts,
        paths_dict["base"],
        config["SPLITS_METHOD_CHOICE"]
        )
    
    dataloaders_dict = handle_splits(
        config["SPLITS_PRIORITY_CHOICE"],
        config["SPLITS_METHOD_CHOICE"],
        config["DATA"]["FILTERS"]["EXCLUDE_WILDTYPE_INFERENCE"],
        config["DOWNSTREAM_MODELS"]["TRAINING_PARAMETERS"]["BATCH_SIZE"],
        n_workers,
        dataset_dicts,
        config["DATASETS_SPLITS_DICT"],
        paths_dict["homology"],
        paths_dict["results"]
        )
    
    model, criterion, optimiser = handle_models(
        config["DOWNSTREAM_MODELS"]["MODEL_ARCHITECTURE"]["HIDDEN_LAYERS"],
        config["DOWNSTREAM_MODELS"]["MODEL_ARCHITECTURE"]["DROPOUT_LAYERS"],
        config["DOWNSTREAM_MODELS"]["TRAINING_PARAMETERS"]["LEARNING_RATE"],
        config["DOWNSTREAM_MODELS"]["MODEL_ARCHITECTURE"]["WEIGHT_DECAY"],
        config["DOWNSTREAM_MODELS"]["TRAINING_PARAMETERS"]["MIN_EPOCHS"],
        config["DOWNSTREAM_MODELS"]["TRAINING_PARAMETERS"]["MAX_EPOCHS"],
        config["DOWNSTREAM_MODELS"]["TRAINING_PARAMETERS"]["PATIENCE"],
        config["DOWNSTREAM_MODELS_LIST"],
        embedding_size,
        dataloaders_dict,
        config["PREDICTED_FEATURES_LIST"],
        config["ACTIVATION_FUNCTIONS_LIST"],
        config["LOSS_FUNCTIONS"],
        config["OPTIMISERS"],
        #rnn_type,
        #bidirectional,
        paths_dict["results"],
        device
        )
    
    trained_model = handle_training_models(
        config["DOWNSTREAM_MODELS_LIST"][0],    # For now we only use one downstream model, maybe in the future will add multiple
        model,
        dataloaders_dict,
        config["PREDICTED_FEATURES_LIST"],
        criterion,
        optimiser,
        paths_dict["results"],
        config["DOWNSTREAM_MODELS"]["TRAINING_PARAMETERS"]["MIN_EPOCHS"],
        config["DOWNSTREAM_MODELS"]["TRAINING_PARAMETERS"]["MAX_EPOCHS"],
        config["DOWNSTREAM_MODELS"]["TRAINING_PARAMETERS"]["PATIENCE"],
        device,
        embedding_size
    )
    
    predictions_df, overall_metrics, domain_specific_metrics = handle_inference(
        config["DOWNSTREAM_MODELS"]["TRAINING_PARAMETERS"]["BATCH_SIZE"],
        config["DOWNSTREAM_MODELS_LIST"],
        trained_model,
        dataloaders_dict,
        criterion,
        device,
        config["PREDICTED_FEATURES_LIST"],
        paths_dict["results"]
        )
    
def train(config, results_path_override = None):
    
    device = get_device(None)
    n_workers = get_n_workers()
    paths_dict = {"base": setup_folders()}
    
    if results_path_override == None:
        
        paths_dict["results"] = get_results_path(paths_dict["base"])
    
    else:
        
        paths_dict["results"] = results_path_override
    
    dataset_dicts = handle_data(
        paths_dict["base"],
        config["DATASETS_IN_USE"],
        config["DATA"]["DATASETS"],
        config["DATA"]["FILTERS"]["FILTER_ONE_WILDTYPE_PER_DOMAIN"],
        config["PREDICTED_FEATURES_LIST"],
        AMINO_ACIDS
        )

    dataset_dicts, embedding_size = handle_embeddings(
        config["UPSTREAM_MODELS_LIST"],
        config["DOWNSTREAM_MODELS"]["TRAINING_PARAMETERS"]["BATCH_SIZE"],
        config["UPSTREAM_MODELS"]["EMBEDDING_LAYERS"],
        config["EMBEDDING_POOL_TYPE_LIST"],
        config["DIMENSIONAL_REDUCTION_CHOICE"],
        config["UPSTREAM_MODELS"]["POSTPROCESSING"]["N_DESIRED_DIMENSIONS"],
        config["UPSTREAM_MODELS"]["POSTPROCESSING"]["WILDTYPE_EMBEDDING"]["CONCAT"],
        config["UPSTREAM_MODELS"]["POSTPROCESSING"]["WILDTYPE_EMBEDDING"]["DELTA"],
        config["DOWNSTREAM_MODELS_LIST"],
        dataset_dicts,
        device,
        paths_dict["base"],
        n_workers
        )
    
    dataset_dicts = handle_filtering(
        dataset_dicts
        )

    paths_dict["homology"] = handle_homology(
        dataset_dicts,
        paths_dict["base"],
        config["SPLITS_METHOD_CHOICE"]
        )
    
    dataloaders_dict = handle_splits(
        config["SPLITS_PRIORITY_CHOICE"],
        config["SPLITS_METHOD_CHOICE"],
        config["DATA"]["FILTERS"]["EXCLUDE_WILDTYPE_INFERENCE"],
        config["DOWNSTREAM_MODELS"]["TRAINING_PARAMETERS"]["BATCH_SIZE"],
        n_workers,
        dataset_dicts,
        config["DATASETS_SPLITS_DICT"],
        paths_dict["homology"],
        paths_dict["results"]
        )
    
    model, criterion, optimiser = handle_models(
        config["DOWNSTREAM_MODELS"]["MODEL_ARCHITECTURE"]["HIDDEN_LAYERS"],
        config["DOWNSTREAM_MODELS"]["MODEL_ARCHITECTURE"]["DROPOUT_LAYERS"],
        config["DOWNSTREAM_MODELS"]["TRAINING_PARAMETERS"]["LEARNING_RATE"],
        config["DOWNSTREAM_MODELS"]["MODEL_ARCHITECTURE"]["WEIGHT_DECAY"],
        config["DOWNSTREAM_MODELS"]["TRAINING_PARAMETERS"]["MIN_EPOCHS"],
        config["DOWNSTREAM_MODELS"]["TRAINING_PARAMETERS"]["MAX_EPOCHS"],
        config["DOWNSTREAM_MODELS"]["TRAINING_PARAMETERS"]["PATIENCE"],
        config["DOWNSTREAM_MODELS_LIST"],
        embedding_size,
        dataloaders_dict,
        config["PREDICTED_FEATURES_LIST"],
        config["ACTIVATION_FUNCTIONS_LIST"],
        config["LOSS_FUNCTIONS"],
        config["OPTIMISERS"],
        #rnn_type,
        #bidirectional,
        paths_dict["results"],
        device
        )
    
    trained_model = handle_training_models(
        config["DOWNSTREAM_MODELS_LIST"][0],    # For now we only use one downstream model, maybe in the future will add multiple
        model,
        dataloaders_dict,
        config["PREDICTED_FEATURES_LIST"],
        criterion,
        optimiser,
        paths_dict["results"],
        config["DOWNSTREAM_MODELS"]["TRAINING_PARAMETERS"]["MIN_EPOCHS"],
        config["DOWNSTREAM_MODELS"]["TRAINING_PARAMETERS"]["MAX_EPOCHS"],
        config["DOWNSTREAM_MODELS"]["TRAINING_PARAMETERS"]["PATIENCE"],
        device,
        embedding_size
    )

def test(config, results_path_override = None):
    
    device = get_device(None)
    n_workers = get_n_workers()
    paths_dict = {"base": setup_folders()}
    
    if results_path_override == None:
        
        paths_dict["results"] = get_results_path(paths_dict["base"])
    
    else:
        
        paths_dict["results"] = results_path_override
    
    dataset_dicts = handle_data(
        paths_dict["base"],
        config["DATA"]["DATASETS"],
        config["DATA"]["FILTERS"]["FILTER_ONE_WILDTYPE_PER_DOMAIN"],
        config["PREDICTED_FEATURES_LIST"],
        AMINO_ACIDS
        )
    
    dataset_dicts, embedding_size = handle_embeddings(
        config["UPSTREAM_MODELS_LIST"],
        config["DOWNSTREAM_MODELS"]["TRAINING_PARAMETERS"]["BATCH_SIZE"],
        config["UPSTREAM_MODELS"]["EMBEDDING_LAYERS"],
        config["EMBEDDING_POOL_TYPE_LIST"],
        config["DIMENSIONAL_REDUCTION_CHOICE"],
        config["UPSTREAM_MODELS"]["POSTPROCESSING"]["N_DESIRED_DIMENSIONS"],
        config["UPSTREAM_MODELS"]["POSTPROCESSING"]["WILDTYPE_EMBEDDING"]["CONCAT"],
        config["UPSTREAM_MODELS"]["POSTPROCESSING"]["WILDTYPE_EMBEDDING"]["DELTA"],
        config["DOWNSTREAM_MODELS_LIST"],
        dataset_dicts,
        device,
        paths_dict["base"],
        n_workers
        )
    
    dataset_dicts = handle_filtering(
        dataset_dicts
        )
    
    datasets_dict["spoof_training_dataset"] = make_spoof_train_dataset(
        load_training_sequences(config["DOWNSTREAM_MODELS"]["PATH"]),
        config["PREDICTED_FEATURES_LIST"]
        )
    
    paths_dict["homology"] = handle_homology(
        dataset_dicts,
        paths_dict["base"],
        config["SPLITS_METHOD_CHOICE"]
        )
    
    dataset_dicts = remove_homologous_sequences_from_inference(
        dataset_dicts,
        paths_dict["homology"]
        )
    
    dataloaders_dict = handle_splits(
        config["SPLITS_PRIORITY_CHOICE"],
        config["SPLITS_METHOD_CHOICE"],
        config["DATA"]["FILTERS"]["EXCLUDE_WILDTYPE_INFERENCE"],
        config["DOWNSTREAM_MODELS"]["TRAINING_PARAMETERS"]["BATCH_SIZE"],
        n_workers,
        dataset_dicts,
        config["DATASETS_SPLITS_DICT"],
        paths_dict["homology"],
        paths_dict["results"]
        )
    
    model, criterion, optimiser = handle_models(
        config["DOWNSTREAM_MODELS"]["MODEL_ARCHITECTURE"]["HIDDEN_LAYERS"],
        config["DOWNSTREAM_MODELS"]["MODEL_ARCHITECTURE"]["DROPOUT_LAYERS"],
        config["DOWNSTREAM_MODELS"]["TRAINING_PARAMETERS"]["LEARNING_RATE"],
        config["DOWNSTREAM_MODELS"]["MODEL_ARCHITECTURE"]["WEIGHT_DECAY"],
        config["DOWNSTREAM_MODELS"]["TRAINING_PARAMETERS"]["MIN_EPOCHS"],
        config["DOWNSTREAM_MODELS"]["TRAINING_PARAMETERS"]["MAX_EPOCHS"],
        config["DOWNSTREAM_MODELS"]["TRAINING_PARAMETERS"]["PATIENCE"],
        config["DOWNSTREAM_MODELS_LIST"],
        embedding_size,
        dataloaders_dict,
        config["PREDICTED_FEATURES_LIST"],
        config["ACTIVATION_FUNCTIONS_LIST"],
        config["LOSS_FUNCTIONS"],
        config["OPTIMISERS"],
        #rnn_type,
        #bidirectional,
        paths_dict["results"],
        device
        )

    trained_model = load_trained_model(
        model,
        config["DOWNSTREAM_MODELS"]["PATH"],
        device
        )

    predictions_df, overall_metrics, domain_specific_metrics = handle_inference(
        config["DOWNSTREAM_MODELS"]["TRAINING_PARAMETERS"]["BATCH_SIZE"],
        config["DOWNSTREAM_MODELS_LIST"],
        trained_model,
        dataloaders_dict,
        criterion,
        device,
        config["PREDICTED_FEATURES_LIST"],
        paths_dict["results"]
        )