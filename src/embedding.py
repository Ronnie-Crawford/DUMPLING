# Standard modules
import os
import itertools
import pickle
from pathlib import Path

# Third party modules
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, EsmModel, EsmForProteinFolding, EsmTokenizer

# Local modules
from helpers import concatenate_embeddings, normalise_embeddings, fit_principal_components, compute_dataset_hash
from config_loader import config

def load_embeddings(
    datasets_dict: dict,
    batch_size: int,
    model_selections: list,
    embedding_layers: list,
    embedding_types: list,
    device: str,
    package_folder: Path
) -> tuple[list, int]:
    
    embedding_combinations = list(itertools.product(model_selections, embedding_layers, embedding_types))
    merged_embeddings_df = pd.DataFrame()
    
    for dataset_name, dataset in datasets_dict["all"].items():
        
        embeddings_list = []
        dataset_hash = compute_dataset_hash(dataset)
        
        for model_selection, embedding_layer, embedding_type in embedding_combinations:
            
            embedding_df = pd.DataFrame()
            
            embedding_sorted_path = package_folder / f"embeddings/dataset[{dataset_name}]" / f"model[{model_selection}]" / f"layer[{embedding_layer}]" / f"embedding_type[{embedding_type}]"
            embedding_tensor_path = embedding_sorted_path / "embeddings_tensor.pt"
            metadata_path = embedding_sorted_path / "metadata.pkl"
            
            # Attempt to load embeddings and metadata
            try:
                
                # Load embeddings
                embeddings = torch.load(embedding_tensor_path, map_location = "cpu")

                # Load metadata
                with open(metadata_path, 'rb') as f:
                    
                    metadata = pickle.load(f)

                # Compare dataset hash
                if metadata.get('dataset_hash') != dataset_hash:
                    
                    print(f"Dataset hash mismatch for {dataset_name}. Regenerating embeddings.")
                    raise ValueError("Dataset hash mismatch")

                # If everything is fine, append embeddings to the list
                embeddings_list.append(embeddings)
                print(f"Loaded embeddings for dataset '{dataset_name}' from '{embedding_tensor_path}'")

            except (FileNotFoundError, ValueError, EOFError, pickle.UnpicklingError) as e:
                
                print(f"Could not load embeddings for dataset '{dataset_name}' due to {e}. Generating embeddings.")
                
                # Ensure the directory exists
                if not os.path.exists(embedding_sorted_path):
                    
                    os.makedirs(embedding_sorted_path)

                model, embedding_size, tokeniser = setup_model(model_selection, device)
                embeddings = fetch_embeddings(dataset, model, tokeniser, batch_size, device, embedding_type, embedding_layer, embedding_size)
                
                # Save embeddings and metadata
                torch.save(embeddings, embedding_tensor_path)
                metadata = {"dataset_hash": dataset_hash}
                
                with open(metadata_path, 'wb') as f:
                    
                    pickle.dump(metadata, f)
                
                # Append embeddings to the list
                embeddings_list.append(embeddings)
                print(f"Generated and saved embeddings for dataset '{dataset_name}' at '{embedding_tensor_path}'")

        if config["NORMALISE_EMBEDDINGS"]:

            embeddings_list = normalise_embeddings(embeddings_list)

        dataset.sequence_representations = concatenate_embeddings(embeddings_list)
        datasets_dict["all"][dataset_name] = dataset

    # Fetch arbitrary dataset to check shape, should all be the same so order does not matter
    first_dataset = next(iter(datasets_dict["all"].values()))
    embedding_dimensions = first_dataset.sequence_representations[0].dim()
    embedding_size = first_dataset.sequence_representations[0].shape[embedding_dimensions - 1]

    return datasets_dict, embedding_size
                
def setup_model(model_selection: list, device: str):
    
    match model_selection:
        
        case "AMPLIFY_120M":

            model = AutoModel.from_pretrained("chandar-lab/AMPLIFY_120M", trust_remote_code = True)
            tokeniser = AutoTokenizer.from_pretrained("chandar-lab/AMPLIFY_120M", trust_remote_code = True)

        case "AMPLIFY_120M_base":

            model = AutoModel.from_pretrained("chandar-lab/AMPLIFY_120M_base", trust_remote_code = True)
            tokeniser = AutoTokenizer.from_pretrained("chandar-lab/AMPLIFY_120M_base", trust_remote_code = True)

        case "AMPLIFY_350M":

            model = AutoModel.from_pretrained("chandar-lab/AMPLIFY_350M", trust_remote_code = True)
            tokeniser = AutoTokenizer.from_pretrained("chandar-lab/AMPLIFY_350M", trust_remote_code = True)

        case "AMPLIFY_350M_base":

            model = AutoModel.from_pretrained("chandar-lab/AMPLIFY_350M_base", trust_remote_code = True)
            tokeniser = AutoTokenizer.from_pretrained("chandar-lab/AMPLIFY_350M_base", trust_remote_code = True)

        case "ESMFold":
            
            model = EsmForProteinFolding.from_pretrained("./models/upstream_models/esmfold_3B_v1", output_hidden_states = True)
            tokeniser = EsmTokenizer.from_pretrained("./models/upstream_models/esmfold_3B_v1")
        
        case "ESM2_T6_8M_UR50D":
            
            model = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
            tokeniser = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
        
        case "ESM2_T12_35M_UR50D":
            
            model = EsmModel.from_pretrained("facebook/esm2_t12_35M_UR50D")
            tokeniser = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
        
        case "ESM2_T30_150M_UR50D":
            
            model = EsmModel.from_pretrained("facebook/esm2_t30_150M_UR50D")
            tokeniser = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")
        
        case "ESM2_T33_650M_UR50D":
        
            model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
            tokeniser = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        
        case "ESM2_T36_3B_UR50D":

            model = EsmModel.from_pretrained("facebook/esm2_t36_3B_UR50D")
            tokeniser = AutoTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D")
        
        case "ESM2_T48_15B_UR50D":
            
            model = EsmModel.from_pretrained("facebook/esm2_t48_15B_UR50D")
            tokeniser = AutoTokenizer.from_pretrained("facebook/esm2_t48_15B_UR50D")
        
    model.eval()
    model.to(device)
    
    embedding_size = model.config.hidden_size
    
    return model, embedding_size, tokeniser

def fetch_embeddings(
    dataset: Dataset,
    model,
    tokeniser,
    batch_size: int,
    device: str,
    embedding_type: str,
    embedding_layer: int,
    embedding_size: int
) -> torch.tensor:
    
    pooled_batch_embeddings = []
    full_embeddings = [None] * len(dataset)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = False)

    with torch.no_grad():

        for batch_idx, batch in enumerate(dataloader):

            sequences = batch["variant_aa_seq"]
            inputs = tokeniser(sequences, padding = True, truncation = True, return_tensors = "pt", max_length = 1024)
            inputs = inputs["input_ids"].to(device)
            model = model.to(device)
            output = model(inputs, output_hidden_states = True)
            batch_embeddings = output.hidden_states[embedding_layer]

            if embedding_type == "RAW":
                
                pooled_batch_embeddings = []
                
                for seq_idx, seq in enumerate(batch["variant_aa_seq"]):
                    
                    seq_len = len(seq)
                    seq_embeddings = batch_embeddings[seq_idx, 1:seq_len + 1, :].cpu()
                    pooled_batch_embeddings.append(seq_embeddings)

            elif embedding_type == "MEAN":

                pooled_batch_embeddings = batch_embeddings.mean(dim = 1).float().cpu()

            elif embedding_type == "MAX":

                pooled_batch_embeddings = batch_embeddings.max(dim = 1).values.cpu()

            elif embedding_type == "MIN":

                pooled_batch_embeddings = batch_embeddings.min(dim = 1).values.cpu()

            elif embedding_type == "STD":

                pooled_batch_embeddings = batch_embeddings.std(dim = 1).float().cpu()

            elif embedding_type == "PC1":

                pooled_batch_embeddings = fit_principal_components(batch_embeddings, 1, device).cpu()

            elif embedding_type == "PC2":

                pooled_batch_embeddings = fit_principal_components(batch_embeddings, 2, device).cpu()

            elif embedding_type == "PC3":

                pooled_batch_embeddings = fit_principal_components(batch_embeddings, 3, device).cpu()

            elif embedding_type == "LDSP":
                
                # Not yet implemented
                pooled_batch_embeddings = find_latent_distance_to_stable_point(batch_embeddings).cpu()

            else:

                raise Exception("Invalid embedding type.")

            for i in range(len(sequences)):

                full_embeddings[batch_idx * batch_size + i] = pooled_batch_embeddings[i]

            print(f"Fetched batch {batch_idx + 1} out of {len(dataloader)}")

    print("Fetching embeddings complete!")

    return full_embeddings