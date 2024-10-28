# Standard modules
import os
import itertools
import pickle

# Third party modules
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import esm
from transformers import AutoModel, AutoTokenizer

# Local modules
from config_loader import config
from preprocessing import pad_variable_length_sequences
from datasets import ProteinDataset
from helpers import manage_memory, concatenate_embeddings, normalise_embeddings, fit_principal_components

def load_embeddings(
    embedding_flag: str,
    datasets: list,
    batch_size: int,
    device: str,
    datasets_in_use: list,
    model_selections: list,
    embedding_layers: list,
    embedding_types: list
) -> tuple[list, int]:

    embedding_size = 0
    embeddings = []

    embedding_combinations = itertools.product(model_selections, embedding_layers, embedding_types)
    merged_embeddings_df = pd.DataFrame()

    for dataset_name, dataset in zip(datasets_in_use, datasets):

        embeddings_list = []

        for model_selection, embedding_layer, embedding_type in embedding_combinations:

            embedding_df = pd.DataFrame()
            path = f"./embeddings/dataset[{dataset_name}]/model[{model_selection}]/layer[{embedding_layer}]/embedding_type[{embedding_type}]"

            if not os.path.exists(f"{path}/embeddings_tensor.pt"):

                if "AMPLIFY" in model_selection:

                    model, embedding_size, tokeniser = setup_amplify(device, model_selection)
                    embeddings = fetch_amplify_embeddings_batched(dataset, model, tokeniser, batch_size, device, embedding_type, embedding_layer, embedding_size)
                    if not os.path.exists(path): os.makedirs(path)
                    torch.save(embeddings, f"{path}/embeddings_tensor.pt")

                elif "ESM" in model_selection:

                    model, alphabet, batch_converter, embedding_size, n_layers = setup_esm(device, model_selection)
                    embeddings = fetch_esm_embeddings_batched(dataset, model, alphabet, batch_converter, n_layers, device, batch_size, embedding_layer, embedding_type)
                    if not os.path.exists(path): os.makedirs(path)
                    torch.save(embeddings, f"{path}/embeddings_tensor.pt")

            else:

                embeddings = torch.load(f"{path}/embeddings_tensor.pt", weights_only = False)

            embeddings_list.append(embeddings)

        if config["NORMALISE_EMBEDDINGS"]:

            embeddings_list = normalise_embeddings(embeddings_list)

        dataset.sequence_representations = concatenate_embeddings(embeddings_list)

    embedding_size = len(datasets[0].__dict__["sequence_representations"][0])

    return datasets, embedding_size

def setup_amplify(device: str, model_selection: list):

    print("Setting up amplify")

    model, tokeniser = None, None

    match model_selection:

        case "AMPLIFY_120M":

            model = AutoModel.from_pretrained("chandar-lab/AMPLIFY_120M", trust_remote_code=True)
            tokeniser = AutoTokenizer.from_pretrained("chandar-lab/AMPLIFY_120M", trust_remote_code=True)

        case "AMPLIFY_120M_base":

            model = AutoModel.from_pretrained("chandar-lab/AMPLIFY_120M_base", trust_remote_code=True)
            tokeniser = AutoTokenizer.from_pretrained("chandar-lab/AMPLIFY_120M_base", trust_remote_code=True)

        case "AMPLIFY_350M":

            model = AutoModel.from_pretrained("chandar-lab/AMPLIFY_350M", trust_remote_code=True)
            tokeniser = AutoTokenizer.from_pretrained("chandar-lab/AMPLIFY_350M", trust_remote_code=True)

        case "AMPLIFY_350M_base":

            model = AutoModel.from_pretrained("chandar-lab/AMPLIFY_350M_base", trust_remote_code=True)
            tokeniser = AutoTokenizer.from_pretrained("chandar-lab/AMPLIFY_350M_base", trust_remote_code=True)

    model.eval()
    model.to(device)
    embedding_size = 960

    return model, embedding_size, tokeniser

def setup_esm(device: str, model_selection: list):

    model, alphabet = None, None

    match model_selection:

        case "ESM1_T6_43M_UR50S":

            model, alphabet = esm.pretrained.esm1_t6_43M_UR50S()

        case "ESM1_T12_85M_UR50S":

            model, alphabet = esm.pretrained.esm1_t12_85M_UR50S()

        case "ESM1_T34_670M_UR100":

            model, alphabet = esm.pretrained.esm1_t34_670M_UR100()

        case "ESM1_T34_670M_UR50D":

            model, alphabet = esm.pretrained.esm1_t34_670M_UR50D()

        case "ESM1_T34_670M_UR50S":

            model, alphabet = esm.pretrained.esm1_t34_670M_UR50S()

        case "ESM2_T6_8M_UR50D":

            model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()

        case "ESM2_T12_35M_UR50D":

            model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()

        case "ESM2_T30_150M_UR50D":

            model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()

        case "ESM2_T33_650M_UR50D":

            model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

        case "ESM2_T36_3B_UR50D":

            model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()

        case "ESM2_T48_15B_UR50D":

            model, alphabet = esm.pretrained.esm2_t48_15B_UR50D()

    batch_converter = alphabet.get_batch_converter()
    model = model.to(device)
    model.eval()
    embedding_size = model.embed_dim
    n_layers = len(model.layers)

    return model, alphabet, batch_converter, embedding_size, n_layers

def fetch_amplify_embeddings_batched(
    dataset: Dataset,
    model,
    tokeniser,
    batch_size: int,
    device: str,
    embedding_type: str,
    embedding_layer: int,
    embedding_size: int
) -> torch.tensor:

    print("Fetching amplify embeddings in batches")

    pooled_batch_embeddings = []
    full_embeddings = [None] * len(dataset)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = False)

    with torch.no_grad():

        for batch_idx, batch in enumerate(dataloader):

            sequences = batch["variant_aa_seq"]
            inputs = tokeniser(sequences, padding = True, truncation = True, return_tensors="pt", max_length=1024)
            inputs = inputs["input_ids"].to(device)
            output = model(inputs, output_hidden_states=True)
            batch_embeddings = output.hidden_states[embedding_layer]

            if embedding_type == "MEAN":

                pooled_batch_embeddings = batch_embeddings.mean(dim = 1).float().to(device)

            elif embedding_type == "MAX":

                pooled_batch_embeddings = batch_embeddings.max(dim = 1).values

            elif embedding_type == "MIN":

                pooled_batch_embeddings = batch_embeddings.min(dim = 1).values

            elif embedding_type == "STD":

                pooled_batch_embeddings = batch_embeddings.std(dim = 1).float().to(device)

            elif embedding_type == "PC1":

                pooled_batch_embeddings = fit_principal_components(batch_embeddings, 1, device)

            elif embedding_type == "PC2":

                pooled_batch_embeddings = fit_principal_components(batch_embeddings, 2, device)

            elif embedding_type == "PC3":

                pooled_batch_embeddings = fit_principal_components(batch_embeddings, 3, device)

            else:

                raise Exception("Invalid embedding type.")

            for i in range(len(sequences)):

                full_embeddings[batch_idx * batch_size + i] = pooled_batch_embeddings[i]

            print(f"Fetched batch {batch_idx + 1} out of {len(dataloader)}")
            manage_memory()

    print("Fetching embeddings complete!")

    return full_embeddings

def fetch_esm_embeddings_batched(
    dataset: Dataset,
    model,
    alphabet,
    batch_converter,
    n_layers: int,
    device: str,
    batch_size: int,
    embedding_layer: int,
    embedding_type: str
) -> torch.tensor:

    """
    Fetch ESM representations in batches, usually the most efficient method.
    Recommended for all but the smallest datasets.
    """

    pooled_batch_embeddings = []
    full_embeddings = [None] * len(dataset)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = False)
    embedding_layer = n_layers + 1 + embedding_layer

    with torch.no_grad():

        for batch_idx, batch in enumerate(dataloader):

            batch_tuples = list(zip(batch["domain_name"], batch["variant_aa_seq"]))
            batch_labels, batch_strs, batch_tokens = batch_converter(batch_tuples)
            batch_tokens = batch_tokens.to(device)
            output = model(batch_tokens, repr_layers = [embedding_layer], return_contacts = False)
            batch_embeddings = output["representations"][embedding_layer]

            if embedding_type == "MEAN":

                pooled_batch_embeddings = batch_embeddings.mean(dim = 1).float().to(device)

            elif embedding_type == "MAX":

                pooled_batch_embeddings = batch_embeddings.max(dim = 1).values

            elif embedding_type == "MIN":

                pooled_batch_embeddings = batch_embeddings.min(dim = 1).values

            elif embedding_type == "STD":

                pooled_batch_embeddings = batch_embeddings.std(dim = 1).float().to(device)

            elif embedding_type == "PC1":

                pooled_batch_embeddings = fit_principal_components(batch_embeddings, 1, device)

            elif embedding_type == "PC2":

                pooled_batch_embeddings = fit_principal_components(batch_embeddings, 2, device)

            elif embedding_type == "PC3":

                pooled_batch_embeddings = fit_principal_components(batch_embeddings, 3, device)

            else:

                raise Exception("Invalid embedding type.")

            for i in range(len(batch["variant_aa_seq"])):

                full_embeddings[batch_idx * batch_size + i] = pooled_batch_embeddings[i]

            print(f"Fetched ESM representations for batch {batch_idx + 1} of {len(dataloader)}")
            manage_memory()

    print(f"Completed fetching ESM representations for all {len(dataset)} items")

    return full_embeddings
