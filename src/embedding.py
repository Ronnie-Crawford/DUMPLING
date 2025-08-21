#Standard modules
import shutil
import gc
import itertools
import hashlib
import pickle
import os
from pathlib import Path
from collections import defaultdict
import re
from typing import cast
import copy

# Third-party modules
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, EsmModel, AutoModelForMaskedLM, EsmForProteinFolding, EsmTokenizer, EsmTokenizer, EsmForMaskedLM

# Local modules
from datasets import ProteinDataset

def handle_embeddings(
    unique_datasets_dict: dict,
    embedding_model_names: list,
    batch_size: int,
    special_tokens_in_context_flag: bool,
    special_tokens_in_pool_flag: bool,
    embedding_layers: list,
    embedding_types: list,
    dimensional_reduction_method: str,
    n_desired_dimensions: int,
    normalise_embeddings_flag: bool,
    wildtype_concat: bool,
    wildtype_delta: bool,
    base_path,
    device: torch.device,
    n_workers: int,
    amino_acids: str
    ):

    """
    The handler for embeddings, runs the upstream models and coordinates and post-processing on the extracted embeddings:
    concatination, dimensional reduction, etc.
    """

    unique_datasets_dict, embedding_size = load_embeddings(
        unique_datasets_dict,
        batch_size,
        special_tokens_in_context_flag,
        special_tokens_in_pool_flag,
        embedding_model_names,
        embedding_layers,
        embedding_types,
        base_path,
        device,
        n_workers,
        amino_acids
        )

    unique_datasets_dict = post_process_embeddings(
        unique_datasets_dict,
        normalise_embeddings_flag,
        wildtype_concat,
        wildtype_delta,
        dimensional_reduction_method,
        n_desired_dimensions
        )

    return unique_datasets_dict, embedding_size

def load_embeddings(
    unique_datasets_dict: dict,
    batch_size: int,
    special_tokens_in_context_flag: bool,
    special_tokens_in_pool_flag: bool,
    model_selections: list[str],
    embedding_layers: list[int],
    embedding_types: list[str],
    package_folder: Path,
    device: torch.device,
    n_workers: int,
    amino_acids: str
    ) -> tuple[dict, int]:

    """
    Loads embeddings for each dataset, embeddings are saved for each dataset, not dataset-label subset,
    to save time generating embeddings and save space storing them.
    The trade-off is that embeddings must be loaded for the entire dataset of each subset - before it is filtered.
    The dataset hash matching is far from perfect, if the sequences are in a different order for example it will not recognise the embeddings,
    but it takes a lot of computation to regenerate each time, so I try to minimise changing the mechanisms.
    Generally there will be only one upstream model, hidden layer and post-processing type selected,
    but it is possible to load more and the results will be concatinated together,
    however I find that the increase in dimensionality quickly loses any benefit that increasing the available information might have had.
    """

    # Make list of all combinations of upstream models, hidden layer to extract embedding from, post-processing to apply to embeddings
    embedding_combinations = list(itertools.product(model_selections, embedding_layers, embedding_types))
    # Create dict to store all emebddings, once per dataset
    unique_embeddings = {}

    for dataset_name, dataset in unique_datasets_dict.items():

        dataset_hash = compute_dataset_hash(dataset)
        embeddings_list = []

        for model_selection, embedding_layer, embedding_type in embedding_combinations:

            # Each embedding choice fills its own dataframe, before being merged later
            embedding_df = pd.DataFrame()
            embedding_sorted_path = package_folder / f"embeddings/dataset[{dataset_name}]" / f"model[{model_selection}]" / f"layer[{embedding_layer}]" / f"embedding_type[{embedding_type}]" / f"special_token_context[{special_tokens_in_context_flag}]" / f"special_token_pooling[{special_tokens_in_pool_flag}]"
            embedding_tensor_path = embedding_sorted_path / "embeddings_tensor.pt"
            metadata_path = embedding_sorted_path / "metadata.pkl"
            partial_embeddings_path = embedding_sorted_path / "partial_embeddings"

            # Attempt to load embeddings and metadata
            try:

                # Load embeddings on CPU for faster access
                embeddings = torch.load(embedding_tensor_path, map_location = "cpu")

                # If full embeddings can be loaded, we can remove any partial embeddings still around
                if os.path.exists(partial_embeddings_path):

                    shutil.rmtree(partial_embeddings_path)

                # Load metadata
                with open(metadata_path, "rb") as f:

                    metadata = pickle.load(f)

                # Compare dataset hash
                if metadata.get("dataset_hash") != dataset_hash:

                    raise ValueError("Dataset hash mismatch")

                # If everything is fine, append embeddings to the list
                embeddings_list.append(embeddings)

            except (FileNotFoundError, ValueError, EOFError, pickle.UnpicklingError) as e:

                print(f"Could not load embeddings for dataset '{dataset_name}', Generating embeddings.")

                # Ensure the directory exists
                if not os.path.exists(embedding_sorted_path):

                    os.makedirs(embedding_sorted_path)

                embeddings = fetch_embeddings(
                    dataset,
                    model_selection,
                    batch_size,
                    special_tokens_in_context_flag,
                    special_tokens_in_pool_flag,
                    device,
                    embedding_type,
                    embedding_layer,
                    n_workers,
                    partial_embeddings_path,
                    amino_acids
                )

                # Save embeddings and metadata
                torch.save(embeddings, embedding_tensor_path)
                metadata = {"dataset_hash": dataset_hash}

                # If full embeddings have been saved, we can remove any partial embeddings still around
                if os.path.exists(partial_embeddings_path):

                    shutil.rmtree(partial_embeddings_path)

                with open(metadata_path, 'wb') as f:

                    pickle.dump(metadata, f)

                # Append embeddings to the list
                embeddings_list.append(embeddings)

            print(f"Loaded embeddings for [{dataset_name}] - [{model_selection}] - [{embedding_layer}] - [{embedding_type}]")

        unique_embeddings[dataset_name] = embeddings_list

    for dataset_name, dataset in unique_datasets_dict.items():

        dataset.sequence_embeddings = unique_embeddings[dataset_name][0]

    del unique_embeddings
    gc.collect()

    # Fetch arbitrary dataset to check shape, should all be the same so order does not matter
    first_dataset = next(iter(unique_datasets_dict.values()))
    embedding_dimensions = first_dataset.sequence_embeddings[0].dim()
    embedding_size = first_dataset.sequence_embeddings[0].shape[embedding_dimensions - 1]

    return unique_datasets_dict, embedding_size

def compute_dataset_hash(dataset):

    """
    Hashing everything at once was causing slow down,
    so instead we incrementally hash each sequence in turn
    """

    md5 = hashlib.md5()

    for sequence in dataset.aa_seqs:

        md5.update(sequence.encode("utf-8"))

    return md5.hexdigest()

def setup_model(model_selection: str, device: torch.device):

    """
    This feels like a hacky way to fetch the models but I can't think of a more pythonic way that doesn't break.
    """
    context_length = None
    additive_attention_flag = False
    attention_bias_allignment_mask = False
    add_masked_structure_flag = False
    add_spoof_structural_input_ids_flag = False

    match model_selection:

        case "AMPLIFY_120M":

            tokeniser = AutoTokenizer.from_pretrained("chandar-lab/AMPLIFY_120M", trust_remote_code = True)
            tokeniser = process_tokeniser(tokeniser)
            model = AutoModel.from_pretrained("chandar-lab/AMPLIFY_120M", trust_remote_code = True)
            model = model.half()    # Xformers module which AMPLIFY uses requires half-precision
            additive_attention_flag = True
            attention_bias_allignment_mask = True
            context_length = 2048

        case "AMPLIFY_120M_base":

            tokeniser = AutoTokenizer.from_pretrained("chandar-lab/AMPLIFY_120M_base", trust_remote_code = True)
            tokeniser = process_tokeniser(tokeniser)
            model = AutoModel.from_pretrained("chandar-lab/AMPLIFY_120M_base", trust_remote_code = True)
            model = model.half()    # Xformers module which AMPLIFY uses requires half-precision
            additive_attention_flag = True
            attention_bias_allignment_mask = True
            context_length = 2048

        case "AMPLIFY_350M":

            tokeniser = AutoTokenizer.from_pretrained("chandar-lab/AMPLIFY_350M", trust_remote_code = True)
            tokeniser = process_tokeniser(tokeniser)
            model = AutoModel.from_pretrained("chandar-lab/AMPLIFY_350M", trust_remote_code = True)
            model = model.half()    # Xformers module which AMPLIFY uses requires half-precision
            additive_attention_flag = True
            attention_bias_allignment_mask = True
            context_length = 2048

        case "AMPLIFY_350M_base":

            tokeniser = AutoTokenizer.from_pretrained("chandar-lab/AMPLIFY_350M_base", trust_remote_code = True)
            tokeniser = process_tokeniser(tokeniser)
            model = AutoModel.from_pretrained("chandar-lab/AMPLIFY_350M_base", trust_remote_code = True)
            model = model.half()    # Xformers module which AMPLIFY uses requires half-precision
            additive_attention_flag = True
            attention_bias_allignment_mask = True
            context_length = 2048

        case "ESMFold":

            # Currently broken, needs python version incompatable with everything else
            raise NotImplementedError("Doesn't quite work yet.")
            tokeniser = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
            tokeniser = process_tokeniser(tokeniser)
            model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", output_hidden_states = True)

        case "ESM2_T6_8M_UR50D":

            tokeniser = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
            tokeniser = process_tokeniser(tokeniser)
            model = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")

        case "ESM2_T12_35M_UR50D":

            tokeniser = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
            tokeniser = process_tokeniser(tokeniser)
            model = EsmModel.from_pretrained("facebook/esm2_t12_35M_UR50D")

        case "ESM2_T30_150M_UR50D":

            tokeniser = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")
            tokeniser = process_tokeniser(tokeniser)
            model = EsmModel.from_pretrained("facebook/esm2_t30_150M_UR50D")

        case "ESM2_T33_650M_UR50D":

            tokeniser = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
            tokeniser = process_tokeniser(tokeniser)
            model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")

        case "ESM2_T36_3B_UR50D":

            tokeniser = AutoTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D")
            tokeniser = process_tokeniser(tokeniser)
            model = EsmModel.from_pretrained("facebook/esm2_t36_3B_UR50D")

        case "ESM2_T48_15B_UR50D":

            tokeniser = AutoTokenizer.from_pretrained("facebook/esm2_t48_15B_UR50D")
            tokeniser = process_tokeniser(tokeniser)
            model = EsmModel.from_pretrained("facebook/esm2_t48_15B_UR50D")

        case "PROGEN_2_SMALL":

            tokeniser = AutoTokenizer.from_pretrained("hugohrban/progen2-small", trust_remote_code = True)
            tokeniser = process_tokeniser(tokeniser)
            model = AutoModelForCausalLM.from_pretrained("hugohrban/progen2-small", trust_remote_code = True)

        case "PROGEN_2_MEDIUM":

            tokeniser = AutoTokenizer.from_pretrained("hugohrban/progen2-medium", trust_remote_code = True)
            tokeniser = process_tokeniser(tokeniser)
            model = AutoModelForCausalLM.from_pretrained("hugohrban/progen2-medium", trust_remote_code = True)

        case "PROGEN_2_LARGE":

            tokeniser = AutoTokenizer.from_pretrained("hugohrban/progen2-large", trust_remote_code = True)
            tokeniser = process_tokeniser(tokeniser)
            model = AutoModelForCausalLM.from_pretrained("hugohrban/progen2-large", trust_remote_code = True)

        case "PROSST_128":

            tokeniser = AutoTokenizer.from_pretrained("AI4Protein/ProSST-128", trust_remote_code = True)
            tokeniser = process_tokeniser(tokeniser)
            model = AutoModelForMaskedLM.from_pretrained("AI4Protein/ProSST-128", trust_remote_code = True)
            context_length = 1024
            add_spoof_structural_input_ids_flag = True

        case "PROSST_512":

            tokeniser = AutoTokenizer.from_pretrained("AI4Protein/ProSST-512", trust_remote_code = True)
            tokeniser = process_tokeniser(tokeniser)
            model = AutoModelForMaskedLM.from_pretrained("AI4Protein/ProSST-512", trust_remote_code = True)
            context_length = 1024
            add_spoof_structural_input_ids_flag = True

        case "PROSST_1024":

            tokeniser = AutoTokenizer.from_pretrained("AI4Protein/ProSST-1024", trust_remote_code = True)
            tokeniser = process_tokeniser(tokeniser)
            model = AutoModelForMaskedLM.from_pretrained("AI4Protein/ProSST-1024", trust_remote_code = True)
            context_length = 1024
            add_spoof_structural_input_ids_flag = True

        case "PROSST_2048":

            tokeniser = AutoTokenizer.from_pretrained("AI4Protein/ProSST-2048", trust_remote_code = True)
            tokeniser = process_tokeniser(tokeniser)
            model = AutoModelForMaskedLM.from_pretrained("AI4Protein/ProSST-2048", trust_remote_code = True)
            context_length = 1024
            add_spoof_structural_input_ids_flag = True

        case "PROSST_4096":

            tokeniser = AutoTokenizer.from_pretrained("AI4Protein/ProSST-4096", trust_remote_code = True)
            tokeniser = process_tokeniser(tokeniser)
            model = AutoModelForMaskedLM.from_pretrained("AI4Protein/ProSST-4096", trust_remote_code = True)
            context_length = 1024
            add_spoof_structural_input_ids_flag = True

        case "SAPROT_650M":

            tokeniser = EsmTokenizer.from_pretrained("westlake-repl/SaProt_650M_AF2")
            tokeniser = process_tokeniser(tokeniser)
            model = EsmForMaskedLM.from_pretrained("westlake-repl/SaProt_650M_AF2")
            add_masked_structure_flag = True

        case _:

            raise ValueError(f"Unknown model_selection: {model_selection}")

    model.eval()
    model.to(device)    # type: ignore

    if "hidden_size" in model.config:

        embedding_size = model.config.hidden_size

    elif "embed_dim" in model.config:

        embedding_size = model.config.embed_dim

    else:

        raise ValueError(f"Could not retrieve embedding size, check the names of variables in model config: {model.config.keys()}")

    return model, embedding_size, tokeniser, additive_attention_flag, attention_bias_allignment_mask, add_masked_structure_flag, add_spoof_structural_input_ids_flag, context_length

def process_tokeniser(tokeniser):

    """
    Tokenisers vary between models, some including BOS and EOS tokens, some having CLS tokens,
    some include special tokens automatically while others do not.
    Here we attempt to standardise these.
    """

    # Some tokenisers don't come with a pad token set, so we set one.
    if tokeniser.pad_token is None:

        tokeniser.add_special_tokens({"pad_token": "<|pad|>"})
        tokeniser.pad_token = "<|pad|>"

    tokeniser.bos_token = tokeniser.cls_token
    tokeniser.bos_token_id = tokeniser.cls_token_id

    return tokeniser

def fetch_embeddings(
    dataset: ProteinDataset,
    model_selection: str,
    batch_size: int,
    special_tokens_in_context_flag: bool,
    special_tokens_in_pool_flag: bool,
    device: torch.device,
    embedding_type: str,
    embedding_layer: int,
    n_workers: int,
    partial_embeddings_folder: Path,
    amino_acids: str
    ) -> torch.Tensor:

        # Make sure partial embeddings directory exists
        os.makedirs(partial_embeddings_folder, exist_ok = True)
        # Scan for already saved temp files
        completed_sequences = set()
        pattern = re.compile(r"checkpointed_embeddings_(\d+)-(\d+)\.pkl")

        for fname in os.listdir(partial_embeddings_folder):

            match = pattern.match(fname)

            if match:

                start_sequence_index, end_sequence_index = map(int, match.groups())
                completed_sequences.update(range(start_sequence_index, end_sequence_index + 1))

        last_completed_sequence_index = max(completed_sequences) if completed_sequences else -1 # -1 here is a placeholder when there are no compeleted indicies

        full_embeddings = [None] * len(dataset)
        dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = False, num_workers = n_workers, persistent_workers = True)
        model, embedding_size, tokeniser, additive_attention_flag, attention_bias_allignment_mask, add_masked_structure_flag, add_spoof_structural_input_ids_flag, context_length = setup_model(
            model_selection,
            device
            )

        with torch.no_grad():

            model = model.to(device)    # type: ignore
            model.eval()

            if context_length == None:

                context_length = get_context_length(model)

            batch_accumulation = []
            batch_indices = []

            for batch_index, batch in enumerate(dataloader):

                # Sequences are saved in groups of batches but batch size might change between incomplete runs,
                # so we check the last sequence in the batch, and otherwise recalculate the whole batch
                sequence_index = (batch_index + 1) * batch_size - 1

                if sequence_index in completed_sequences:

                    continue

                sequences = batch["aa_seq"]
                batch_embeddings, pool_mask = fetch_batch_embeddings(
                    sequences,
                    model,
                    tokeniser,
                    special_tokens_in_context_flag,
                    special_tokens_in_pool_flag,
                    additive_attention_flag,
                    attention_bias_allignment_mask,
                    add_masked_structure_flag,
                    add_spoof_structural_input_ids_flag,
                    device,
                    embedding_layer,
                    context_length,
                    amino_acids
                    )

                if embedding_type == "RAW":

                    # Need to update embedding types other than Mean.
                    raise NotImplementedError("Doesn't quite work yet.")
                    pooled_batch_embeddings =[]

                    for sequence_index, seq in enumerate(batch["aa_seq"]):

                        sequence_embeddings = batch_embeddings[sequence_index, 1:len(seq) + 1, :].detach().cpu()
                        pooled_batch_embeddings.append(sequence_embeddings)

                if embedding_type == "MEAN":

                    for sequence_embedding in batch_embeddings:

                        for position_embedding in sequence_embedding:

                            if torch.isinf(position_embedding).any().item():

                                print(position_embedding)

                    sum_embeddings = batch_embeddings.sum(dim = 1, dtype=torch.float32) # Convert back to float32 to avoid numeric overflow

                    if torch.isinf(sum_embeddings).any():

                        raise ValueError("Summed embeddings contain inf value, possibly due to numeric overflow.")

                    counts = pool_mask.sum(dim = 1).clamp(min = 1).unsqueeze(-1)
                    pooled_batch_embeddings = sum_embeddings / counts

                elif embedding_type == "MAX":

                    # Need to update embedding types other than Mean.
                    raise NotImplementedError("Doesn't quite work yet.")
                    pooled_batch_embeddings = batch_embeddings.max(dim = 1).values.detach().cpu()

                elif embedding_type == "MIN":

                    # Need to update embedding types other than Mean.
                    raise NotImplementedError("Doesn't quite work yet.")
                    pooled_batch_embeddings = batch_embeddings.min(dim = 1).values.detach().cpu()

                elif embedding_type == "STD":

                    # Need to update embedding types other than Mean.
                    raise NotImplementedError("Doesn't quite work yet.")
                    pooled_batch_embeddings = batch_embeddings.std(dim = 1).float().detach().cpu()

                elif embedding_type == "PC1":

                    # Need to update embedding types other than Mean.
                    raise NotImplementedError("Doesn't quite work yet.")
                    pooled_batch_embeddings = fit_principal_components(batch_embeddings, 1, device).detach().cpu()

                elif embedding_type == "PC2":

                    # Need to update embedding types other than Mean.
                    raise NotImplementedError("Doesn't quite work yet.")
                    pooled_batch_embeddings = fit_principal_components(batch_embeddings, 2, device).detach().cpu()

                elif embedding_type == "PC3":

                    # Need to update embedding types other than Mean.
                    raise NotImplementedError("Doesn't quite work yet.")
                    pooled_batch_embeddings = fit_principal_components(batch_embeddings, 3, device).detach().cpu()

                else:

                    raise Exception("Invalid embedding type.")

                # Store in memory until time to save
                batch_accumulation.extend(pooled_batch_embeddings.detach().cpu())
                batch_indices.append(batch_index)

                low_sequence_index_to_save = min(batch_indices) * batch_size
                high_sequence_index_to_save = (max(batch_indices) + 1) * batch_size - 1

                if last_completed_sequence_index >= low_sequence_index_to_save:

                    offset = last_completed_sequence_index - low_sequence_index_to_save + 1  # +1 because last_completed included
                    low_sequence_index_to_save = last_completed_sequence_index + 1
                    batch_accumulation = batch_accumulation[offset:]

                # Save every X batches
                if (batch_index + 1) % 100 == 0 or (batch_index + 1) == len(dataloader):

                    temp_fname = os.path.join(partial_embeddings_folder, f"checkpointed_embeddings_{low_sequence_index_to_save}-{high_sequence_index_to_save}.pkl")
                    batch_accumulation = torch.stack(batch_accumulation)

                    with open(temp_fname, "wb") as f:

                        pickle.dump(batch_accumulation, f)

                    batch_accumulation = []
                    batch_indices = []

                    # A desperate attempt to claw back any VRAM and RAM
                    del batch, sequences, batch_embeddings, pooled_batch_embeddings
                    gc.collect()
                    torch.cuda.empty_cache()

                print(f"Fetched embedding {batch_index + 1} out of {len(dataloader)}")

        # Once all partial embeddings are saved, we need to load them all into one object
        pattern = re.compile(r"checkpointed_embeddings_(\d+)-(\d+)\.pkl")

        for fname in sorted(os.listdir(partial_embeddings_folder)):

            match = pattern.match(fname)

            if match:

                start_sequence_index, end_sequence_index = map(int, match.groups())

                with open(os.path.join(partial_embeddings_folder, fname), "rb") as f:

                    partial_embeddings_list = pickle.load(f)

                for sequence_offset, pooled_embedding in enumerate(partial_embeddings_list):

                    global_sequence_index = start_sequence_index + sequence_offset
                    full_embeddings[global_sequence_index] = pooled_embedding

        assert all(isinstance(x, torch.Tensor) for x in full_embeddings), "Some entries in full_embeddings are not torch tensors."
        full_embeddings = cast(list[torch.Tensor], full_embeddings)
        full_embeddings = torch.stack(full_embeddings)

        return full_embeddings

def post_process_embeddings(
    unique_datasets_dict: dict,
    normalise_embeddings_flag: bool,
    wildtype_concat: bool,
    wildtype_delta: bool,
    dimensional_reduction_method: str,
    n_desired_dimensions: int
    ):

    for dataset_name, dataset in unique_datasets_dict.items():

        if normalise_embeddings_flag:

            embeddings = normalise_embeddings(dataset)

        if wildtype_concat:

            dataset = concat_wildtype_embeddings(dataset)

        elif wildtype_delta:

            dataset = find_wildtype_delta(dataset)

        unique_datasets_dict[dataset_name] = dataset

    return unique_datasets_dict

def get_context_length(model):

    # Fetch the context length, if we can find it
    if hasattr(model.config, "max_position_embeddings"):

        context_length = model.config.max_position_embeddings

    elif hasattr(model.config, "n_ctx"):

        context_length = model.config.n_ctx

    elif hasattr(model.config, "n_positions"):

        context_length = model.config.n_positions

    elif hasattr(model.config, "model_max_length"):

        context_length = model.config.model_max_length

    else:

        raise ValueError(f"Context length attribute not found in model config for {type(model).__name__}.")

    return context_length

def fetch_batch_embeddings(
    sequences,
    model,
    tokeniser,
    special_tokens_in_context_flag: bool,
    special_tokens_in_pool_flag: bool,
    additive_attention_flag: bool,
    attention_bias_allignment_flag: bool,
    add_masked_structure_flag: bool,
    add_spoof_structural_input_ids_flag: bool,
    device: torch.device,
    embedding_layer: int,
    context_length: int,
    amino_acids: str
    ):

    is_split_into_words_flag = False

    if add_masked_structure_flag:

        # Some models require a structural context for each token, here we add "#" meaning an unknown structural context for each residue
        # Since tokens are therefore more than one character, we also then split by word rather than character
        sequences = [[residue + "#" for residue in sequence] for sequence in sequences]
        is_split_into_words_flag = True

    if not attention_bias_allignment_flag:

        inputs = tokeniser(sequences, is_split_into_words = is_split_into_words_flag, padding = True, truncation = False, return_tensors = "pt")

    else:

        # If attention_bias_allignment_mask, requires padding batches to be divisible by 8
        max_sequence_length_in_batch = max([len(sequence) for sequence in sequences])
        max_length_plus_special_tokens = max_sequence_length_in_batch + 2
        minimum_divisibile_context_length = max_length_plus_special_tokens + 8 if max_length_plus_special_tokens % 8 == 0 else max_length_plus_special_tokens + (8 - max_length_plus_special_tokens % 8)
        # There must always be padding, so even if exactly diviisble by 8, we add 8.
        inputs = tokeniser(sequences, is_split_into_words = is_split_into_words_flag, padding = "max_length", truncation = False, return_tensors = "pt", max_length = minimum_divisibile_context_length)

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    modified_attention_mask, special_token_mask = modify_special_token_attention(
        input_ids,
        attention_mask,
        tokeniser,
        amino_acids,
        special_tokens_in_context_flag,
        add_masked_structure_flag,
        device
        )

    # Create pool mask of which positions should be included in pooling
    # We make sure to use the unmodified attention mask, to keep the logic for the 2 flags separate
    if special_tokens_in_pool_flag:

        pool_mask = attention_mask.bool()

    else:

        # Pool mask only true where pool mask True and special mask false
        pool_mask = attention_mask.bool() & (~special_token_mask.bool())

    # Check if we need to use sliding window method, or can use faster method
    if input_ids.shape[1] < context_length:

        if additive_attention_flag:

            modified_attention_mask = torch.where(
                modified_attention_mask.bool(),
                torch.tensor(0.0, dtype = torch.float16, device = device),
                torch.tensor(float("-inf"), dtype = torch.float16, device = device)
            )

        # If the model requires strutural input ids, we spoof them (all padding tokens, adding no information but not disrupting)
        if add_spoof_structural_input_ids_flag:

            # Pad token id for ProSST is 2
            pad_token = 2
            spoof_structure_input_ids = torch.full((1, input_ids.size(1)), pad_token, dtype = torch.long).to(device)
            output = model(input_ids, attention_mask = modified_attention_mask, ss_input_ids = spoof_structure_input_ids, output_hidden_states = True)

        else:

            output = model(input_ids, attention_mask = modified_attention_mask, output_hidden_states = True)

        batch_embeddings = output.hidden_states[embedding_layer]
        batch_embeddings = batch_embeddings * pool_mask.unsqueeze(-1)

        return batch_embeddings.detach().cpu(), pool_mask.detach().cpu()

    else:

        # Set up windows
        stride = context_length // 2    # This seems the most popular method in the literature to save compute over doing stride length == 1
        windows = []
        window_start = 0

        if additive_attention_flag:

            window_length = context_length - 8

        else:

            window_length = context_length

        while window_start + window_length < input_ids.shape[1]:

            window_end = window_start + window_length
            windows.append((window_start, window_end))
            window_start += stride

        last_window_start = max(0, input_ids.shape[1] - window_length)
        last_window = (last_window_start, input_ids.shape[1])

        if not windows or windows[-1][0] != last_window[0]:

            windows.append(last_window)

        position_to_embeddings = [defaultdict(list) for _position in range(input_ids.shape[0])]  # Stores arbitray number of embeddings for each position

        # Iterate through windows
        for window_start, window_end in windows:

            print(f"Fetching embeddings for window {window_start} - {window_end}.")
            window_input_ids = input_ids[:, window_start:window_end]
            window_attention_mask = modified_attention_mask[:, window_start:window_end]
            window_pool_mask = pool_mask[:, window_start:window_end]

            if additive_attention_flag:

                prepped_window_attention_mask = torch.where(
                    window_attention_mask.bool(),
                    torch.tensor(0.0, dtype = torch.float16, device = device),
                    torch.tensor(float("-inf"), dtype = torch.float16, device = device)
                )

                if window_end < input_ids.shape[1]:

                   window_input_ids = append_eight_pad_tokens(tokeniser, window_input_ids)
                   prepped_window_attention_mask = append_eight_masked_out_elements(prepped_window_attention_mask)
                   window_pool_mask = append_eight_pool_mask(window_pool_mask)

                else:

                    window_input_ids = prepend_eight_pad_tokens(tokeniser, window_input_ids)
                    prepped_window_attention_mask = prepend_eight_masked_out_elements(prepped_window_attention_mask)
                    window_pool_mask = prepend_eight_pool_mask(window_pool_mask)

            else:

                prepped_window_attention_mask = window_attention_mask

            # If the model requires strutural input ids, we spoof them (all padding tokens, adding no information but not disrupting)
            if add_spoof_structural_input_ids_flag:

                # Pad token id for ProSST is 2
                pad_token = 2
                spoof_structure_input_ids = torch.full((1, window_input_ids.size(1)), pad_token, dtype = torch.long).to(device)
                output = model(window_input_ids, attention_mask = modified_attention_mask, ss_input_ids = spoof_structure_input_ids, output_hidden_states = True)

            else:

                output = model(window_input_ids, attention_mask = prepped_window_attention_mask, output_hidden_states = True)

            batch_window_embeddings = output.hidden_states[embedding_layer]
            batch_window_embeddings = batch_window_embeddings * window_pool_mask.unsqueeze(-1)

            # Set embeddings of sequences that have no valid tokens in this window to zero
            valid_seq_mask = window_attention_mask.sum(dim = 1) > 0

            for batch_index, is_valid in enumerate(valid_seq_mask):

                if not is_valid:

                    batch_window_embeddings[batch_index] = 0  # zero all embeddings for this seq in this window

            # Line up the results from each window with the correct positions
            for batch_index in range(input_ids.shape[0]):

                for window_position in range(window_end - window_start):

                    sequence_position = window_start + window_position
                    position_to_embeddings[batch_index][sequence_position].append(batch_window_embeddings[batch_index, window_position].detach().cpu())

            # Deciding whether to keep this, it really slows things down, but sometimes necessary for very large models + long sequences
            del window_input_ids, window_attention_mask, output, batch_window_embeddings, window_pool_mask, valid_seq_mask
            gc.collect()
            torch.cuda.empty_cache()

        embedding_dimension = list(position_to_embeddings[0].values())[0][0].shape[0]
        batch_embeddings = []

        for batch_index in range(input_ids.shape[0]):

            position_embeddings = torch.zeros((input_ids.shape[1], embedding_dimension))

            for position in range(input_ids.shape[1]):

                embeddings_list = position_to_embeddings[batch_index].get(position, [])

                if embeddings_list != []:

                    mean_embedding_for_position = torch.stack(embeddings_list, dim = 0).mean(dim = 0)
                    position_embeddings[position, :] = mean_embedding_for_position

            batch_embeddings.append(position_embeddings)

        batch_embeddings = torch.stack(batch_embeddings)

        return batch_embeddings.detach().cpu(), pool_mask.detach().cpu()

def modify_special_token_attention(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokeniser,
    amino_acids: str,
    special_tokens_in_context_flag: bool,
    add_masked_structure_flag: bool,
    device: torch.device
    ):

    # Build set of valid amino acid and pad token IDs
    vocab = tokeniser.get_vocab()

    if add_masked_structure_flag:

        amino_acids = [res + "#" for res in amino_acids]

    aa_token_ids = set(vocab[res] for res in amino_acids if res in vocab)
    aa_token_ids.add(vocab[tokeniser.pad_token])    # Should be set from previously padding
    special_token_mask = torch.tensor(
        [[id not in aa_token_ids for id in ids] for ids in input_ids.tolist()],
        dtype = torch.bool,
        device = device
        )

    if special_tokens_in_context_flag:

        # Inlcude everything as-is
        return attention_mask, special_token_mask

    else:

        # Zero attention for special tokens
        modified_attention_mask = attention_mask.clone()
        modified_attention_mask[special_token_mask] = 0

        # Return unmodified input_ids and special_token_masks
        return modified_attention_mask, special_token_mask

def append_eight_pad_tokens(tokeniser, tensor):

    pad = torch.full((tensor.shape[0], 8), tokeniser.pad_token_id, device = tensor.device, dtype = tensor.dtype)

    return torch.cat([tensor, pad], dim = 1)

def prepend_eight_pad_tokens(tokeniser, tensor):

    pad = torch.full((tensor.shape[0], 8), tokeniser.pad_token_id, device = tensor.device, dtype = tensor.dtype)

    return torch.cat([pad, tensor], dim = 1)

def append_eight_masked_out_elements(mask_tensor):

    pad = torch.full((mask_tensor.shape[0], 8), float("-inf"), device = mask_tensor.device, dtype = mask_tensor.dtype)

    return torch.cat([mask_tensor, pad], dim = 1)

def prepend_eight_masked_out_elements(mask_tensor,):

    pad = torch.full((mask_tensor.shape[0], 8), float("-inf"), device = mask_tensor.device, dtype = mask_tensor.dtype)
    return torch.cat([pad, mask_tensor], dim = 1)

def append_eight_pool_mask(pool_mask):

    pad = torch.zeros((pool_mask.shape[0], 8), dtype = torch.bool, device = pool_mask.device)

    return torch.cat([pool_mask, pad], dim = 1)

def prepend_eight_pool_mask(pool_mask):

    pad = torch.zeros((pool_mask.shape[0], 8), dtype = torch.bool, device = pool_mask.device)

    return torch.cat([pad, pool_mask], dim = 1)

def concatenate_embeddings(embeddings_list: list) -> list:

    concatenated_embeddings = []

    for variant_embeddings in list(zip(*embeddings_list)):

        concatenated_embeddings.append(torch.concat(variant_embeddings))

    return concatenated_embeddings

def normalise_embeddings(embeddings_list):

    normalised_embeddings_list = []

    for embeddings in embeddings_list:

        normalised_embeddings = []

        for embedding in embeddings:

            normalised_embedding = normalise_tensor(embedding)
            normalised_embeddings.append(normalised_embedding)

        normalised_embeddings_list.append(normalised_embeddings)

    return normalised_embeddings_list

def normalise_tensor(tensor):

    vector = tensor.cpu().numpy()

    try:

        normalised_vector = (vector - np.mean(vector)) / np.std(vector)

    except Exception as e:

        print(f"Could not normalise vector: {e}")
        normalised_vector = vector

    return torch.tensor(normalised_vector)

def fit_principal_components(embeddings, component_index: int, device: torch.device):

    assert embeddings.dim() == 3, "Input embeddings must be a 3D tensor such that each slice is a point."

    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    with torch.no_grad():

        batch_size, n_sequences, n_residues = embeddings.shape
        flattened_embeddings = embeddings.reshape(-1, n_residues)
        centered_points = flattened_embeddings - flattened_embeddings.mean(dim = 1, keepdim = True)

        # Compute the covariance matrices for each sample
        centered_points = centered_points.reshape(batch_size, n_sequences, n_residues)
        covariance_matrices = torch.einsum("ijk,ijl->ikl", centered_points, centered_points) / (n_sequences - 1)

        # Compute eigenvalues and eigenvectors for each covariance matrix
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrices)

        # Sort eigenvalues and eigenvectors
        sorted_indices = torch.argsort(eigenvalues, dim=-1, descending=True)
        sorted_eigenvectors = eigenvectors.gather(2, sorted_indices.unsqueeze(-1).expand(-1, -1, eigenvectors.size(-1)))

    # Select principal component
    if component_index < 1 or component_index > sorted_eigenvectors.shape[1]:

        raise ValueError("Component_index must be between 1 and the number of components.")

    principal_components = sorted_eigenvectors[:, :, component_index - 1]

    # Try desperately to reduce memory usage and salvage back anything from the belly of the beast
    del embeddings, component_index, batch_size, n_sequences, n_residues, flattened_embeddings, centered_points, covariance_matrices, eigenvalues, eigenvectors, sorted_indices, sorted_eigenvectors
    gc.collect()
    torch.cuda.empty_cache()

    return principal_components

def concat_wildtype_embeddings(dataset):

    domain_to_wt = {}
    new_embeddings = []

    # First loop collects WT embeddings
    for index, domain in enumerate(dataset.domain_names):

        if dataset.wt_flags[index]:

            sequence_embedding = dataset.sequence_embeddings[index]
            domain_to_wt[domain] = sequence_embedding

    # Second loop matches them with mutants
    for index, domain in enumerate(dataset.domain_names):

        sequence_embedding = dataset.sequence_embeddings[index]
        wildtype_embedding = domain_to_wt.get(domain)
        assert isinstance(wildtype_embedding, torch.Tensor), f"Expected wildtype embedding to be Tensor, got {type(wildtype_embedding)}"
        concatinated_embedding = torch.cat((sequence_embedding, wildtype_embedding), dim = 0)
        new_embeddings.append(concatinated_embedding)

    dataset.sequence_embeddings = new_embeddings

    return dataset

def find_wildtype_delta(dataset):

    domain_to_wt = {}
    new_embeddings = []

    for index, domain in enumerate(dataset.domain_names):

        if dataset.wt_flags[index]:

            domain_to_wt[domain] = dataset.sequence_embeddings[index]

    for index, domain in enumerate(dataset.domain_names):

        delta_embedding = dataset.sequence_embeddings[index] - domain_to_wt.get(domain)
        new_embeddings.append(delta_embedding)

    dataset.sequence_embeddings = new_embeddings

    return dataset
