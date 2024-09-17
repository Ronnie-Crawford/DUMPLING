# Third party modules
import torch
from torch.utils.data import DataLoader
import esm

# Local modules
from config_loader import config
from datasets import ProteinDataset

def setup_esm(device, model_selections):

    #model, alphabet = esm.pretrained.esm1_t6_43M_UR50S()
    #model, alphabet = esm.pretrained.esm1_t12_85M_UR50S()
    #model, alphabet = esm.pretrained.esm1_t34_670M_UR100()
    #model, alphabet = esm.pretrained.esm1_t34_670M_UR50D()
    #model, alphabet = esm.pretrained.esm1_t34_670M_UR50S()

    if "ESM2_T6_8M_UR50D" in model_selections:

        model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()

    elif "ESM2_T12_35M_UR50D" in model_selections:

        model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()

    elif "ESM2_T30_150M_UR50D" in model_selections:

        model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
    #model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    #model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    #model, alphabet = esm.pretrained.esm2_t48_15B_UR50D()
    #model, alphabet = esm.pretrained.esm1v_t33_650M_UR90S_5() # Does not work with current architecture
    batch_converter = alphabet.get_batch_converter()
    model = model.to(device)
    model.eval()
    embedding_size = model.embed_dim
    n_layers = len(model.layers)

    return model, alphabet, batch_converter, embedding_size, n_layers

def fetch_esm_embeddings_item_by_item(dataset, model, alphabet: list, batch_converter, representation_layer: int, device: str = "cpu", ) -> None:

    """
    The simpliest way to fetch ESM representations, but also the slowest,
    applies to the entire dataset one domain at a time.
    Only for very small datasets.
    """

    for idx in range(len(dataset)):

        item = dataset[idx]
        domain_name = item["domain_name"]
        sequence = item["variant_aa_seq"]

        batch_tuples = [(domain_name, sequence)]
        batch_labels, batch_strs, batch_tokens = batch_converter(batch_tuples)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[representation_layer], return_contacts=False)
        token_representations = results["representations"][representation_layer]

        dataset.sequence_representations[idx] = token_representations[0, 1 : batch_lens[0] - 1].mean(0).float().to(device)

        if device.type == 'cuda' or device.type == 'mps': torch.cuda.empty_cache()

        if (idx + 1) % 1 == 0: print(f"Fetched ESM representations for batch {idx + 1} of {len(dataset)}")



def fetch_esm_embeddings_batched(dataset, model, alphabet, batch_converter, representation_layer: int, device: str = "cpu", batch_size: int = 32) -> None:

    """
    Fetch ESM representations in batches, usually the most efficient method.
    Recommended for all but the smallest datasets.
    """

    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_batches = len(dataloader)

    for batch_idx, batch in enumerate(dataloader):
        domain_names = batch["domain_name"]
        sequences = batch["variant_aa_seq"]

        batch_tuples = list(zip(domain_names, sequences))
        batch_labels, batch_strs, batch_tokens = batch_converter(batch_tuples)
        batch_tokens = batch_tokens.to(device)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[representation_layer], return_contacts=False)
        token_representations = results["representations"][representation_layer]

        for i, idx in enumerate(range(batch_idx * batch_size, min((batch_idx + 1) * batch_size, len(dataset)))):
            dataset.sequence_representations[idx] = token_representations[i, 1 : batch_lens[i] - 1].mean(0).float().to(device)

        if device == 'cuda' or device == 'mps':
            torch.cuda.empty_cache()

        print(f"Fetched ESM representations for batch {batch_idx + 1} of {total_batches}")

    print(f"Completed fetching ESM representations for all {len(dataset)} items")
