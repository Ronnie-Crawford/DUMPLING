# Standard modules
import datetime
import gc
from typing import cast

# Third-party modules
import numpy as np
import pandas as pd
import torch

def handle_inference(
    dataloaders_dict: dict,
    downstream_models,
    output_features,
    trained_model,
    criterion_dict,
    batch_size: int,
    device,
    results_path,
    ) -> pd.DataFrame:

    match downstream_models[0]:

        case "FFNN":

            test_loss, predictions_df = run_inference_on_ffnn(trained_model, dataloaders_dict["TEST"], criterion_dict, device, output_features, results_path, batch_size)

        case _:

            raise ValueError(f"Unknown model_selection: {downstream_models[0]}")

    predictions_df = save_results(predictions_df, results_path)

    return predictions_df

def run_inference_on_ffnn(
    model,
    test_loader,
    criterion_dict,
    device: str,
    output_features: list,
    results_path,
    batch_size: int
):

    model.eval()
    total_loss = 0.0

    predictions_dict = {feature: [] for feature in output_features}
    truths_dict = {feature: [] for feature in output_features}

    domains = []
    subsets = []
    sequences = []

    with torch.no_grad():

        for batch in test_loader:

            inputs = batch["sequence_embedding"].float().to(device)

            outputs = model(inputs)

            batch_losses = []

            for idx, feature in enumerate(output_features):

                truths = batch[f"{feature}_value"].float().to(device)
                masks = batch[f"{feature}_mask"].bool().to(device)

                preds = outputs[:, idx]

                # Mask and compute loss
                valid_preds = preds[masks]
                valid_truths = truths[masks]

                if valid_preds.nelement() > 0:

                    feature_loss = criterion_dict[feature](valid_preds, valid_truths)
                    batch_losses.append(feature_loss.item())

                else:

                    batch_losses.append(0.0)

                # Store predictions and truths (masked)
                cpu_preds = preds.cpu().numpy()
                cpu_truths = truths.cpu().numpy()
                cpu_masks = masks.cpu().numpy()

                #predictions_dict[feature].append(np.where(cpu_masks, cpu_preds, np.nan))
                predictions_dict[feature].append(cpu_preds)
                truths_dict[feature].append(np.where(cpu_masks, cpu_truths, np.nan))

            total_loss += sum(batch_losses)

            # Collect batch metadata
            domains.extend(batch["domain_name"])
            subsets.extend(batch["subset"])
            sequences.extend(batch["aa_seq"])

    # Average loss
    average_test_loss = total_loss / len(test_loader)

    # Concatenate results from all batches
    final_results: dict = {
        "domain": domains,
        "subset": subsets,
        "sequence": sequences
    }

    for feature in output_features:

        final_results[f"{feature}_predictions"] = np.concatenate(predictions_dict[feature])
        final_results[f"{feature}_truth"] = np.concatenate(truths_dict[feature])

    results_df = pd.DataFrame(final_results)

    del test_loader
    gc.collect()

    return average_test_loss, results_df

def save_results(results_df, results_path):

    pair = results_path.name
    # strip off the leading "trained_on_"
    if pair.startswith("trained_on_") and "_tested_on_" in pair:

        _, rest = pair.split("trained_on_", 1)
        trained_on, tested_on = rest.split("_tested_on_")

    else:

        trained_on, tested_on = "unknown", "unknown"

    # Header
    timestamp = datetime.datetime.now().isoformat()
    header_lines = [
        f"# Timestamp: {timestamp}\n",
        f"# Trained_on: {trained_on}\n",
        f"# Tested_on: {tested_on}\n"
    ]
    with open((results_path / "results.csv"), "w") as results_file:

        results_file.writelines(header_lines)

    # Map results to subsets
    # subset_rows = []

    # for subset_name, sequence_list in test_subset_to_sequence_dict.items():

    #     for sequence in sequence_list:

    #         subset_rows.append({"sequences": sequence, "subset": subset_name})

    # subset_rows_df = pd.DataFrame(subset_rows)
    # subset_df = cast(pd.DataFrame, subset_rows_df.drop_duplicates(subset = ["sequences", "subset"]))
    # results_df = pd.merge(results_df, subset_df, on = "sequences", how = "left")

    # Data
    results_df.to_csv(results_path / "results.csv", mode = "a", index = False)

    return results_df
