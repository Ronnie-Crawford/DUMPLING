# Third-party modules
import numpy as np
from torch.utils.data import WeightedRandomSampler

def apply_sampling(split_datasets, predicted_features, subset_weighting_flag = False, severity_weighting_flag = False, reliability_weighting_flag = False):

    total_length_of_split = sum([len(dataset) for dataset in split_datasets.values()])
    weights = np.ones(total_length_of_split, dtype = np.float32)   # Initialise weights as equal for all samples
    print("Base weights: ", weights)

    if subset_weighting_flag:

        weights = weights * np.array(weight_by_subsets(split_datasets))
        print("Subset weighting: ", weights)

    if severity_weighting_flag:

        weights = weights * np.array(weight_by_severity(split_datasets, predicted_features))
        print("Severity weighting: ", weights)

    if reliability_weighting_flag:

        weights = weights * np.array(weight_by_reliability(split_datasets, predicted_features))
        print("Reliability weighting: ", weights)

    sampler = WeightedRandomSampler(
        weights = weights,
        num_samples = (len(weights)),
        replacement = True
        )

    return sampler

def weight_by_subsets(split_datasets) -> list[float]:

    # Get inverse lengths for each subset (used as weight) - Empty subsets set to 0
    weights = []

    for name, subset in split_datasets.items():

        if len(subset) > 0:

            weights.extend([(1 / len(subset))] * len(subset))

        else:

            # Subset is empty so skip
            continue

    return weights

def weight_by_severity(split_datasets, predicted_features, lower_percentile = 2, upper_percentile = 99, step = 0.05) -> list[float]:

    list_of_feature_weights = []

    for feature in predicted_features:

        feature_values = []
        feature_masks = []

        for name, subset in split_datasets.items():

            feature_values.extend(subset.feature_values[feature])
            feature_masks.extend(subset.feature_masks[feature])

        feature_values = np.array(feature_values)
        feature_masks = np.array(feature_masks)
        valid_values = feature_values[feature_masks]

        if len(valid_values) == 0:

            # No valid values to weight, all set to 0
            return np.zeros_like(feature_values, dtype = float)

        # Bin the valid values of each sample, dropping top and bottom percentiles as outliers
        low = np.percentile(valid_values, lower_percentile)
        high = np.percentile(valid_values, upper_percentile)

        low_rounded = round(low / step) * step
        high_rounded = round(high / step) * step

        # Incase of incredibly narrow distribution, smash glass
        if high_rounded <= low_rounded:

            high_rounded = low_rounded + step

        # Find how many steps we can get between the upper and lower limit
        n_bins = int(np.ceil((high_rounded - low_rounded) / step))
        # Convert the amount into the positions along the axis of predicted feature
        bin_edges = np.linspace(low_rounded, high_rounded, n_bins + 1)
        # Assign each valid sample to a bin
        bin_indices = np.digitize(feature_values, bin_edges) - 1  # -1 so bins start at 0
        # For samples outside edges (less than low, or greater than high), clamp to edge bins
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        # Count samples per bin (only valid)
        bin_counts = np.bincount(bin_indices[feature_masks], minlength = n_bins)

        # Compute weights: inverse count for each bin
        weights = np.ones_like(feature_values, dtype = float)

        for index in range(len(feature_values)):

            if not feature_masks[index]:

                weights[index] = 0.0    # Invalid values weighted to 0

            else:

                bin_index = bin_indices[index]
                count = bin_counts[bin_index] if bin_counts[bin_index] > 0 else 1
                weights[index] = 1.0 / count

        list_of_feature_weights.append(weights)

    overall_weights = [sum(weights) for weights in zip(*list_of_feature_weights)]

    return overall_weights

def weight_by_reliability(split_datasets, predicted_features):

    list_of_feature_weights = []

    for feature in predicted_features:

        feature_weights = []

        for name, subset in split_datasets.items():

            reliabilities = subset.fetch_column(f"{feature}_reliability")
            feature_weights.extend(reliabilities)

        list_of_feature_weights.append(feature_weights)

    overall_weights = [sum(weights) for weights in zip(*list_of_feature_weights)]

    return overall_weights
