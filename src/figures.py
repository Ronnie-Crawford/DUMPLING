# Standard modules
import os
import copy
import shutil
import json
from pathlib import Path
import pickle

# Third-party modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
import scipy.stats as stats
from scipy.stats import spearmanr

# Local modules
from runner import setup_folders, train, test, train_and_test

def generate_figures(config: dict):

    # This function is used to recreate the figures of the paper
    # it generates the 6 following fogures
    # 1) DUMPLING architecture, DUMPLING datasets, DUMPLING performance (prenew data)
    # 2) DUMPLING performance with different PLMs, sampling, ablations, etc
    # 3) New indel experimental datasets - design, scale, quality, distributions
    # 4) Compare old + new DUMPLING against other predictors, on new data, old data, etc
    # 5) DUMPLING on proteome, clinvar

    regenerate_data_flag = False

    # First set up a folder to keep things neat
    base_folder = setup_folders()
    figures_directory = base_folder / "figures"

    if regenerate_data_flag:

        if os.path.exists(figures_directory):

            shutil.rmtree(figures_directory)

        figures_directory.mkdir(parents = True, exist_ok = True)

        # Generate the data used in each figure
        generate_figure_1_data(config, figures_directory)
        generate_figure_2_data(config, figures_directory)
        # Figure 3 needs no generated data
        generate_figure_4_data(config, figures_directory)
        generate_figure_5_data(config, figures_directory)

    # Plot each figure
    plot_figure_1(config, figures_directory)
    plot_figure_2a(figures_directory)
    plot_figure_2b(figures_directory)
    #plot_figure_3(figures_directory)
    plot_figure_4(figures_directory)

def generate_figure_1_data(config, results_path):

    # 1a) Domainome data distribution
    # 1b) Small indel data distribution
    # 1c) Megascale data distribution
    # 1d) Performance

    figure_results_path = results_path / "figure_1" / "distributions"
    figure_results_path.mkdir(parents = True, exist_ok = True)

    # Set up config so that it's just the prenew aPCA & cDNA data to train DUMPLING
    datasets_config = copy.deepcopy(config)
    subsets_for_this_figure = [
        ("APCA_WITHOUT_NEW_DATA", "SUBSTITUTION"),
        ("APCA_WITHOUT_NEW_DATA", "INSERTION"),
        ("APCA_WITHOUT_NEW_DATA", "DELETION"),
        ("CDNA-DP", "SUBSTITUTION"),
        ("CDNA-DP", "INSERTION"),
        ("CDNA-DP", "DELETION")
        ]

    figure_results_path = results_path / "figure_1" / "performance"
    figure_results_path.mkdir(parents = True, exist_ok = True)

    # A little bit of set up to run the model for this figure
    datasets_config["SUBSETS_IN_USE"] = subsets_for_this_figure

    # Set up the splits for them as well
    subsets_splits_dict = {}

    for dataset_name, label_name in config["SUBSETS_IN_USE"]:

        unique_key = f"{dataset_name}-{label_name}"
        subsets_splits_dict[unique_key] = {
            "TRAIN": 0.8,
            "VALIDATION": 0.0,
            "TEST": 0.2,
        }

    config["SUBSETS_SPLITS_DICT"] = subsets_splits_dict

    train_and_test(datasets_config, results_path_override = figure_results_path)

def generate_figure_2_data(config, results_path):

    # 2a) Compare PLMs
    # 2b) Compare other things
    compare_plms(config, results_path)
    compare_sampling(config, results_path)

def compare_plms(config, results_path):

    plm_config = copy.deepcopy(config)

    upstream_models = [
        ["ESM2_T6_8M_UR50D"],
        ["ESM2_T12_35M_UR50D"],
        ["ESM2_T30_150M_UR50D"],
        ["ESM2_T33_650M_UR50D"],
        ["AMPLIFY_120M"],
        ["AMPLIFY_120M_base"],
        ["AMPLIFY_350M"],
        ["AMPLIFY_350M_base"],
        ["PROGEN_2_SMALL"],
        ["PROGEN_2_MEDIUM"],
        ["PROSST_128"],
        ["PROSST_512"],
        ["PROSST_1024"],
        ["PROSST_2048"],
        ["PROSST_4096"],
        ["SAPROT_650M"],
    ]

    for upstream_model in upstream_models:

        figure_results_path = results_path / "figure_2" / "upstream_models" / str(upstream_model)
        figure_results_path.mkdir(parents = True, exist_ok = True)
        plm_config["UPSTREAM_MODELS_LIST"] = upstream_model
        train_and_test(plm_config, results_path_override = figure_results_path)

def compare_sampling(config, results_path):

    sampling_config = copy.deepcopy(config)
    sampling_flag_combinations = {
        "No Sampling": {
          "SAMPLING_FLAG": False,
          "SUBSET_WEIGHTING_FLAG": False,
          "SEVERITY_WEIGHTING_FLAG": False,
          "RELIABILITY_WEIGHTING_FLAG": False
        },
        "Random Sampling": {
          "SAMPLING_FLAG": True,
          "SUBSET_WEIGHTING_FLAG": False,
          "SEVERITY_WEIGHTING_FLAG": False,
          "RELIABILITY_WEIGHTING_FLAG": False
        },
        "Subset Sampling": {
          "SAMPLING_FLAG": True,
          "SUBSET_WEIGHTING_FLAG": True,
          "SEVERITY_WEIGHTING_FLAG": False,
          "RELIABILITY_WEIGHTING_FLAG": False
        },
        "Severity Sampling": {
          "SAMPLING_FLAG": True,
          "SUBSET_WEIGHTING_FLAG": False,
          "SEVERITY_WEIGHTING_FLAG": True,
          "RELIABILITY_WEIGHTING_FLAG": False
        },
        "Reliability Sampling": {
          "SAMPLING_FLAG": True,
          "SUBSET_WEIGHTING_FLAG": False,
          "SEVERITY_WEIGHTING_FLAG": False,
          "RELIABILITY_WEIGHTING_FLAG": True
        },
    }

    for sampling_name, sampling_dict in sampling_flag_combinations.items():

        figure_results_path = results_path / "figure_2" / "sampling" / sampling_name
        figure_results_path.mkdir(parents = True, exist_ok = True)
        sampling_config["DATA"]["SAMPLING"] = sampling_dict
        train_and_test(sampling_config, results_path_override = figure_results_path)

def generate_figure_4_data(config, results_path):

    # Get results for prenew dumpling
    prenew_results_path = results_path / "figure_4" / "prenew_performance"
    prenew_results_path.mkdir(parents = True, exist_ok = True)
    prenew_config = copy.deepcopy(config)
    prenew_config["DATA"]["SPLITS_FILE"]["PATH"] = "splits/prenew_splits.pkl"
    train_and_test(prenew_config, results_path_override = prenew_results_path)

    # Get results for postnew dumpling
    postnew_results_path = results_path / "figure_4" / "postnew_performance"
    postnew_results_path.mkdir(parents = True, exist_ok = True)
    postnew_config = copy.deepcopy(config)
    postnew_config["DATA"]["SPLITS_FILE"]["PATH"] = "splits/postnew_splits.pkl"
    train_and_test(postnew_config, results_path_override = postnew_results_path)

    # Get results for other predictors
    # Format thermoMPNN results into JSON format correlations
    thermoMPNN_ddmut_results = pd.read_csv("data/predictions_test_set.csv")
    megascale = pd.read_csv("data/clean_data/megascale.csv")

    thermompnn_correlations = {}
    megascale_substitution_domain_correlations = []
    megascale_insertion_domain_correlations = []
    megascale_deleton_domain_correlations = []

    for domain in thermoMPNN_ddmut_results["name"].unique():

        domain_results = thermoMPNN_ddmut_results[thermoMPNN_ddmut_results["name"] == domain]
        substitutions = []
        insertions = []
        deletions = []

        for _index, row in domain_results.iterrows():

            sequence = row["sequence"]
            matching_row = megascale[megascale["aa_seq"] == sequence]

            if (len(matching_row["ddG"]) == 1) & (matching_row["is_substitution"] == True):

                substitutions.append((matching_row["ddG"].iloc, row["ddG"]))

            elif (len(matching_row["ddG"]) == 1) & (matching_row["is_insertion"] == True):

                insertions.append((matching_row["ddG"].iloc, row["ddG"]))

            elif (len(matching_row["ddG"]) == 1) & (matching_row["is_deletion"] == True):

                deletions.append((matching_row["ddG"].iloc, row["ddG"]))

            else:

                raise ValueError("Sequence is not unique")

        domain_substitutions_spearmans_correlation = spearmanr(domain_results["thermoMPNN_pred"], experimental_truth)
        megascale_domain_correlations.append(domain_spearmans_correlation)

    thermompnn_correlations["megascale"] = {}
    thermompnn_correlations["megascale"][""] = megascale_domain_correlations

    thermompnn_results_path = results_path / "figure_4" / "thermompnn_performance" / "domain_correlations.pkl"
    pickle.dump(domain_spearmans_correlation, thermompnn_results_path)

def generate_figure_5_data(config, results_path):

    pass

    # Read in all human proteome O.O
    # Saturation mutatgenesis
    # Predict fitness of every mutant

def plot_figure_1(config, results_path):

    datasets_for_this_figure = ["APCA_WITHOUT_NEW_DATA", "CDNA-DP"]
    predicted_features = ["APCA_FITNESS", "CDNAPD_ENERGY"]

    # Plot distributions of datasets
    for dataset in datasets_for_this_figure:

        for predicted_feature in predicted_features:

            print("-------Plotting distribution------------")
            dataset_path = config["DATA"]["DATASETS"][dataset]["PATH"]
            plot_dataset_distribution(dataset, dataset_path, predicted_feature, results_path)

    # Plot performance of prenew DUMPLING on each dataset
    plot_nominal_performance(results_path, predicted_features)

def plot_dataset_distribution(dataset_name, dataset_path, predicted_feature, results_path):

    dataset = pd.read_csv(dataset_path)

    if predicted_feature == "APCA_FITNESS": predicted_feature = "fitness"

    if predicted_feature == "CDNAPD_ENERGY": predicted_feature = "stability_prediction"

    if predicted_feature in dataset.columns:

        dataset_subs = dataset[dataset["is_substitution"] == True]
        dataset_ins = dataset[dataset["is_insertion"] == True]
        dataset_dels = dataset[dataset["is_deletion"] == True]

        dataset_subs = dataset_subs.loc[~dataset_subs[predicted_feature].isna()]

        plt.figure(figsize=(10, 6))

        # Prepare range for KDE x axis (cover all data)
        all_data = np.concatenate([
            dataset_subs[predicted_feature].values,
            dataset_ins[predicted_feature].values,
            dataset_dels[predicted_feature].values
            ])
        x_min, x_max = all_data.min(), all_data.max()
        x_grid = np.linspace(x_min, x_max, 1000)

        # KDE for substitutions
        kde_subs = stats.gaussian_kde(dataset_subs[predicted_feature])
        plt.plot(x_grid, kde_subs(x_grid), color = "blue", label="Substitutions")

        # KDE for insertions
        kde_ins = stats.gaussian_kde(dataset_ins[predicted_feature])
        plt.plot(x_grid, kde_ins(x_grid), color = "green", label="Insertions")

        # KDE for deletions
        kde_dels = stats.gaussian_kde(dataset_dels[predicted_feature])
        plt.plot(x_grid, kde_dels(x_grid), color = "red", label = "Deletions")

        if predicted_feature == "fitness": x_label = "Relative Abundance"
        else: x_label = "Relative Stability"

        plt.xlabel(x_label)
        plt.ylabel("Density")
        plt.grid(axis = "y", linestyle = "--", color = "gray", alpha = 0.7)
        plt.legend()

        ax = plt.gca()
        formatter = mticker.ScalarFormatter()
        formatter.set_scientific(False)
        formatter.set_useOffset(False)
        ax.yaxis.set_major_formatter(formatter)

        plt.tight_layout()
        plt.savefig(results_path / f"dataset_distribution_{dataset_name}_{predicted_feature}.png")
        plt.close()

def old_plot_dataset_distribution(dataset_name, dataset_path, predicted_feature, results_path):

    dataset = pd.read_csv(dataset_path)
    if predicted_feature == "APCA_FITNESS": predicted_feature = "fitness"
    if predicted_feature == "CDNAPD_ENERGY": predicted_feature = "stability_prediction"

    if predicted_feature in dataset.columns:

        dataset_wts = dataset[dataset["is_wt"] == True]
        dataset_subs = dataset[dataset["is_substitution"] == True]
        dataset_ins = dataset[dataset["is_insertion"] == True]
        dataset_dels = dataset[dataset["is_deletion"] == True]

        dataset_subs = dataset_subs.loc[~dataset_subs[predicted_feature].isna()]


        plt.hist([dataset_dels[predicted_feature], dataset_ins[predicted_feature], dataset_subs[predicted_feature]], bins = 100, stacked = True, color = ["Red", "Green", "Blue"], label = ["Deletions", "Insertions", "Substitutions"])
        plt.xlabel(f"{predicted_feature}")
        plt.ylabel("Number of Variants")
        plt.legend()
        plt.savefig(results_path / f"dataset_distribution_{dataset_name}_{predicted_feature}.png")
        plt.close()

def plot_nominal_performance(results_path, predicted_features):

    # Get results file
    all_correlations = get_spearman_correlations_for_each_domain(results_path / "figure_1" / "performance"/ "metrics_by_subset.json")

    predicted_subsets = [
        "APCA_WITHOUT_NEW_DATA-SUBSTITUTION",
        "APCA_WITHOUT_NEW_DATA-INSERTION",
        "APCA_WITHOUT_NEW_DATA-DELETION",
        "CDNA-DP-SUBSTITUTION",
        "CDNA-DP-INSERTION",
        "CDNA-DP-DELETION"
        ]
    subset_correlations = {}

    for subset in predicted_subsets:

        for predicted_feature in predicted_features:

            if predicted_feature in all_correlations[subset]["domain"]:

                subset_correlations[subset] = list(all_correlations[subset]["domain"][predicted_feature]["Spearman"].values())

    # Plot each as a violin
    labels = ["aPCA\nSubstitutions", "aPCA\nInsertions", "aPCA\nDeletions", "cDNA\nSubstitutions", "cDNA\nInsertions", "cDNA\nDeletions"]
    #labels = ["aPCA\nSubstitutions", "cDNA\nSubstitutions", "cDNA\nInsertions", "cDNA\nDeletions"]
    #plt.rcParams.update({"font.size": 8, "font.family": "serif"})
    violins = plt.violinplot(list(subset_correlations.values()), showmeans = False, showmedians = True)
    #colours = ["black"] * 6
    colour_map = plt.get_cmap("tab10")
    colours = [colour_map(i % 10) for i in range(len(violins["bodies"]))]

    for body, colour in zip(violins["bodies"], colours):

        body.set_facecolor(colour)
        #body.set_linewidth(1.5)
        body.set_edgecolor("gray")
        body.set_alpha(0.3)

    violins["cbars"].set_color("black")
    violins["cmedians"].set_color("black")
    violins["cmins"].set_color("black")
    violins["cmaxes"].set_color("black")

    plt.xticks(range(1, len(labels) + 1), labels = labels, rotation = 0)
    plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.ylabel("Spearman correlation\nof predicted and true values\nper domain")
    plt.grid(axis = "y", linestyle = "--", color = "gray", alpha = 0.7)

    for index, data in enumerate(list(subset_correlations.values()), start = 1):

        # Generate jitter around the violin position i
        jitter = np.random.uniform(-0.1, 0.1, size = len(data))
        plt.scatter(np.full_like(data, index) + jitter, data, color = "black", alpha = 0.8, s = 1)

    plt.tight_layout()
    plt.savefig(results_path / "figure_1b", dpi = 2000)
    plt.clf()

def get_spearman_correlations_for_each_domain(metrics_file_path):

    spearman_data = {}

    with open(metrics_file_path, 'r') as f:

        metrics_data = json.load(f)

        # Process each subset
        for subset, metrics in metrics_data.items():

            for metric_type in ["overall", "domain"]:

                for feature, values in metrics[metric_type].items():

                    for metric_name, value in values.items():

                        if value == None:

                            continue

                        elif metric_type == "overall":

                            spearman_data.setdefault(subset, {}).setdefault(metric_type, {}).setdefault(feature, {})[metric_name] = value

                        elif metric_type == "domain":

                            for domain, domain_values in value.items():

                                spearman_data.setdefault(subset, {}).setdefault(metric_type, {}).setdefault(feature, {}).setdefault(metric_name, {})[domain] = domain_values

        return spearman_data

def plot_figure_2a(results_path):

    predicted_subsets = [
        "APCA_WITHOUT_NEW_DATA-SUBSTITUTION",
        #"APCA_WITHOUT_NEW_DATA-INSERTION",
        #"APCA_WITHOUT_NEW_DATA-DELETION",
        "CDNA-DP-SUBSTITUTION",
        "CDNA-DP-INSERTION",
        "CDNA-DP-DELETION"
        ]
    predicted_features = [
        "APCA_FITNESS",
        "CDNAPD_ENERGY"
        ]
    upstream_model_order = [
        "['ESM2_T6_8M_UR50D']",
        "['ESM2_T12_35M_UR50D']",
        "['ESM2_T30_150M_UR50D']",
        "['ESM2_T33_650M_UR50D']",
        "['AMPLIFY_120M']",
        "['AMPLIFY_120M_base']",
        "['AMPLIFY_350M']",
        "['AMPLIFY_350M_base']",
        "['PROGEN_2_SMALL']",
        "['PROGEN_2_MEDIUM']",
        "['PROSST_128']",
        "['PROSST_512']",
        "['PROSST_1024']",
        "['PROSST_2048']",
        "['PROSST_4096']",
        "['SAPROT_650M']"
        ]
    x_tick_map = {
        "['ESM2_T6_8M_UR50D']": "ESM2\n8M",
        "['ESM2_T12_35M_UR50D']": "ESM2\n35M",
        "['ESM2_T30_150M_UR50D']": "ESM2\n150M",
        "['ESM2_T33_650M_UR50D']": "ESM2\n650M",
        "['AMPLIFY_120M']": "AMPLIFY\n120M",
        "['AMPLIFY_120M_base']": "AMPLIFY\n120M-Base",
        "['AMPLIFY_350M']": "AMPLIFY\n350M",
        "['AMPLIFY_350M_base']": "AMPLIFY\n350M-Base",
        "['PROGEN_2_SMALL']": "Progen\nSmall",
        "['PROGEN_2_MEDIUM']": "Progen\nMedium",
        "['PROSST_128']": "ProSST\n128",
        "['PROSST_512']": "ProSST\n512",
        "['PROSST_1024']": "ProSST\n1024",
        "['PROSST_2048']": "ProSST\n2048",
        "['PROSST_4096']": "ProSST\n4096",
        "['SAPROT_650M']": "SaProt"
        }

    # Violin plot for PLM results
    with os.scandir(results_path / "figure_2" / "upstream_models") as plm_subfolders:

        plm_correlations = {}

        for plm_subfolder in plm_subfolders:

            plm_correlations[plm_subfolder.name] = {}
            all_correlations = get_spearman_correlations_for_each_domain(Path(plm_subfolder.path) / "metrics_by_subset.json")

            for subset in predicted_subsets:

                for predicted_feature in predicted_features:

                    if predicted_feature in all_correlations[subset]["domain"]:

                        plm_correlations[plm_subfolder.name][subset] = list(all_correlations[subset]["domain"][predicted_feature]["Spearman"].values())

        # This feels very clunky but I cant think of a better way
        ordered_plm_correlations = {}

        for model_key in upstream_model_order:

            if model_key in plm_correlations:

                subset_dict = plm_correlations[model_key]
                ordered_subset_dict = {}

                for subset_key in predicted_subsets:

                    if subset_key in subset_dict:

                        ordered_subset_dict[subset_key] = subset_dict[subset_key]

                ordered_plm_correlations[model_key] = ordered_subset_dict

        plm_correlations = ordered_plm_correlations
        flat_plm_correlations = []

        for subset_dict in plm_correlations.values():

            flat_plm_correlations.extend(subset_dict.values())

        # To force violin plots to be grouped by PLM, we manually set their positions
        num_subsets = len(predicted_subsets)
        num_plms = len(plm_correlations)
        x_positions = []
        base_positions = []
        intragroup_spacing = 0.8
        intergroup_spacing = 4.8

        for group in range(num_plms):

            base_position = group * intergroup_spacing
            base_positions.append(base_position)
            x_positions.extend([base_position + index * intragroup_spacing for index in range(num_subsets)])

        plt.figure(figsize = (20, 6))

        violins = plt.violinplot(flat_plm_correlations, positions = x_positions, showmeans = False, showmedians = True, widths = 1.0)

        #colours = ["Red", "Orange", "Yellow", "Green", "Blue", "Gray"] * (num_plms)
        #colours = ["Red"] * (num_plms)
        colour_map = plt.get_cmap("tab10")
        subset_colors = {subset: colour_map(i % colour_map.N) for i, subset in enumerate(predicted_subsets)}

        for i, body in enumerate(violins["bodies"]):
            # ith violin corresponds to subset index in repeated PLM groups because of ordering
            subset_index = i % num_subsets
            colour = subset_colors[predicted_subsets[subset_index]]
            body.set_facecolor(colour)
            body.set_linewidth(0)

        violins["cbars"].set_color("black")
        violins["cmedians"].set_color("black")
        violins["cmins"].set_color("black")
        violins["cmaxes"].set_color("black")

        # Add scatterplots with some jitter at correct positions
        for i, (plm_key, plm_dict) in enumerate(plm_correlations.items()):

            for j, subset_key in enumerate(plm_dict.keys()):

                base_pos = base_positions[i] + j * intragroup_spacing
                jittered_positions = np.random.normal(loc=base_pos, scale=0.15, size=len(plm_correlations[plm_key][subset_key]))
                plt.scatter(jittered_positions, plm_correlations[plm_key][subset_key], color="black", alpha=0.6, s=1)

        # Add formatting
        new_labels = [x_tick_map.get(label, label) for label in list(plm_correlations.keys())]
        plt.xticks(base_positions, labels = new_labels, rotation = 0, fontsize = 8)
        plt.xlabel("Upstream Protein Language Model")
        plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        plt.ylabel("Spearman correlation\nof predicted and true values\nper domain")
        plt.grid(axis = "y", linestyle = "--", color = "gray", alpha = 0.7)

        plt.tight_layout()
        plt.savefig(results_path / "figure_2a", dpi = 500)
        plt.clf()

def plot_figure_2b(results_path):

    predicted_subsets = [
        "APCA_WITHOUT_NEW_DATA-SUBSTITUTION",
        #"APCA_WITHOUT_NEW_DATA-INSERTION",
        #"APCA_WITHOUT_NEW_DATA-DELETION",
        "CDNA-DP-SUBSTITUTION",
        "CDNA-DP-INSERTION",
        "CDNA-DP-DELETION"
        ]
    predicted_features = [
        "APCA_FITNESS",
        "CDNAPD_ENERGY"
        ]
    upstream_sampling_order = [
        "No Sampling",
        "Random Sampling",
        "Subset Sampling",
        "Severity Sampling",
        "Reliability Sampling",
        ]

    # Violin plot for PLM results
    with os.scandir(results_path / "figure_2" / "sampling") as sampling_subfolders:

        sampling_correlations = {}

        for sampling_subfolder in sampling_subfolders:

            sampling_correlations[sampling_subfolder.name] = {}
            all_correlations = get_spearman_correlations_for_each_domain(Path(sampling_subfolder.path) / "metrics_by_subset.json")

            for subset in predicted_subsets:

                for predicted_feature in predicted_features:

                    if predicted_feature in all_correlations[subset]["domain"]:

                        sampling_correlations[sampling_subfolder.name][subset] = list(all_correlations[subset]["domain"][predicted_feature]["Spearman"].values())

        # This feels very clunky but I cant think of a better way
        ordered_sampling_correlations = {}

        for model_key in upstream_sampling_order:

            if model_key in sampling_correlations:

                subset_dict = sampling_correlations[model_key]
                ordered_subset_dict = {}

                for subset_key in predicted_subsets:

                    if subset_key in subset_dict:

                        ordered_subset_dict[subset_key] = subset_dict[subset_key]

                ordered_sampling_correlations[model_key] = ordered_subset_dict

        sampling_correlations = ordered_sampling_correlations
        flat_sampling_correlations = []

        for subset_dict in sampling_correlations.values():

            flat_sampling_correlations.extend(subset_dict.values())

        # To force violin plots to be grouped by PLM, we manually set their positions
        num_subsets = len(predicted_subsets)
        num_sampling = len(sampling_correlations)
        x_positions = []
        base_positions = []
        intragroup_spacing = 0.8
        intergroup_spacing = 4.0

        for group in range(num_sampling):

            base_position = group * intergroup_spacing
            base_positions.append(base_position)
            x_positions.extend([base_position + index * intragroup_spacing for index in range(num_subsets)])

        plt.figure(figsize = (16, 6))
        violins = plt.violinplot(flat_sampling_correlations, positions = x_positions, showmeans = False, showmedians = True, widths = 0.8)

        #colours = ["Red", "Orange", "Yellow", "Green", "Blue", "Gray"] * (num_plms)
        #colours = ["Red"] * (num_plms)
        colour_map = plt.get_cmap("tab10")
        subset_colors = {subset: colour_map(i % colour_map.N) for i, subset in enumerate(predicted_subsets)}

        for i, body in enumerate(violins["bodies"]):
            # ith violin corresponds to subset index in repeated PLM groups because of ordering
            subset_index = i % num_subsets
            colour = subset_colors[predicted_subsets[subset_index]]
            body.set_facecolor(colour)
            body.set_linewidth(0)

        violins["cbars"].set_color("black")
        violins["cmedians"].set_color("black")
        violins["cmins"].set_color("black")
        violins["cmaxes"].set_color("black")

        # Add scatterplots with some jitter at correct positions
        for i, (sampling_key, sampling_dict) in enumerate(sampling_correlations.items()):

            for j, subset_key in enumerate(sampling_dict.keys()):

                base_pos = base_positions[i] + j * intragroup_spacing
                jittered_positions = np.random.normal(loc=base_pos, scale=0.15, size=len(sampling_correlations[sampling_key][subset_key]))
                plt.scatter(jittered_positions, sampling_correlations[sampling_key][subset_key], color="black", alpha=0.6, s=1)

        # Add formatting
        plt.xticks(base_positions, labels = list(sampling_correlations.keys()), rotation = 0, fontsize = 8)
        plt.xlabel("Sampling Method", fontsize = 12)
        plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        plt.ylabel("Spearman correlation\nof predicted and true values\nper domain")
        plt.grid(axis = "y", linestyle = "--", color = "gray", alpha = 0.7)

        plt.tight_layout()
        plt.savefig(results_path / "figure_2b", dpi = 500)
        plt.clf()

def plot_figure_3(results_path):

    figure_results_path = results_path / "figure_3" / "new_indel_distribution"
    figure_results_path.mkdir(parents = True, exist_ok = True)
    plot_dataset_distribution("NEW_INDELS", "data/clean_data/new_indels.csv", "fitness", figure_results_path)

def plot_figure_4(results_path):

    predicted_subsets = [
        "APCA_WITHOUT_NEW_DATA-SUBSTITUTION",
        #"APCA_WITHOUT_NEW_DATA-INSERTION",
        #"APCA_WITHOUT_NEW_DATA-DELETION",
        "CDNA-DP-SUBSTITUTION",
        "CDNA-DP-INSERTION",
        "CDNA-DP-DELETION",
        "NEW_INDELS-INSERTION",
        "NEW_INDELS-DELETION"
        ]
    predicted_features = [
        "APCA_FITNESS",
        "CDNAPD_ENERGY"
        ]
    models_order = [
        "prenew_performance",
        "postnew_performance",
        ]
    x_tick_map = {
        "prenew_performance": "Pre-New DUMPLING",
        "postnew_performance": "Post-New DUMPLING",
        }

    # Violin plot for PLM results
    with os.scandir(results_path / "figure_4") as model_subfolders:

        model_correlations = {}

        for model_subfolder in model_subfolders:

            model_correlations[model_subfolder.name] = {}
            all_correlations = get_spearman_correlations_for_each_domain(Path(model_subfolder.path) / "metrics_by_subset.json")

            for subset in predicted_subsets:

                for predicted_feature in predicted_features:

                    if predicted_feature in all_correlations[subset]["domain"]:

                        model_correlations[model_subfolder.name][subset] = list(all_correlations[subset]["domain"][predicted_feature]["Spearman"].values())

        # This feels very clunky but I cant think of a better way
        ordered_model_correlations = {}

        for model_key in models_order:

            if model_key in model_correlations:

                subset_dict = model_correlations[model_key]
                ordered_subset_dict = {}

                for subset_key in predicted_subsets:

                    if subset_key in subset_dict:

                        ordered_subset_dict[subset_key] = subset_dict[subset_key]

                ordered_model_correlations[model_key] = ordered_subset_dict

        model_correlations = ordered_model_correlations
        flat_model_correlations = []

        for subset_dict in model_correlations.values():

            flat_model_correlations.extend(subset_dict.values())

        # To force violin plots to be grouped by PLM, we manually set their positions
        num_subsets = len(predicted_subsets)
        num_models = len(model_correlations)
        x_positions = []
        base_positions = []
        intragroup_spacing = 0.4
        intergroup_spacing = 3.2

        for group in range(num_models):

            base_position = group * intergroup_spacing
            base_positions.append(base_position)
            x_positions.extend([base_position + index * intragroup_spacing for index in range(num_subsets)])

        plt.figure(figsize = (16, 6))

        violins = plt.violinplot(flat_model_correlations, positions = x_positions, showmeans = False, showmedians = True, widths = 0.6)

        #colours = ["Red", "Orange", "Yellow", "Green", "Blue", "Gray"] * (num_plms)
        #colours = ["Red"] * (num_plms)
        colour_map = plt.get_cmap("tab10")
        subset_colors = {subset: colour_map(i % colour_map.N) for i, subset in enumerate(predicted_subsets)}

        for i, body in enumerate(violins["bodies"]):
            # ith violin corresponds to subset index in repeated PLM groups because of ordering
            subset_index = i % num_subsets
            colour = subset_colors[predicted_subsets[subset_index]]
            body.set_facecolor(colour)
            body.set_linewidth(0)

        violins["cbars"].set_color("black")
        violins["cmedians"].set_color("black")
        violins["cmins"].set_color("black")
        violins["cmaxes"].set_color("black")

        # Add scatterplots with some jitter at correct positions
        for i, (model_key, model_dict) in enumerate(model_correlations.items()):

            for j, subset_key in enumerate(model_dict.keys()):

                base_pos = base_positions[i] + j * intragroup_spacing
                jittered_positions = np.random.normal(loc = base_pos, scale = 0.15, size = len(model_correlations[model_key][subset_key]))
                plt.scatter(jittered_positions, model_correlations[model_key][subset_key], color = "black", alpha = 0.6, s = 1)

        # Add formatting
        new_labels = [x_tick_map.get(label, label) for label in list(model_correlations.keys())]
        plt.xticks(base_positions, labels = new_labels, rotation = 0)
        plt.xlabel("Variant Effect Predictor")
        plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        plt.ylabel("Spearman correlation\nof predicted and true values\nper domain")
        plt.grid(axis = "y", linestyle = "--", color = "gray", alpha = 0.7)
        legend_handles = [Patch(color=color, label=subset) for subset, color in subset_colors.items()]
        plt.legend(handles=legend_handles, title="Subsets")#, bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.savefig(results_path / "figure_4", dpi = 500)
        plt.clf()

def plot_figure_5():

    pass
