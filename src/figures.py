# Standard modules
import os
import copy
import shutil
import json
from pathlib import Path

# Third-party modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

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
    plot_figure_2(figures_directory)
    plot_figure_3(figures_directory)

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
        ["SAPROT_650M"]
    ]

    for upstream_model in upstream_models:

        figure_results_path = results_path / "figure_2" / "upstream_models" / str(upstream_model)
        figure_results_path.mkdir(parents = True, exist_ok = True)
        plm_config["UPSTREAM_MODELS_LIST"] = upstream_model
        train_and_test(plm_config, results_path_override = figure_results_path)

def compare_sampling(config, results_path):

    sampling_config = copy.deepcopy(config)

    sampling_flag_combinations = [
        {
          "SAMPLING_FLAG": False,
          "SUBSET_WEIGHTING_FLAG": False,
          "SEVERITY_WEIGHTING_FLAG": False,
          "RELIABILITY_WEIGHTING_FLAG": False
        },
        {
          "SAMPLING_FLAG": True,
          "SUBSET_WEIGHTING_FLAG": False,
          "SEVERITY_WEIGHTING_FLAG": False,
          "RELIABILITY_WEIGHTING_FLAG": False
        },
        {
          "SAMPLING_FLAG": True,
          "SUBSET_WEIGHTING_FLAG": True,
          "SEVERITY_WEIGHTING_FLAG": False,
          "RELIABILITY_WEIGHTING_FLAG": False
        },
        {
          "SAMPLING_FLAG": True,
          "SUBSET_WEIGHTING_FLAG": False,
          "SEVERITY_WEIGHTING_FLAG": True,
          "RELIABILITY_WEIGHTING_FLAG": False
        },
        {
          "SAMPLING_FLAG": True,
          "SUBSET_WEIGHTING_FLAG": False,
          "SEVERITY_WEIGHTING_FLAG": False,
          "RELIABILITY_WEIGHTING_FLAG": True
        },
    ]

    for sampling_dict in sampling_flag_combinations:

        figure_results_path = results_path / "figure_2" / "sampling" / str(sampling_dict.values())
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

            dataset_path = config["DATA"]["DATASETS"][dataset]["PATH"]
            plot_dataset_distribution(dataset, dataset_path, predicted_feature, results_path)

    # Plot performance of prenew DUMPLING on each dataset
    plot_nominal_performance(results_path, predicted_features)

def plot_dataset_distribution(dataset_name, dataset_path, predicted_feature, results_path):

    dataset = pd.read_csv(dataset_path)

    if predicted_feature in dataset.columns:

        dataset_wts = dataset[dataset["is_wt"] == True]
        dataset_subs = dataset[dataset["is_substitution"] == True]
        dataset_ins = dataset[dataset["is_insertion"] == True]
        dataset_dels = dataset[dataset["is_deletion"] == True]

        plt.hist([dataset_wts[predicted_feature]], bins = 100, stacked = True, color = ["black"], label = ["Wildtype"])
        plt.hist([dataset_subs[predicted_feature]], bins = 100, stacked = True, color = ["red"], label = ["Substitutions"])
        plt.hist([dataset_ins[predicted_feature]], bins = 100, stacked = True, color = ["blue"], label = ["Insertions"])
        plt.hist([dataset_dels[predicted_feature]], bins = 100, stacked = True, color = ["green"], label = ["Deletions"])
        plt.title(f"Histogram of {dataset_name} {predicted_feature}")
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
    plt.rcParams.update({"font.size": 8, "font.family": "serif"})
    violins = plt.violinplot(list(subset_correlations.values()), showmeans = False, showmedians = True)
    colours = ["black"] * 6

    for body, colour in zip(violins["bodies"], colours):

        body.set_facecolor(colour)
        #body.set_linewidth(1.5)
        body.set_edgecolor("gray")   # Optional: set edge color
        body.set_alpha(0.3)

    violins["cbars"].set_color("black")
    violins["cmedians"].set_color("black")
    violins["cmins"].set_color("black")
    violins["cmaxes"].set_color("black")

    plt.xticks(range(1, len(labels) + 1), labels = labels, rotation = 90)
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

def plot_figure_2(results_path):

    # Violin plot for PLM results
    plm_results = {}

    with os.scandir(results_path / "figure_2" / "upstream_models") as plm_subfolders:

        plm_correlations = []

        for plm_subfolder in plm_subfolders:

            all_correlations = get_spearman_correlations_for_each_domain(Path(plm_subfolder.path) / "metrics_by_subset.json")

            predicted_subsets = [
                "APCA_WITHOUT_NEW_DATA-SUBSTITUTION",
                "APCA_WITHOUT_NEW_DATA-INSERTION",
                "APCA_WITHOUT_NEW_DATA-DELETION",
                "CDNA-DP-SUBSTITUTION",
                "CDNA-DP-INSERTION",
                "CDNA-DP-DELETION"
                ]
            predicted_features = ["APCA_FITNESS", "CDNAPD_ENERGY"]

            # for subset in predicted_subsets:

            #     for predicted_feature in predicted_features:

            #         if predicted_feature in all_correlations[subset]["domain"]:

            #             #plm_correlations[plm_subfolder][subset] = list(all_correlations[subset]["domain"][predicted_feature]["Spearman"].values())
            #             plm_correlations.append(list(all_correlations[subset]["domain"][predicted_feature]["Spearman"].values()))

            #for subset in predicted_subsets:

            for predicted_feature in predicted_features:

                if predicted_feature in all_correlations["CDNA-DP-SUBSTITUTION"]["domain"]:

                    plm_correlations.append(list(all_correlations["CDNA-DP-SUBSTITUTION"]["domain"][predicted_feature]["Spearman"].values()))


        violins = plt.violinplot(list(plm_correlations), showmeans = False, showmedians = True)
        plt.tight_layout()
        plt.savefig(results_path / "figure_2a", dpi = 2000)



    # Violin plot for sampling results
    sampling_results = {}

    with os.scandir(results_path / "figure_2" / "sampling") as sampling_subfolders:

        for sampling_subfolder in plm_subfolders:

            raw_results_df = pd.read_csv(sampling_subfolder.path, comment = "#")
            # Add outputs to new column (maps both abundance and stability to "data_to_plot" column)
            raw_results_df["data_to_plot"] = np.where(
                raw_results_df["subset"].str.contains("APCA"), raw_results_df["fitness"],
                np.where(raw_results_df["subset"].str.contains("CDNA"), raw_results_df["stability"],
                np.nan
                ))
            sampling_results[sampling_subfolder.name] = raw_results_df

    data_to_plot = [sampling_result["data_to_plot"] for sampling_result in list(sampling_results.values())]
    # Plot each as a violin

    positions = []
    group_spacing = 2  # space between groups
    violin_spacing = 0.3  # overlap spacing within groups

    for g in range(16):
        base_pos = g * group_spacing
        positions.extend([base_pos + i * violin_spacing for i in range(80)])

    plt.violinplot(data_to_plot, positions = positions, showmeans = True, showmedians = True)
    plt.xticks(range(1, len(sampling_results.keys()) + 1), labels = list(plm_results.keys()))
    plt.show()
    plt.clf()

def plot_figure_3(results_path):

    figure_results_path = results_path / "figure_3" / "new_indel_distribution"
    figure_results_path.mkdir(parents = True, exist_ok = True)
    plot_dataset_distribution("NEW_INDELS", "data/clean_data/new_indels.csv", "fitness", figure_results_path)

def plot_figure_4(figures_directory):

    pass
