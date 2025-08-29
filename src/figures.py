# Standard modules
import copy

# Third-party modules
import pandas as pd
import matplotlib.pyplot as plt

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

    # First set up a folder to keep things neat
    base_folder = setup_folders()
    figures_directory = base_folder / "figures"
    figures_directory.mkdir(parents = True, exist_ok = True)

    generate_figure_1(config, figures_directory)
    generate_figure_2(config, figures_directory)
    generate_figure_3(config, figures_directory)
    generate_figure_4(config, figures_directory)
    generate_figure_5(config, figures_directory)

    #def generate_figure_1():

    # First train a model and get predictions for subs and indels individually - on aPCA and cDNA data individually
    # sub_indels_apca_no_new_data_config = copy.deepcopy(config)
    # sub_indels_cdna_no_new_data_config = copy.deepcopy(config)

    # sub_indels_apca_no_new_data_config["SUBSETS_IN_USE"] = ["APCA_WITHOUT_NEW_DATA-SUBSTITUTION", "APCA_WITHOUT_NEW_DATA-INSERTION", "APCA_WITHOUT_NEW_DATA-DELETION"]
    # sub_indels_cdna_no_new_data_config["SUBSETS_IN_USE"] = ["CDNA-DP-SUBSTITUTION", "CDNA-DP-INSERTION", "CDNA-DP-DELETION"]

def generate_figure_1(config, results_path):

    # 1a) Domainome data distribution
    # 1b) Small indel data distribution
    # 1c) Megascale data distribution
    # 1d) Performance

    figure_results_path = results_path / "figure_1" / "distributions"
    figure_results_path.mkdir(parents = True, exist_ok = True)

    # Set up config so that it's just the prenew aPCA & cDNA data to train DUMPLING
    datasets_config = copy.deepcopy(config)
    datasets_for_this_figure = ["APCA_WITHOUT_NEW_DATA", "CDNA-DP"]
    predicted_features = ["fitness", "stability_prediction"]

    # Get distribution plots
    for dataset in datasets_for_this_figure:

        for predicted_feature in predicted_features:

            dataset_path = datasets_config["DATA"]["DATASETS"][dataset]["PATH"]
            plot_dataset_distribution(dataset, dataset_path, predicted_feature, figure_results_path)

    figure_results_path = results_path / "figure_1" / "performance"
    figure_results_path.mkdir(parents = True, exist_ok = True)

    # A little bit of set up to run the model for this figure
    datasets_config["SUBSETS_IN_USE"] = [(dataset, "ALL") for dataset in datasets_for_this_figure]

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

def generate_figure_2(config, results_path):

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

def generate_figure_3(config, results_path):

    figure_results_path = results_path / "figure_3" / "new_indel_distribution"
    figure_results_path.mkdir(parents = True, exist_ok = True)
    plot_dataset_distribution("NEW_INDELS", "data/clean_data/new_indels.csv", "fitness", figure_results_path)

def generate_figure_4(config, results_path):

    # Get results for prenew dumpling
    prenew_results_path = results_path / "figure_4" / "postnew_performance"
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

def generate_figure_5(config, results_path):

    pass

    # Read in all human proteome O.O
    # Saturation mutatgenesis
    # Predict fitness of every mutant
