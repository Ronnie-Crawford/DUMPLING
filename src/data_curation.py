# Standard modules
import math
from pathlib import Path

# Third-party modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Boltzmann constants
R = 0.0019872041
T = 298

# --- Megascale data ---
# Read in raw megascale data
megascale = pd.read_csv("data/raw_data/Tsuboyama2023_Dataset2_Dataset3_20230416.csv", usecols = ["name", "mut_type", "WT_name", "dG_ML", "ddG_ML", "aa_seq"])
# Remove background mutations
megascale = megascale.loc[megascale["WT_name"].str.endswith(".pdb", na = False)]
# Remove rows with invalid ddG
megascale["ddG_ML"] = pd.to_numeric(megascale["ddG_ML"], errors = "coerce")
megascale = megascale[megascale["ddG_ML"].notna()]
# Add derived fitness columns
megascale["fitness_type_1"] = megascale["ddG_ML"].apply(lambda ddG: (1 / (1 + math.exp(-ddG / (R * T))) - 0.5) * 2)                              # Boltzman with scaling and shifting
megascale["fitness_type_2"] = megascale["ddG_ML"].apply(lambda ddG: 7 / (1 + math.exp((-ddG/5.5) / (R * T))) - 3.5 if pd.notna(ddG) else None)   # Boltzman with scaling, shifting and biasing
megascale["fitness_type_3"] = -megascale["ddG_ML"] / 2                                                                                           # Rescaling change in free energy
# Remove samples where the wt names don't match
mismatch_wt_mask = ((megascale["mut_type"] == "wt") & (~megascale["name"].eq(megascale["WT_name"])))
megascale = megascale.loc[~mismatch_wt_mask]
# Add label columns
megascale["is_wt"] = ((megascale["mut_type"] == "wt") & (megascale["name"] == megascale["WT_name"]))
megascale["is_insertion"] = np.where(megascale["mut_type"].str.startswith("ins"), True, False)
megascale["is_deletion"] = np.where(megascale["mut_type"].str.startswith("del"), True, False)
megascale["is_substitution"] = megascale["mut_type"].str.match(r'^[A-Z]\d+[A-Z]$')
# Add "domain_name" column, without file extension
megascale["domain_name"] = megascale["name"].str.split(".pdb").str[0]
# Normalise column names
megascale = megascale.rename(columns = {"name": "variant_name", "ddG_ML": "stability_prediction"})
megascale = megascale[["domain_name", "variant_name", "aa_seq", "is_wt", "stability_prediction", "fitness_type_1", "fitness_type_2", "fitness_type_3", "is_substitution", "is_insertion", "is_deletion"]]
# Add domain classes
megascale_domain_classes = pd.read_csv("data/raw_data/rocklin_wts_domain_classification.csv")
megascale_domain_classes["domain_name"] = megascale_domain_classes["name_no_ext"]
megascale_domain_classes["scope"] = megascale_domain_classes["scope"].str[0]
# There's a few repeats and odd values, filter for consistency
megascale_domain_classes = megascale_domain_classes[megascale_domain_classes["scope"].isin(["a", "b", "c", "d"])][["domain_name", "scope"]].drop_duplicates()
megascale = pd.merge(megascale, megascale_domain_classes[["domain_name", "scope"]], on = "domain_name", how = "left")
# Populate new columns with groud-truth probability of domains containing helicies and sheets
megascale["class_a"] = np.nan
megascale["class_b"] = np.nan
megascale.loc[megascale["scope"].notnull(), "class_a"] = megascale.loc[megascale["scope"].notnull(), "scope"].isin(["a", "c", "d"]).astype(int)
megascale.loc[megascale["scope"].notnull(), "class_b"] = megascale.loc[megascale["scope"].notnull(), "scope"].isin(["b", "c", "d"]).astype(int)
# Save cleaned data
megascale.to_csv("data/clean_data/megascale.csv", index_label = "index")


# --- Domainome ---
# Read in data
domainome = pd.read_csv("data/raw_data/data_main_toni.tsv", sep = "\t")
# Derive reliability score for each sample
domainome["reliability"] = np.exp(-5.0 * domainome["normalized_fitness_sigma"]**2)
# Add label columns
domainome["is_wt"] = np.where(domainome["mut_aa"].isna(), True, False)
domainome["is_substitution"] = np.where(domainome["is_wt"] == False, True, False)
domainome["is_insertion"] = False
domainome["is_deletion"] = False
# Normalise column names
domainome = domainome.rename(columns = {"domain_ID": "domain_name", "fitness": "old_fitness", "normalized_fitness": "fitness"})[["domain_name", "aa_seq", "is_wt", "fitness", "is_substitution", "is_insertion", "is_deletion", "reliability"]]
# Save cleaned data
domainome.to_csv("data/clean_data/domainome.csv", index_label = "index")

# --- Small indels ---
# Read in data
small_indels = pd.read_csv("data/raw_data/data_main_magda.csv")
# Derive reliability score for each sample
small_indels["reliability"] = np.exp(-5.0 * small_indels["scaled_sigma"]**2)
# Add label columns
small_indels["is_wt"] = np.where(small_indels["type"] == "wt", True, False)
small_indels["is_substitution"] = False
small_indels["is_insertion"] = np.where(small_indels["mut_type"] == "insertions", True, False)
small_indels["is_deletion"] = np.where(small_indels["mut_type"] == "deletions", True, False)
# Normalise column names
small_indels = small_indels.rename(columns = {"domain": "domain_name", "scaled_fitness": "fitness"})[["domain_name", "aa_seq", "is_wt", "fitness", "is_substitution", "is_insertion", "is_deletion", "reliability"]]
# Save cleaned data
small_indels.to_csv("data/clean_data/small_indels.csv", index_label = "index")

# --- New indels ---
new_indels = pd.read_csv("data/raw_data/aPCA_new_indels_normalised_2.csv")
#new_indels["unique_key"] = new_indels["ID"] + "_" + new_indels["mut"]
# Derive reliability score for each sample
new_indels["reliability"] = np.exp(-5.0 * new_indels["scaled_sigma"]**2)
# Add label columns
new_indels["is_wt"] = np.where(new_indels["mutation_type"] == "wt", True, False)
new_indels["is_substitution"] = False
new_indels["is_insertion"] = np.where(new_indels["mutation_type"] == "ins", True, False)
new_indels["is_deletion"] = np.where(new_indels["mutation_type"] == "del", True, False)
# Normalise column names
new_indels = new_indels.rename(columns = {"ID": "domain_name", "fitness": "absolute_fitness", "scaled_fitness": "fitness"})[["domain_name", "aa_seq", "is_wt", "fitness", "is_substitution", "is_insertion", "is_deletion", "reliability"]]
new_indels.to_csv("data/clean_data/new_indels.csv", index_label = "index")

# --- Combined aPCA data ---
# Join the datasets
prenew_combined_apca = pd.concat([domainome, small_indels], axis = 0)
postnew_combined_apca = pd.concat([domainome, small_indels, new_indels], axis = 0)
prenew_combined_apca.to_csv("data/clean_data/prenew_apca.csv", index_label = "index")
postnew_combined_apca.to_csv("data/clean_data/postnew_apca.csv", index_label = "index")

# --- Clinvar ---
# Gather the mutants from separate files:
clinvar_subs_folder = Path("data/raw_data/clinical_ProteinGym_substitutions")
clinvar_subs_csv_files = clinvar_subs_folder.glob("*.csv")
clinvar_subs_dfs = []

for file_path in clinvar_subs_csv_files:

    temp_df = pd.read_csv(file_path)
    temp_df["is_wt"] = False
    temp_df["is_substitution"] = True
    temp_df["is_insertion"] = False
    temp_df["is_deletion"] = False
    temp_df["dms_spoof"] = 0

    wt_row = {
        "protein": temp_df["protein"].iloc[0],
        "mutated_sequence": temp_df["protein_sequence"].iloc[0],
        "DMS_bin_score": np.nan,
        "is_wt": True,
        "is_substitution": False,
        "is_insertion": False,
        "is_deletion": False,
        "dms_spooof": 0
        }

    temp_df = pd.concat([pd.DataFrame([wt_row]), temp_df], ignore_index = True)
    clinvar_subs_dfs.append(temp_df)

clinvar_subs = pd.concat(clinvar_subs_dfs, ignore_index = True)
clinvar_subs = clinvar_subs.rename(columns = {"mutated_sequence": "aa_seq", "protein": "uniprot_name", "DMS_bin_score": "ClinicalSignificance"})

clinvar_indels_folder = Path("data/raw_data/clinical_ProteinGym_substitutions")
clinvar_indels_csv_files = clinvar_indels_folder.glob("*.csv")
clinvar_indels_dfs = []

for file_path in clinvar_indels_csv_files:

    temp_df = pd.read_csv(file_path)
    temp_df["is_wt"] = False
    temp_df["is_substitution"] = False
    temp_df["is_insertion"] = np.where(temp_df["length"] > temp_df["length"].iloc[0], True, False)
    temp_df["is_deletion"] = np.where(temp_df["length"] < temp_df["length"].iloc[0], True, False)
    temp_df["dms_spoof"] = 0

    wt_row = {
        "protein": temp_df["protein"].iloc[0],
        "mutated_sequence": temp_df["protein_sequence"].iloc[0],
        "DMS_bin_score": np.nan,
        "is_wt": True,
        "is_substitution": False,
        "is_insertion": False,
        "is_deletion": False,
        "dms_spooof": 0
        }

    temp_df = pd.concat([pd.DataFrame([wt_row]), temp_df], ignore_index = True)
    clinvar_indels_dfs.append(temp_df)

clinvar_indels = pd.concat(clinvar_indels_dfs, ignore_index = True)
clinvar_indels = clinvar_indels.rename(columns = {"mutated_sequence": "aa_seq", "protein": "uniprot_name", "DMS_bin_score": "ClinicalSignificance"})
clinvar = pd.concat([clinvar_subs, clinvar_indels], ignore_index = True)

clinvar.to_csv("data/clean_data/clinvar.csv", index_label = "index")
