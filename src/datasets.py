# Standard modules
import csv

# Third-party modules
import torch
from torch.utils.data import Dataset

# Local modules
from config_loader import config
from helpers import truncate_domain, is_valid_sequence, is_tensor_ready

class ProteinDataset(Dataset):

    def __init__(self, path: str, domain_name_column: str, aa_seq_column: str, energy_column: str, fitness_column: str, domain_name_splitter: str):

        """
        Set up PyTorch dataset by reading in

        Parameters:
            - path (str): Path to the csv/tsv file containing the dataset.
            - domain_name_column (str): The name of the column which contains the unique domain identifiers for each domain.
            - aa_seq_column (str): The name of the column which contains the unique amino acid sequences for each variant.
            - fitness_column (str): The name of the column which contains the fitness of energy measurement for each variant.
        """

        try:

            # Detect delimiter by reading the first line of the file.
            with open(path, mode = "r") as data_file:

                first_line = data_file.readline()
                delimiter = "," if first_line.count(",") > first_line.count("\t") else "\t"

            # Reopen file and read columns of data in
            with open(path, mode = "r") as data_file:

                reader = csv.DictReader(data_file, delimiter = delimiter)
                rows = list(reader)
                
                # If no energy or fitness column given, mask out their values as unknown
                if energy_column == "":
                    
                    for row in rows:
                        
                        row["energy_values"] = False
                        
                    energy_column = "energy_values"
                    
                if fitness_column == "":
                    
                    for row in rows:
                        
                        row["fitness_values"] = False
                        
                    fitness_column = "fitness_values"
                
                # Also mask out any values in given columns that are not parsable
                for row in rows:
                    
                    if is_tensor_ready(row[energy_column]):
                        
                        row["energy_mask"] = True

                    else:

                        row[energy_column] = 0.0
                        row["energy_mask"] = False
                    
                    if is_tensor_ready(row[fitness_column]):
                        
                        row["fitness_mask"] = True

                    else:

                        row[fitness_column] = 0.0
                        row["fitness_mask"] = False

                self.domain_names, self.variant_aa_seqs, self.energy_values, self.fitness_values, self.energy_mask, self.fitness_mask = zip(*((
                    truncate_domain(str(row[domain_name_column]), domain_name_splitter),
                    str(row[aa_seq_column]),
                    float(row[energy_column]),
                    float(row[fitness_column]),
                    bool(row["energy_mask"]),
                    bool(row["fitness_mask"])
                ) for row in rows if is_valid_sequence(str(row[aa_seq_column]), str(config["AMINO_ACIDS"])) and is_tensor_ready(float(row[energy_column])) and is_tensor_ready(float(row[fitness_column]))))

                self.sequence_representations = [torch.zeros(0) for _ in range(len(self.domain_names))]

            print(f"Initialised dataset of length {len(self)}")

        except FileNotFoundError: raise FileNotFoundError(f"File {path} not found while initialising dataset.")
        except ValueError as error: raise ValueError(f"Error processing {path} while initialising dataset: {error}.")
        except Exception as error: raise Exception(f"Error processing {path} while initialising dataset: {error}.")

    def __len__(self):

        return len(self.variant_aa_seqs)

    def __getitem__(self, index: int):

        domain_name = self.domain_names[index]
        variant_aa_seq = self.variant_aa_seqs[index].replace("*", "<unk>")
        energy_value = self.energy_values[index]
        fitness_value = self.fitness_values[index]
        energy_mask = self.energy_mask[index]
        fitness_mask = self.fitness_mask[index]
        sequence_representation = self.sequence_representations[index]

        variant = {
            "domain_name": domain_name,
            "variant_aa_seq": variant_aa_seq,
            "energy_value": energy_value,
            "fitness_value": fitness_value,
            "energy_mask": energy_mask,
            "fitness_mask": fitness_mask,
            "sequence_representation": sequence_representation
        }

        return variant

def get_datasets():

    """
    Sets up the dataset with the given name using the ProteinDataset class.

    Returns:
        - datasets (ProteinDataset): A list of datasets that have been initialised.
    """

    datasets = []

    for dataset_name in config["DATASETS_IN_USE"]:

        try:

            dataset = ProteinDataset(
                    config["DATASETS"][dataset_name]["PATH"],
                    config["DATASETS"][dataset_name]["DOMAIN_NAME_COLUMN"],
                    config["DATASETS"][dataset_name]["VARIANT_AA_SEQ_COLUMN"],
                    config["DATASETS"][dataset_name]["ENERGY_COLUMN"],
                    config["DATASETS"][dataset_name]["FITNESS_COLUMN"],
                    config["DATASETS"][dataset_name]["DROP_DOMAIN_NAME_EXTENSION"]
                )

        except Exception as error: raise Exception(f"Could not initialise dataset: {dataset_name}")

        datasets.append(dataset)

    return datasets
