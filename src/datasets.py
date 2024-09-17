# Standard modules
import csv

# Third-party modules
import torch
from torch.utils.data import Dataset

# Local modules
from config_loader import config
from helpers import truncate_domain, is_valid_sequence, is_floatable

class ProteinDataset(Dataset):

    def __init__(self, path: str, domain_name_column: str, aa_seq_column: str, fitness_column: str, domain_name_splitter: str):

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

                if domain_name_splitter == None:

                    self.domain_names, self.variant_aa_seqs, self.fitness_values = zip(*((
                        #truncate_domain(str(row[domain_name_column])),
                        str(row[domain_name_column]),
                        str(row[aa_seq_column]),
                        float(row[fitness_column])
                    ) for row in rows if is_valid_sequence(str(row[aa_seq_column]), str(config["AMINO_ACIDS"])) and is_floatable(row[fitness_column])))

                else:

                    self.domain_names, self.variant_aa_seqs, self.fitness_values = zip(*((
                        truncate_domain(str(row[domain_name_column]), domain_name_splitter),
                        str(row[aa_seq_column]),
                        float(row[fitness_column])
                    ) for row in rows if is_valid_sequence(str(row[aa_seq_column]), str(config["AMINO_ACIDS"])) and is_floatable(row[fitness_column])))


                #self.sequence_representations = [None] * len(self.domain_names)
                self.sequence_representations = [torch.zeros(1280) for _ in range(len(self.domain_names))]

            print(f"Initialised dataset of length {len(self)}")

        except FileNotFoundError: raise FileNotFoundError(f"File {path} not found while initialising dataset.")
        except ValueError as error: raise ValueError(f"Error processing {path} while initialising dataset: {error}.")
        except Exception as error: raise Exception(f"Error processing {path} while initialising dataset: {error}.")

    def __len__(self):

        return len(self.variant_aa_seqs)

    def __getitem__(self, index: int):

        domain_name = self.domain_names[index]
        variant_aa_seq = self.variant_aa_seqs[index].replace("*", "<unk>")
        fitness_value = self.fitness_values[index]
        sequence_representation = self.sequence_representations[index]

        variant = {
            "domain_name": domain_name,
            "variant_aa_seq": variant_aa_seq,
            "fitness_value": fitness_value,
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
                    config["DATASETS"][dataset_name]["FITNESS_COLUMN"],
                    config["DATASETS"][dataset_name]["DROP_DOMAIN_NAME_EXTENSION"]
                )

        except Exception as error: raise Exception(f"Could not initialise dataset: {dataset_name}")

        datasets.append(dataset)

    return datasets
