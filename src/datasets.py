# Standard modules
import csv

# Third-party modules
import torch
from torch.utils.data import Dataset

# Local modules
from config_loader import config
from helpers import truncate_domain, is_valid_sequence, is_tensor_ready

class ProteinDataset(Dataset):

    def __init__(
        self,
        domain_names: list,
        variant_aa_seqs: list,
        energy_values: list,
        fitness_values: list,
        energy_mask: list,
        fitness_mask: list,
        sequence_representations: list,
        ):
        
        self.domain_names = domain_names
        self.variant_aa_seqs = variant_aa_seqs
        self.energy_values = energy_values
        self.fitness_values = fitness_values
        self.energy_mask = energy_mask
        self.fitness_mask = fitness_mask
        self.sequence_representations = sequence_representations
        
        print(f"Initialised dataset of length {len(self)}")

    @classmethod
    def from_file(
        cls,
        path: str,
        domain_name_column: str,
        aa_seq_column: str,
        energy_column: str,
        fitness_column: str,
        energy_reversal: bool,
        fitness_reversal: bool,
        domain_name_splitter: str
        ):

        """
        Set up PyTorch dataset by reading in

        Parameters:
            - path (str): Path to the csv/tsv file containing the dataset.
            - domain_name_column (str): The name of the column which contains the unique domain identifiers for each domain.
            - aa_seq_column (str): The name of the column which contains the unique amino acid sequences for each variant.
            - energy_column (str): The name of the column which contains the energy measurement for each variant.
            - fitness_column (str): The name of the column which contains the fitness measurement for each variant.
            - domain_name_splitter (str): The substring which can be used to remove the suffix of the values in the domain name column.
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

                filtered_rows = [
                    row for row in rows
                    if (bool(row["energy_mask"]) or bool(row["fitness_mask"]))
                    and is_valid_sequence(str(row[aa_seq_column]), str(config["AMINO_ACIDS"]))
                ]

                domain_names, variant_aa_seqs, energy_values, fitness_values, energy_mask, fitness_mask = zip(*((
                    truncate_domain(
                        str(row[domain_name_column]),
                        domain_name_splitter),
                        str(row[aa_seq_column]),
                        float(row[energy_column]),
                        float(row[fitness_column]),
                        bool(row["energy_mask"]),
                        bool(row["fitness_mask"])
                    ) for row in filtered_rows if is_valid_sequence(
                        str(row[aa_seq_column]),
                        str(config["AMINO_ACIDS"])
                        ) and is_tensor_ready(
                            float(row[energy_column])
                            ) and is_tensor_ready(
                                float(row[fitness_column])
                                )))

                sequence_representations = [torch.zeros(0) for _ in range(len(domain_names))]

                # Apply reversals if needed
                if energy_reversal:
                    
                    energy_values = tuple(-x for x in energy_values)
                    
                if fitness_reversal:
                    
                    fitness_values = tuple(-x for x in fitness_values)
            
            return cls(
                list(domain_names),
                list(variant_aa_seqs),
                list(energy_values),
                list(fitness_values),
                list(energy_mask),
                list(fitness_mask),
                sequence_representations
            )

        except FileNotFoundError: raise FileNotFoundError(f"File {path} not found while initialising dataset.")
        except ValueError as error: raise ValueError(f"Error processing {path} while initialising dataset: {error}.")
        except Exception as error: raise Exception(f"Error processing {path} while initialising dataset: {error}.")

    def __len__(self) -> int:

        """
        Returns the length of the dataset.

        Returns:
            - length (int): The number of items in the dataset.
        """

        return len(self.variant_aa_seqs)

    def __getitem__(self, index: int) -> dict:

        """
        Returns the item in the dataset at the given index.

        Parameters:
            - index (int): The index of the item to return, zero-indexed.

        Returns:
            - variant (dict): A dictionary of the values of the variant selected via the indexing.
        """

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
    
    def filter_by_indices(self, keep_indices):
        
        """
        Returns a new ProteinDataset instance with data filtered by keep_indices.
        """
        
        return ProteinDataset(
            [self.domain_names[i] for i in keep_indices],
            [self.variant_aa_seqs[i] for i in keep_indices],
            [self.energy_values[i] for i in keep_indices],
            [self.fitness_values[i] for i in keep_indices],
            [self.energy_mask[i] for i in keep_indices],
            [self.fitness_mask[i] for i in keep_indices],
            [self.sequence_representations[i] for i in keep_indices],
        )

def get_datasets(all_datasets: list, package_folder) -> list:

    """
    Sets up the dataset with the given name using the ProteinDataset class.

    Returns:
        - datasets (ProteinDataset): A list of datasets that have been initialised.
    """

    datasets = []

    for dataset_name in all_datasets:

        try:

            dataset = ProteinDataset.from_file(
                    package_folder / "data" / config["DATASETS"][dataset_name]["PATH"],
                    config["DATASETS"][dataset_name]["DOMAIN_NAME_COLUMN"],
                    config["DATASETS"][dataset_name]["VARIANT_AA_SEQ_COLUMN"],
                    config["DATASETS"][dataset_name]["ENERGY_COLUMN"],
                    config["DATASETS"][dataset_name]["FITNESS_COLUMN"],
                    config["DATASETS"][dataset_name]["IS_ENERGY_REVERSED"],
                    config["DATASETS"][dataset_name]["IS_FITNESS_REVERSED"],
                    config["DATASETS"][dataset_name]["DROP_DOMAIN_NAME_EXTENSION"]
                )

        except Exception as error: raise Exception(f"Could not initialise dataset: {dataset_name}")

        datasets.append(dataset)

    return datasets
