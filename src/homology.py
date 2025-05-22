# Import standard modules
import subprocess
import os

# Local modules
from helpers import get_homology_path

def handle_homology(dataset_dicts, base_path, splits_method_choice):
    
    if splits_method_choice != "HOMOLOGOUS_AWARE":
        
        return None
    
    dataset_unique_keys = [dataset_dict["unique_key"] for dataset_dict in dataset_dicts]
    homology_path = get_homology_path(base_path, dataset_unique_keys)
    
    if not os.path.isdir(homology_path):
    
        # Write FASTA
        os.makedirs((homology_path), exist_ok = True)
        fasta_path = homology_path / "all_sequences.fasta"
        write_all_sequences_to_fasta(dataset_dicts, fasta_path)
        run_mmseqs2(homology_path, fasta_path)
        
        # Parse mmseqs2 cluster results
        cluster_file = homology_path / "clustered_sequences_cluster.tsv"
        sequence_families = parse_mmseqs2_clusters(cluster_file)

        # Build sequence info mapping
        sequence_info = build_sequence_info(dataset_dicts)

        # Save the sequence families
        output_tsv = homology_path / "sequence_families.tsv"
        save_sequence_families(sequence_families, sequence_info, output_tsv)
        
        # Pick representatives from each domain family
        representatives_fasta = homology_path / "representatives_fasta.fasta"
        write_representative_sequences(sequence_families, sequence_info, representatives_fasta)
        
        # Below is only needed if descriptions for each domain are desired but requires database FASTA file for PHMMER to use
        
        # Run PHMMER
        #phmmer_output = homology_path / "phmmer_output.tbl"
        #run_phmmer_for_descriptions(representatives_fasta, phmmer_output)

        # Fetch descriptions
        #rep_descriptions = parse_phmmer_descriptions(phmmer_output)
    
        # Map descriptions back to families
        #family_descriptions = {}
        
        #for family_id, family in enumerate(sequence_families, start = 1):
            
        #    representative = list(family)[0]
        #    description = rep_descriptions.get(representative, "No description found")
        #    family_descriptions[family_id] = description
        
        #return family_descriptions

        print(f"Homology grouping completed, found {len(sequence_families)} sequence families.")
        
    return homology_path

def write_all_sequences_to_fasta(dataset_dicts, output_fasta):
    
    """
    Writes all sequences from the given datasets to a single FASTA file.

    Parameters: 
        - dataset_names (list): A list of dataset names, each corresponding to an entry in `datasets`.
        - datasets (list): A list of datasets where each dataset's sequences are to be written to the FASTA file.
        - output_fasta (str): Path to the output FASTA file where all sequences will be written.

    Returns:
        - A FASTA file containing all sequences from all datasets with unique IDs.
    """
    
    print("Writing FASTA file with all sequences...")
    
    with open(output_fasta, 'w') as fasta_file:
        
        for dataset_dict in dataset_dicts:
            
            dataset = dataset_dict["dataset"]
            dataset_unique_key = dataset_dict["unique_key"]
            
            for index, sequence in enumerate(dataset.aa_seqs):
                
                sequence_id = f"{dataset_unique_key}_seq{index}"   # Construct a unique sequence ID by combining the dataset-label group name and an index.
                fasta_file.write(f">{sequence_id}\n{sequence}\n")
                
def run_mmseqs2(homology_path, fasta_path):          
      
    temp_directory = homology_path / "tmp"
      
    subprocess.run([
    "mmseqs",
    "easy-cluster",
    fasta_path,
    "clustered_sequences",
    temp_directory
    ],
    cwd = homology_path,
    check = True
    )
    
def parse_mmseqs2_clusters(cluster_file):
    
    """
    Parses the mmseqs2 cluster file. Each line is of the form:
    representative_sequence_id    member_sequence_id
    We will group by the representative_sequence_id as the family.

    Returns:
        sequence_families (list of sets): Each set is a family of sequence IDs.
    """
    
    cluster_dict = {}
    
    with open(cluster_file, 'r') as cf:
        
        for line in cf:
            
            rep, member = line.strip().split('\t')
            
            if rep not in cluster_dict:
                
                cluster_dict[rep] = set()
                
            cluster_dict[rep].add(member)
            cluster_dict[rep].add(rep)

    sequence_families = list(cluster_dict.values())
    
    return sequence_families

def build_sequence_info(dataset_dicts):
    
    """
    Builds a dictionary mapping sequence_id -> {dataset_name, index, sequence}
    """
    
    sequence_info = {}

    for dataset_dict in dataset_dicts:
        
        dataset_name = dataset_dict["dataset_name"]
        dataset = dataset_dict["dataset"]
        label = dataset_dict["label"]
        dataset_unique_key = dataset_dict["unique_key"]
        
        for index, sequence in enumerate(dataset.aa_seqs):
            
            sequence_id = f"{dataset_unique_key}_seq{index}"
            sequence_info[sequence_id] = {
                "dataset_group_name": dataset_unique_key,
                "index": index,
                "sequence": sequence
            }

    return sequence_info

def save_sequence_families(sequence_families, sequence_info, output_tsv):
    
    """
    Saves sequence family data to a .tsv file with the same format as old code:
    original_dataset    sequence_family    name    sequence
    """
    
    with open(output_tsv, 'w') as tsv_file:
        
        tsv_file.write("original_dataset\tsequence_family\tname\tsequence\n")

        for family_id, family in enumerate(sequence_families, start = 1):
            
            for seq_id in family:
                
                if seq_id in sequence_info:
                    
                    seq_details = sequence_info[seq_id]
                    original_dataset = seq_details["dataset_group_name"]
                    sequence = seq_details["sequence"]
                    tsv_file.write(f"{original_dataset}\t{family_id}\t{seq_id}\t{sequence}\n")

def write_representative_sequences(sequence_families, sequence_info, output_fasta):
    
    """
    Writes one representative sequence from each domain family to a FASTA file.
    
    Parameters:
        - sequence_families (list of sets): Domain families (each a set of sequence IDs).
        - sequence_info (dict): Mapping of sequence_id -> {dataset_name, index, sequence}.
        - output_fasta (str): Path to the output FASTA file.
    """
    
    with open(output_fasta, 'w') as fasta_file:
        
        for family in sequence_families:
            
            representative = list(family)[0]
            seq_details = sequence_info.get(representative)
            
            if seq_details:
                
                fasta_file.write(f">{representative}\n{seq_details['sequence']}\n")

def run_phmmer_for_descriptions(query_fasta, output_file):
    
    """
    Runs phmmer to search each representative sequence against the target database.
    
    Parameters:
        - query_fasta (str): FASTA file of representative sequences.
        - output_file (str): File where phmmer output will be saved.
        - database (str): Path to the database to search against.
    """
    
    database = query_fasta
    
    subprocess.run([
        "phmmer",
        "--tblout",
        output_file,
        query_fasta,
        database
    ],
    check = True,
    stdout = subprocess.DEVNULL,
    stderr = subprocess.DEVNULL)

def parse_phmmer_descriptions(tbl_file):
    
    """
    Parses the phmmer output file to extract domain descriptions for each query.
    
    Parameters:
        - tbl_file (str): The phmmer output file (table format).
    
    Returns:
        - descriptions (dict): Mapping from query sequence ID to its domain description.
        
    Note:
        The column index for the description may vary depending on the phmmer version/output format.
    """
    
    descriptions = {}
    
    with open(tbl_file, 'r') as file:
        
        for line in file:
            
            if line.startswith("#"):
                
                continue
            
            fields = line.strip().split()
            query_id = fields[2]

            description = " ".join(fields[18:]) if len(fields) > 18 else "No description"
            descriptions[query_id] = description
            
    return descriptions

def add_family_descriptions(sequence_families, sequence_info, homology_path, database):
    
    """
    Integrates domain descriptions for each family.
    
    Process:
        1. Write representative sequences to a FASTA file.
        2. Run phmmer on these sequences.
        3. Parse phmmer output to get descriptions.
        4. Return a mapping of family ID to domain description.
    """
    
    pass