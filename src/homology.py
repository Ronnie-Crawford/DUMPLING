# Standard modules
import subprocess
import shutil
import os
from pathlib import Path

MAX_FILENAME_LENGTH = 255

def handle_homology(
    dataset_dicts,
    base_path,
    force_regeneration: bool = False
    ):
    
    subset_keys = [dataset_dict["unique_key"] for dataset_dict in dataset_dicts]
    homology_path = get_homology_path(base_path, subset_keys)
    
    if force_regeneration and homology_path.exists():
        
        print(f"Forcing regeneration of {homology_path}")
        shutil.rmtree(homology_path)
    
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
    
    return homology_path

def get_homology_path(package_folder, all_dataset_names):
    
    datasets_key = "-".join(sorted(all_dataset_names))
    homology_folder_path = package_folder / "homology" / f"homology[{datasets_key}]"
    
    return Path(safe_filename(homology_folder_path))

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
    check = True,
    stdout = subprocess.DEVNULL
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

def safe_filename(path):
    
    dirpath, name = os.path.split(path)
    stem, ext = os.path.splitext(name)

    if len(name) <= MAX_FILENAME_LENGTH:
        
        return path

    # Remove vowels from the stem first
    vowels = set("aeiouAEIOU")
    no_vowel = "".join(ch for ch in stem if ch not in vowels)

    # If still too long, truncate
    truncated = no_vowel[: MAX_FILENAME_LENGTH - len(ext)]

    safe_name = truncated + ext
    safe_name = os.path.join(dirpath, safe_name)
    
    return safe_name