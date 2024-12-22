# Import standard modules
import subprocess
import os

def handle_homology(datasets_dict, homology_path):
    
    if not os.path.isdir(homology_path):
    
        # Write FASTA
        os.makedirs((homology_path), exist_ok = True)
        fasta_path = homology_path / "all_sequences.fasta"
        write_all_sequences_to_fasta(datasets_dict, fasta_path)
        run_mmseqs2(homology_path, fasta_path)
        
        # Parse mmseqs2 cluster results
        cluster_file = homology_path / "clustered_sequences_cluster.tsv"
        sequence_families = parse_mmseqs2_clusters(cluster_file)

        # Build sequence info mapping
        sequence_info = build_sequence_info(datasets_dict)

        # Save the sequence families
        output_tsv = homology_path / "sequence_families.tsv"
        save_sequence_families(sequence_families, sequence_info, output_tsv)

        print(f"Homology grouping completed, found {len(sequence_families)} sequence families.")


def write_all_sequences_to_fasta(datasets_dict, output_fasta):
    
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
        
        for dataset_name, dataset in datasets_dict["all"].items():
            
            for index, sequence in enumerate(dataset.variant_aa_seqs):
                
                sequence_id = f"{dataset_name}_seq{index}"          # Construct a unique sequence ID by combining the dataset name and an index.
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


def build_sequence_info(datasets_dict):
    
    """
    Builds a dictionary mapping sequence_id -> {dataset_name, index, sequence}
    """
    
    sequence_info = {}
    
    for dataset_name, dataset in datasets_dict["all"].items():
        
        for index, sequence in enumerate(dataset.variant_aa_seqs):
            
            sequence_id = f"{dataset_name}_seq{index}"
            sequence_info[sequence_id] = {
                "dataset_name": dataset_name,
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
                    original_dataset = seq_details["dataset_name"]
                    sequence = seq_details["sequence"]
                    tsv_file.write(f"{original_dataset}\t{family_id}\t{seq_id}\t{sequence}\n")


    
       