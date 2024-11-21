# Standard modules
import os
import subprocess
import glob
from collections import defaultdict

# Third-party modules
import pandas as pd

def handle_homology(dataset_names, datasets, homology_path):
    
    """
    Handles the identification of homology between sequences using the phmmer method to group them into domain families.
    
    Process:
    1. Write all sequences from the datasets into a single FASTA file.
    2. Run an all-vs-all homology search using phmmer for each sequence against the database of all sequences.
    3. Parse phmmer outputs to build an adjacency list representing homologous relationships.
    4. Group sequences into domain families using this adjacency list.
    5. Print the number of identified domain families.

    Parameters:
        - dataset_names (list): A list of names for each dataset.
        - datasets (list): A list of dataset objects (or structures) that contain sequences in "variant_aa_seqs".
        - package_folder (str): The base directory where results and intermediate files will be stored.

    Returns:
        - sequence_families (list of sets): Each set contains sequence IDs that belong to the same domain family.
    """
    
    if not os.path.isdir(homology_path):
    
        # Write FASTA
        os.makedirs((homology_path), exist_ok = True)
        fasta_path = homology_path / "all_sequences.fasta"
        write_all_sequences_to_fasta(dataset_names, datasets, fasta_path)
        
        # Run PHMMER
        phmmer_output_directory = homology_path / "phmmer_output"
        os.makedirs(phmmer_output_directory, exist_ok = True)
        run_phmmer(str(fasta_path), str(phmmer_output_directory))
        
        # Parse PHMMER output and group sequences
        adjacency_list = parse_phmmer_results(str(phmmer_output_directory))
        sequence_families = group_sequences(adjacency_list)
        save_sequence_families(sequence_families, datasets, dataset_names, homology_path / "sequence_families.tsv")
        
        print(f"Homology grouping completed, found {len(sequence_families)} sequence families.")

def write_all_sequences_to_fasta(dataset_names, datasets, output_fasta):
    
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
        
        for dataset_name, dataset in zip(dataset_names, datasets):
            
            for index, sequence in enumerate(dataset.variant_aa_seqs):
                
                sequence_id = f"{dataset_name}_seq{index}"          # Construct a unique sequence ID by combining the dataset name and an index.
                fasta_file.write(f">{sequence_id}\n{sequence}\n")

def run_phmmer(input_fasta, output_dir):
    
    """
    Runs phmmer for each sequence in the input FASTA file, searching against the same FASTA as a database.
    
    Process:
    1. Read each sequence from the input FASTA file.
    2. Create a query FASTA file for each sequence.
    3. Use `phmmer` to search the query sequence against the entire database of sequences.
    4. Save the output (in table format) for each query in separate files in the output directory.

    Parameters:
        - input_fasta (str): Path to the FASTA file containing all sequences (the database).
        - output_dir (str): Path to the directory where phmmer output files will be stored.

    Returns:
        - For each sequence in the input FASTA, a file containing phmmer results will be created in the output directory.
    """
    
    sequences = []
    
    with open(input_fasta, 'r') as f:
        
        current_seq = []
        current_id = None
        
        for line in f:
            
            if line.startswith('>'):
                
                if current_id:
                    
                    sequences.append((current_id, ''.join(current_seq)))
                    
                current_id = line.strip()[1:]
                current_seq = []
                
            else:
                
                current_seq.append(line.strip())
        
        if current_id:
            
            sequences.append((current_id, ''.join(current_seq)))
            
    total_sequences = len(sequences)
    sequences_processed = 0

    for i, (seq_id, seq) in enumerate(sequences, start=1):
        
        query_fasta = f"{output_dir}/{seq_id}_query.fasta"
        
        with open(query_fasta, 'w') as f:
            
            f.write(f">{seq_id}\n{seq}\n")

        output_tbl = f"{output_dir}/{seq_id}_phmmer.tbl"

        # Run Phmmer:
        subprocess.run([
            "phmmer",
            "--tblout",
            output_tbl,
            query_fasta,
            input_fasta
        ],
        check = True,
        stdout = subprocess.DEVNULL,
        stderr = subprocess.DEVNULL
        )
        
        sequences_processed += 1
        
        progress_percent = (i / total_sequences) * 100
        print(f"Processed {i}/{total_sequences} sequences ({progress_percent:.1f}%) with phmmer...")
    
def parse_phmmer_results(output_dir, identity_threshold = 30.0):
    
    """
    Parses the phmmer results from each query and constructs an adjacency list indicating sequence homology.
    This is quite clumsy and I'd prefer to use mmseqs2 later.

    Parameters:
    - output_dir (str): The directory where phmmer output files (table format) are stored.
    - identity_threshold (float): The percent identity threshold for considering sequences as homologous.

    Process:
    1. For each phmmer output file:
       - Read the file line by line, ignoring comment lines (those starting with '#').
       - Extract homology information for each matched sequence (subject) if it meets the criteria.
    2. Build an adjacency list where keys are query sequence IDs and values are sets of sequence IDs considered homologous to the query.

    Output:
    - adjacency_list (dict): A dictionary mapping each sequence ID to a set of sequence IDs it is homologous to.
    """
    
    adjacency_list = defaultdict(set)

    for file in os.listdir(output_dir):
        
        if file.endswith("_phmmer.tbl"):
            
            # Extract the query sequence ID from the filename.
            query_id = file.replace("_phmmer.tbl", "")
            file_path = os.path.join(output_dir, file)
            
            # Read the phmmer results from the table output file.
            with open(file_path) as tbl:
                
                for line in tbl:
                    
                    if line.startswith("#"):        # Skip comments
                        
                        continue
                    
                    fields = line.strip().split()   # Split the line into fields for analysis.
                    target_id = fields[0]           # Target sequence ID.
                    e_value = float(fields[4])
                    
                    if e_value < 1e-5 and query_id != target_id:
                        
                        adjacency_list[query_id].add(target_id)
                        adjacency_list[target_id].add(query_id)

    return adjacency_list

def group_sequences(adjacency_list):
    
    """
    Groups sequences into domain families using a graph-based approach based on an adjacency list.

    Process:
    1. Initialize a set 'visited' to keep track of sequences that have already been assigned to a family.
    2. Iterate through each sequence in the adjacency list:
       - If the sequence is not visited, use a depth-first search (DFS) or breadth-first search (BFS) to find all sequences connected (directly or indirectly) to this sequence.
       - This connected component forms a domain family.
    3. Add the family (as a set of sequence IDs) to the list of domain families.

    Parameters:
    - adjacency_list (dict): A dictionary where keys are sequence IDs, and values are sets of sequence IDs that are homologous to the key.

    Output:
    - sequence_families (list of sets): Each set contains sequence IDs that belong to the same domain family.
    """
    
    visited = set()
    families = []

    for seq_id in adjacency_list:
        
        if seq_id not in visited:
            
            stack = [seq_id]
            family = set()

            while stack:
                
                current = stack.pop()
                
                if current not in visited:
                    
                    visited.add(current)
                    family.add(current)

                    for neighbor in adjacency_list[current]:
                        
                        if neighbor not in visited:
                            
                            stack.append(neighbor)

            families.append(family)

    return families

def save_sequence_families(sequence_families, datasets, dataset_names, output_tsv):
    
    """
    Saves sequence family data to a TSV file.

    Process:
    1. Create a dictionary that maps sequence IDs to their dataset name, index, and sequence 
       (the wildtype sequence).
    2. Iterate through each family in `sequence_families`, assigning a numeric family ID.
    3. For each sequence in each family, gather the required fields:
        - original_dataset: The dataset name from where the sequence originated.
        - sequence_family: The ID of the domain family.
        - sequence_name: The sequence ID.
        - sequence: The actual amino acid sequence.
    4. Write these details to the output TSV file.

    Parameters:
        - sequence_families (list of sets): A list where each set contains sequence IDs that belong to the same sequence.
        - datasets (list): Corresponding datasets, each with sequence data and sequences IDs used in sequence families. 
        - dataset_names (list): List of dataset names that match up to elements in `datasets`.
        - output_tsv (str): Path to the output TSV file where the domain family data will be saved.

    Returns:
        - A TSV file with columns: original_dataset, sequence_family, name, wildtype_sequence.
    """
    
    sequence_info = {}
    
    for dataset_name, dataset in zip(dataset_names, datasets):
        
        for index, sequence in enumerate(dataset.variant_aa_seqs):
            
            sequence_id = f"{dataset_name}_seq{index}"
            sequence_info[sequence_id] = {
                'dataset_name': dataset_name,
                'index': index,
                'sequence': sequence
            }

    with open(output_tsv, 'w') as tsv_file:
        
        tsv_file.write("original_dataset\tsequence_family\tname\tsequence\n")

        for family_id, family in enumerate(sequence_families, start = 1):
            
            for seq_id in family:
                
                if seq_id in sequence_info:
                    
                    seq_details = sequence_info[seq_id]
                    original_dataset = seq_details['dataset_name']
                    sequence = seq_details['sequence']
                    
                else:
                    
                    # This shouldn't happen
                    continue

                tsv_file.write(f"{original_dataset}\t{family_id}\t{seq_id}\t{sequence}\n")
    