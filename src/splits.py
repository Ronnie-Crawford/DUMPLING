import numpy as np
import pandas as pd
from datasets import get_dataset_info

def split_data(df, dataset_name, train_size=0.75, val_size=0.10, test_size=0.15, random_state=None):
    """
    Splits the data into training, validation, and testing sets while ensuring all rows with the same domain are in the same split.
    
    Parameters:
        df (DataFrame): The input data.
        dataset_name (str): The name of the dataset to determine column names.
        train_size (float): Proportion of the data to include in the training set.
        val_size (float): Proportion of the data to include in the validation set.
        test_size (float): Proportion of the data to include in the testing set.
        random_state (int): Random seed for reproducibility.
    
    Returns:
        DataFrame: Training set.
        DataFrame: Validation set.
        DataFrame: Testing set.
    """
    dataset_info = get_dataset_info(dataset_name)
    if not dataset_info:
        raise ValueError(f"Dataset {dataset_name} is not defined in datasets.py")
    
    domain_column = dataset_info.domain_column
    assert train_size + val_size + test_size == 1.0, "Train, validation, and test sizes must sum to 1"
    
    if random_state:
        np.random.seed(random_state)
    
    domains = df[domain_column].unique()
    np.random.shuffle(domains)
    
    # Calculate number of domains to assign to each split
    n_total_domains = len(domains)
    n_train_domains = int(n_total_domains * train_size)
    n_val_domains = int(n_total_domains * val_size)
    
    # Ensure at least one domain is assigned to each split
    if n_val_domains == 0:
        n_val_domains = 1
    if n_total_domains - n_train_domains - n_val_domains == 0:
        n_val_domains += 1
    
    train_domains = domains[:n_train_domains]
    val_domains = domains[n_train_domains:n_train_domains + n_val_domains]
    test_domains = domains[n_train_domains + n_val_domains:]
    
    train_df = df[df[domain_column].isin(train_domains)]
    val_df = df[df[domain_column].isin(val_domains)]
    test_df = df[df[domain_column].isin(test_domains)]
    
    # Debugging statements
    print(f"Domains: {domains}")
    print(f"Train domains: {train_domains}")
    print(f"Validation domains: {val_domains}")
    print(f"Test domains: {test_domains}")
    print(f"Train dataset size: {len(train_df)}")
    print(f"Validation dataset size: {len(val_df)}")
    print(f"Test dataset size: {len(test_df)}")

    return train_df, val_df, test_df
