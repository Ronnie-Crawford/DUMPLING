# DUMPLING
### Delta of Unseen Mutant Protein Language embeddings for INdel effect Generation

This is a variant effect predictor which attempts to predict the fitness, or other feature, of a variant protein relative to the wildtype protein. It only requires the wildtype and mutant sequence.

## Environment

You can set up the environment for this package using conda using the command: ```conda env create -f environment.yml```.

## Usage

1. Prepare your dataset and update the ```config.py``` file with the correct paths and parameters. Ensure you add the details of your dataset to the ```DATASETS``` dictionary.

2. Run the main script with:
    ```python src```

## Configuration

All configurations such as data paths, model parameters, and training parameters can be found and updated in the `config.py` file.

## Flags

- ```--run```       - Allows the selection of which pipeline to run, currently the options are ```[train-test]```, ```[train]```, ```[test]```, or ```[ava-benchmarking]```.

The train-test option trains once on all selected dataset subsets together, and then tests on all selected subsets together, as described by the splits by label options in the config. The ava-benchmarking (all-against-all) trains a separate model on each subset, and then tests separately on each subset, generating a matrix of results for each pair of training and testing.

## Modules

- ```__main__.py```         - The main function which reads flags and executes functions based on user selection.
- ```config_loader.py```    - Reads the values in the config file and parses them for the rest of the package to use.
- ```datasets.py```         - Defines the dataset class to store data in, and contains a function to set up datasets from config.
- ```inference.py```        - Contains functions to run trained models to get predictions from datasets.
- ```models.py```           - Contains model classes, and function to set it up from config.
- ```embeddings.py```       - Contains functions for fetching and using protein language models to generate upstream embeddings.
- ```splits.py```           - Contains functions to read homology files and split data based on config.
- ```training.py```         - Contains functions to train models on datasets.
- ```benchmarking.py```     - First trains a model on each selected subset, then tests each model on each subset selected for testing.
- ```homology.py```         - Makes use of ```mmseqs2``` to find homology clusters within all data, so that data homologous to training data is not used in testing, reducing model leakage.
- ```metrics.py```          - Uses results of inference to calculate performance metrics for overall and domain-specific results.
- ```runner.py```           - This module outlines the structure of each pipeline, calling other modules as needed.
- ```visuals.py```          - Handles the creation of any plots and figures.
