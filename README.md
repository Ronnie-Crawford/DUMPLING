# Protein Domain Stability Predictor

The goal of this project is to explore how different architectures and how different datasets, and features of datasets, impact the accuracy of predicting fitness and stability of protein domains.

Currently 2 architectures have been tried:

- An RNN autoencoder to represent a sequence, with a downstream fitness predictor.
- ESM2 embeddings with downstream stability prediction module.
- AMPLIFY embeddings with downstream prediction module.
## Environment

You can set up the environment for this package using conda using the command: ```conda env create -f environment.yml```.

## Usage

1. Prepare your dataset and update the ```config.py``` file with the correct paths and parameters. Ensure you add the details of your dataset to the ```DATASETS``` dictionary.

2. Run the main script with:
    ```python src```

## Configuration

All configurations such as data paths, model parameters, and training parameters can be found and updated in the `config.py` file.

## Flags

- ```--device```                    - Can overide automatic device specification, to run package on ```CPU```, ```MPS``` or ```CUDA```.
- ```--embeddings```                - Can be used to
- ```--splits```                    - Choose how to split data; ```homologous-aware``` to ensure homologous domains are in the same split (requires domain family file), or ```random``` for entirely random assigning.
- ```--tune```                      - Choose whether to tune hyperparameters; if blank, hyperparameters in the config will be used, ```grid-search``` iterates over every possible value within config ranges, ```random-search``` searches random values within config ranges.

So for example, ```python src --splits homologous-aware --tune random-search``` would run with the automatically determined best-available device, the data would be split ensuring homologous domains are in the same splits, and the optimum hyperparameters would be searched for randomly.

## Modules

- ```__main__.py```                 - The main function which reads flags and executes functions based on user selection.
- ```config_loader.py```            - Reads the values in the config file and parses them for the rest of the package to use.
- ```datasets.py```                 - Defines the dataset class to store data in, and contains a function to set up datasets from config.
- ```helpers.py```                  - Contains small functions used by other modules.
- ```inference.py```                - Contains functions to run trained models to get predictions from datasets.
- ```models.py```                   - Contains model classes, and function to set it up from config.
- ```preprocessing.py```            - Contains functions for processing data before it is used for training or inference.
- ```embeddings.py```               - Contains functions for fetching and using protein language models to generate upstream embeddings.
- ```splits.py```                   - Contains functions to read homology files and split data based on config.
- ```training.py```                 - Contains functions to train models on datasets.
- ```tuning.py```                   - Contains functions to find optimal hyperparameters.
