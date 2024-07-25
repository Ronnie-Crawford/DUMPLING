# Protein Domain Stability Predictor

The goal of this project is to explore how different architectures and how different datasets, and features of datasets, impact the accuracy of predicting fitness and stability of protein domains.

Currently 1 architecture has been tried:

- A simple RNN autoencoder to represent a sequence, with a downstream fitness predictor.

Best results (Pearson's rank):

- 0.0854482835354009 - RNN autoencoder and FFNN predictor

## To Do

- Rework datasets class to be a different bespoke class for each dataset
- Try a more complex autoencoder, LSTM or a simple transformer.
- Create a module to optimise hyperparameters.
- Create a module to identify secondary structures from sequences, and create a vectorised representation.
- Add a base custom dataset class in datasets.py


## Usage

1. Prepare your dataset and update the `config.py` file with the correct paths and parameters.

2. Create a custom dataset class for your data

2. Run the main script:
    ```bash
    python main.py
    ```

## Configuration

All configurations such as data paths, model parameters, and training parameters can be found and updated in the `config.py` file.

## Architecture

- Encoder - RNN - Reads a vectorised amino acid sequence, updates a 50-node hidden layer as it reads. After reading, returns the final state of the hidden layer (a representation of the sequence).

- Decoder - RNN - Uses 50-node representation of the sequence to regenerate original vectorised amino acid sequence. Also requires original length of the sequence.

- Predictor - FFNN - Uses 100-node representation (concatination of 50-node variant and 50-node wildtype representations) to predict fitness of variant. Contains 50 node hidden layer.

## Project Structure

- `main.py`: Entry point of the project. It loads calls all methods from other modules.
- `preprocessing.py`: Contains functions for encoding sequences and assigning wildtype sequences.
- `datasets.py`: Defines the `ProteinDataset` class.
- `model.py`: Defines the model architecture for the autoencoder and predictor.
- `training.py`: Contains functions for training the autoencoder and predictor.
- `inference.py`: Contains the function for predicting fitness scores.
- `metrics.py`: Contains functions for computing metrics and validating the autoencoder.
- `helpers.py`: Utility functions such as device selection and data collation.
- `splits.py`: Contains the function for splitting the dataset.
- `config.py`: Configuration file for paths and parameters.
