# config.py

# Paths to the data
DATA_PATH = '/path/to/data.csv'

# Model parameters
INPUT_SIZE = len('ACDEFGHIKLMNPQRSTVWY') + 1  # Number of unique amino acids + 1 for padding
HIDDEN_SIZE = 50
OUTPUT_SIZE = 1
NUM_LAYERS = 1

# Training parameters
TRAIN_SIZE = 0.75
VAL_SIZE = 0.10
TEST_SIZE = 0.15
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10
RANDOM_STATE = 42
