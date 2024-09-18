# Standard modules
import json

def read_config(file_path: str) -> dict:

    """
    Reads a JSON configuration file and returns a dictionary of configuration parameters.

    Parameters:
    - file_path (str): The path to the JSON configuration file.

    Returns:
    - config (dict): A dictionary containing the configuration parameters.
    """

    with open(file_path, 'r') as file:
        
        config = json.load(file)

    return config

# Load the configuration file once
config_path = "./config.json"
config = read_config(config_path)
