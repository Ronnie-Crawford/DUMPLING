# Standard modules
import argparse
from pathlib import Path

# Local modules
from config_loader import config
from runner import train_and_test, train, test
from benchmarking import all_against_all_benchmarking
from visuals import plot_benchmark_grid

def main(
    run_flag: str
    ):

    """
    The entry for the package, allows choosing which pipeline to run via arguments.
    """

    match run_flag:

        case "train-inference":

            train_and_test(config)

        case "train":

            train(config)

        case "inference":

            test(config)

        case "ava-benchmarking":

            results_path = all_against_all_benchmarking(config)

            for output_feature in config["PREDICTED_FEATURES_LIST"]:

                plot_benchmark_grid(results_path, output_feature)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type = str, default = None, help = "Select whether to run [train-infernce], [train], [inference] or [ava-benchmarking].")
    args = parser.parse_args()
    main(run_flag = args.run)
