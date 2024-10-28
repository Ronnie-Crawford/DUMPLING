# Third party modules
import pandas as pd
import matplotlib.pyplot as plt

def plot_predictions_vs_true(predictions_df: pd.DataFrame):

    truth_column = "True Fitness"
    predicted_column = "Predicted Fitness"

    true_values = predictions_df[truth_column]
    predicted_values = predictions_df[predicted_column]

    plt.scatter(true_values, predicted_values, color = "blue", label = "Predicted vs True", s = 0.1, alpha = 0.8)
    min_val = min(true_values.min(), predicted_values.min())
    max_val = max(true_values.max(), predicted_values.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--")
    plt.xlabel("True Fitness")
    plt.ylabel("Predicted Fitness")
    plt.title("Predicted vs True Fitness Values")
    plt.legend()

    plt.savefig("./results/figures/accuracy_scatter.png")
    plt.close()

def plot_input_histogram(predictions_df: pd.DataFrame):

    truth_column = "True Fitness"
    true_values = predictions_df[truth_column]

    plt.hist(true_values, bins = 100)
    plt.title("Histogram of True Fitness")

    plt.savefig("./results/figures/input_histogram.png")
    plt.close()
