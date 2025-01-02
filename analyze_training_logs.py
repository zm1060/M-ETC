import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def parse_logs(file_path):
    """
    Parses a log file to extract training and validation metrics for multiple models and datasets.

    Args:
        file_path (str): Path to the log file.

    Returns:
        dict: Nested dictionary with model names, dataset names, and their corresponding metrics.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Log file not found: {file_path}")

    results = {}
    current_model = None
    current_dataset = None

    # Regular expressions for parsing
    model_pattern = re.compile(r"Logger initialized for model: (\S+)")
    dataset_pattern = re.compile(r"Training data directory: .*/([A-Za-z0-9\-_]+)$")  # Extract dataset name
    metrics_pattern = re.compile(
        r"Epoch (\d+)/\d+, (Train|Validation) Metrics: \{.*?'accuracy': ([\d.]+),.*?'precision': ([\d.]+),"
        r".*?'recall': ([\d.]+),.*?'f1': ([\d.]+),.*?'loss': ([\d.]+)"
    )

    with open(file_path, 'r') as log_file:
        for line in log_file:
            model_match = model_pattern.search(line)
            dataset_match = dataset_pattern.search(line)
            metrics_match = metrics_pattern.search(line)

            # Capture the model name
            if model_match:
                current_model = model_match.group(1)
                if current_model not in results:
                    results[current_model] = {}

            # Capture the dataset name
            if dataset_match:
                current_dataset = dataset_match.group(1)
                if current_model and current_dataset not in results[current_model]:
                    results[current_model][current_dataset] = {
                        "train": [], "validation": []
                    }

            # Capture metrics
            if metrics_match:
                epoch = int(metrics_match.group(1))
                phase = metrics_match.group(2).lower()
                accuracy = float(metrics_match.group(3))
                precision = float(metrics_match.group(4))
                recall = float(metrics_match.group(5))
                f1 = float(metrics_match.group(6))
                loss = float(metrics_match.group(7))

                if current_model and current_dataset:
                    results[current_model][current_dataset][phase].append({
                        "epoch": epoch,
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "loss": loss,
                    })
    return results


def plot_metrics_combined(data, model, dataset, save_plots=False):
    """
    Generates advanced combined plots for training/validation metrics.

    Args:
        data (dict): Parsed log data.
        model (str): Model name.
        dataset (str): Dataset name.
        save_plots (bool): Whether to save the plots as PNG files.
    """
    sns.set(style="whitegrid", context="paper")

    phases = ["train", "validation"]
    for phase in phases:
        if phase not in data[model][dataset]:
            continue

        df = pd.DataFrame(data[model][dataset][phase])
        if not df.empty:
            metrics = ["accuracy", "precision", "recall", "f1"]

            # Plot each metric combined with loss
            for metric in metrics:
                fig, ax1 = plt.subplots(figsize=(10, 6))

                # Plot the main metric on the left y-axis
                color = "tab:blue"
                ax1.set_xlabel("Epoch")
                ax1.set_ylabel(metric.title(), color=color)
                ax1.plot(df["epoch"], df[metric], label=f"{phase.capitalize()} {metric.title()}", marker="o", color=color)
                ax1.tick_params(axis='y', labelcolor=color)

                # Annotate the best metric performance
                best_epoch = df[metric].idxmax() if metric != "loss" else df[metric].idxmin()
                best_value = df.loc[best_epoch, metric]
                best_epoch_num = df.loc[best_epoch, "epoch"]

                ax1.annotate(
                    f"Best: {best_value:.4f}\nEpoch: {best_epoch_num}",
                    (best_epoch_num, best_value),
                    textcoords="offset points",
                    xytext=(-20, 10),
                    ha="center",
                    fontsize="small",
                    color=color,
                    arrowprops=dict(arrowstyle="->", color=color)
                )

                # Plot the loss on the right y-axis
                ax2 = ax1.twinx()
                color = "tab:red"
                ax2.set_ylabel("Loss", color=color)
                ax2.plot(df["epoch"], df["loss"], label=f"{phase.capitalize()} Loss", linestyle="--", marker="x", color=color)
                ax2.tick_params(axis='y', labelcolor=color)

                # Annotate the lowest loss
                best_loss_epoch = df["loss"].idxmin()
                best_loss_value = df.loc[best_loss_epoch, "loss"]
                best_loss_epoch_num = df.loc[best_loss_epoch, "epoch"]

                ax2.annotate(
                    f"Best Loss: {best_loss_value:.4f}\nEpoch: {best_loss_epoch_num}",
                    (best_loss_epoch_num, best_loss_value),
                    textcoords="offset points",
                    xytext=(20, -10),
                    ha="center",
                    fontsize="small",
                    color=color,
                    arrowprops=dict(arrowstyle="->", color=color)
                )

                # Title and legends
                plt.title(f"{metric.title()} and Loss over Epochs\nModel: {model}, Dataset: {dataset} ({phase.capitalize()})")
                fig.tight_layout()
                plt.grid(True)

                # Save plots if required
                if save_plots:
                    plot_filename = f"{model}_{dataset}_{phase}_{metric}_combined.png"
                    plt.savefig(plot_filename, dpi=300)
                    print(f"Saved plot: {plot_filename}")

                plt.close()



# Path to the log file
log_file_path = './logs/CNN_BiGRU_Attention.log'  # Update this path if necessary

try:
    # Parse the logs
    parsed_logs = parse_logs(log_file_path)

    # Visualize metrics for all models and datasets
    for model in parsed_logs:
        for dataset in parsed_logs[model]:
            print(f"Plotting metrics for model '{model}' on dataset '{dataset}'")
            plot_metrics_combined(parsed_logs, model, dataset, save_plots=True)
except FileNotFoundError as e:
    print(e)
