import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd

from utils.utils_root import get_project_root


# Function to read and parse log file
def parse_log_file(log_path):
    with open(log_path, "r") as file:
        data = file.read()

    pattern = re.compile(
        r"(?P<timestamp>[\d-]+\s[\d:,]+)\s\|\|\sINFO\s+\|\|\s\d+\s+\|\| _testing\s+\|\| "
        r"Current Experiment: (?P<experiment>.*?)\s+\|\| Current Subexperiment Samples Seen: (?P<subexp_samples>\d+)"
        r"\s+\|\| Total Samples Seen: (?P<total_samples>\d+)\s+\|\| Test Accuracy on (?P<accuracy_experiment>.*?): (?P<accuracy>[\d\.]+)"
    )

    matches = pattern.finditer(data)

    rows = []
    for match in matches:
        rows.append(match.groupdict())

    df = pd.DataFrame(rows)
    df["subexp_samples"] = df["subexp_samples"].astype(int)
    df["total_samples"] = df["total_samples"].astype(int)
    df["accuracy"] = df["accuracy"].astype(float)
    return df


# Function to plot accuracy trends
def plot_accuracy_trends(df, title):
    mnist_experiments = [
        "MNIST_0_1",
        "MNIST_2_3",
        "MNIST_4_5",
        "MNIST_6_7",
        "MNIST_8_9",
    ]

    plt.figure(figsize=(10, 6))
    for exp in mnist_experiments:
        sub_df = df[df["accuracy_experiment"] == exp]
        total_samples = sub_df["total_samples"].to_numpy()
        accuracy = sub_df["accuracy"].to_numpy()
        plt.plot(total_samples, accuracy, label=exp)

    plt.xlabel("Total Samples Seen")
    plt.ylabel("Test Accuracy")
    plt.title(f"Accuracy Trends for {title}")
    plt.legend()
    plt.grid()
    plt.show()


def plot_logger(filename: str, Training: bool) -> None:

    root = get_project_root(1)

    log_file_path = f"{root}/results/{filename}/{'training_accuracy' if Training else 'testing_accuracy'}.log"

    data = defaultdict(lambda: {"steps": [], "accuracy": []})

    log_line_pattern = re.compile(
        r"current subexperiment samples seen: (\d+) .*?test accuracy on (mnist_\d+_\d+): ([0-9.]+)"
    )

    # parse the log file
    with open(log_file_path, "r") as f:
        for line in f:
            match = log_line_pattern.search(line)
            if match:
                step = int(match.group(1))
                experiment = match.group(2)
                accuracy = float(match.group(3))
                data[experiment]["steps"].append(step)
                data[experiment]["accuracy"].append(accuracy)

    # plot each subexperiment
    for experiment, values in data.items():
        plt.figure()
        plt.plot(values["steps"], values["accuracy"], marker="o")
        plt.title(f"accuracy over time - {experiment}")
        plt.xlabel("samples seen")
        plt.ylabel("accuracy")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def main():
    root = get_project_root(1)
    results_folder = f"{root}/results/"
    experiments = [
        d
        for d in os.listdir(results_folder)
        if os.path.isdir(os.path.join(results_folder, d))
    ]

    for experiment in experiments:
        log_path = os.path.join(results_folder, experiment, "testing_accuracy.log")
        if os.path.exists(log_path):
            df = parse_log_file(log_path)
            plot_accuracy_trends(df, experiment)
        else:
            print("file does not exist.")


if __name__ == "__main__":
    main()
