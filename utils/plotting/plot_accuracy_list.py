from typing import List

import matplotlib.pyplot as plt

from utils.path.path import get_root_dir


def plot_acc(accuracy_list: List[float]) -> None:

    epochs = list(range(1, len(accuracy_list) + 1))

    plots_dir = get_root_dir() / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plt.plot(epochs, accuracy_list, marker="o")
    plt.title("Testing Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)

    plt.savefig(plots_dir / "testing_accuracy_plot.png")

    plt.show()
