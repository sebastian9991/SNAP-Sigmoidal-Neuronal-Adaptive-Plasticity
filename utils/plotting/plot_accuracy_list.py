from typing import List

import matplotlib.pyplot as plt

from utils.path.path import get_root_dir

from typing import List

def plot_acc(accuracy_list: List[float], experiment_name: str, y_label: str) -> None:

    epochs = list(range(1, len(accuracy_list) + 1))

    plots_dir = get_root_dir() / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plt.plot(epochs, accuracy_list)
    plt.title(f"{experiment_name}")
    plt.xlabel("Epoch")
    plt.ylabel(f"{y_label}")
    plt.legend()
    plt.grid(True)

    plt.savefig(plots_dir / f"{experiment_name}.png")

    plt.close()


def average_lists(lists: List[List[float]]) -> List[float]:
    if not lists:
        return []
    if len(lists) == 1:
        return lists[0]
    
    length = len(lists[0])
    avg = [0.0] * length
    
    for lst in lists:
        if len(lst) != length:
            raise ValueError("All inner lists must have the same length.")
        for i in range(length):
            avg[i] += lst[i]
    
    return [x / len(lists) for x in avg]

