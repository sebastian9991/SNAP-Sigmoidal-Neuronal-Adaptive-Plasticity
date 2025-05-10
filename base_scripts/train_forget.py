import multiprocessing
from typing import List, Tuple

import torch
from experiments.forget_softhebb_experiment import ForgetExperiment
from interfaces.experiment import Experiment
from interfaces.network import Network
from models.hebbian_network import HebbianNetwork
from utils.experiment_logger import *
from utils.experiment_parser import *
from utils.experiment_stats import *

from models.MLP.models import NewMLPBaseline_Model

# Create log
results_log = configure_logger("Forget Result Log", "./results/results_forget.log")

# Get arguments
ARGS = parse_arguments()


def main():
    # Parallel training
    train_acc_lists, test_acc_lists = parallel_training(ARGS, 1)


# Model Training
def train_and_eval(args: Tuple) -> List[List[float]]:
    params: argparse.Namespace

    num: int
    params, num = args
    model: Network = NewMLPBaseline_Model(
        params.K,
        params.focus,
        params.hsize,
        params.lamb,
        params.w_lr,
        params.b_lr,
        params.l_lr,
        params.nclasses,
        params.device,
        params.weight_growth,
    )
    experiment: Experiment = ForgetExperiment(
        model,
        params,
        f"-{params.experiment_name}-{params.experiment_type.lower()}-{params.lr}--{params.heb_learn.lower()}-{params.heb_growth.lower()}-{params.heb_focus.lower()}-{params.heb_inhib.lower()}-{params.heb_lamb}---{params.class_learn.lower()}-{params.class_growth.lower()}-{params.class_focus.lower()}-{num}",
    )
    accuracies = list(experiment.run())
    experiment.cleanup()

    return [
        accuracies[0:5],  # Test accuracies for digit pairs 0-1, 2-3, 4-5, 6-7, 8-9
        accuracies[5:10],  # Train accuracies for digit pairs 0-1, 2-3, 4-5, 6-7, 8-9
    ]


# Parallel Training
def parallel_training(
    params: argparse.Namespace, total: int
) -> Tuple[List[List[float]], List[List[float]]]:
    # Create a pool of worker processes
    with multiprocessing.Pool(processes=total) as pool:
        # Map the list of parameters to the function that performs training
        param_list = [(params, process_id) for process_id in range(total)]
        results = pool.map(train_and_eval, param_list)

    # Split results into train and test accuracy lists for each digit pair
    test_acc_lists = [[] for _ in range(5)]
    train_acc_lists = [[] for _ in range(5)]

    for result in results:
        test_acc, train_acc = result
        for i in range(5):
            test_acc_lists[i].append(test_acc[i])
            train_acc_lists[i].append(train_acc[i])

    return train_acc_lists, test_acc_lists


if __name__ == "__main__":
    main()
    print("Process Completed.")
