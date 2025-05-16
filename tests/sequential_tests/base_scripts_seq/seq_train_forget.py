import argparse
from typing import List, Tuple

from experiments.forget_softhebb_experiment import ForgetExperiment
from interfaces.experiment import Experiment
from interfaces.network import Network
from models.MLP.baseline_mlp import MLPBaseline
from utils.experiment_utils.experiment_logger import configure_logger
from utils.experiment_utils.experiment_parser import *


def run_experiment_direct(
    arg_list: List[str],
) -> Tuple[List[List[float]], List[List[float]]]:
    params = parse_arguments(arg_list)

    model: Network = MLPBaseline(
        params.K,
        params.epsilon,
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
        f"-{params.experiment_name}-{params.experiment_type.lower()}-{params.lr}--{params.heb_learn.lower()}-{params.heb_growth.lower()}-{params.heb_focus.lower()}-{params.heb_inhib.lower()}-{params.heb_lamb}---{params.class_learn.lower()}-{params.class_growth.lower()}-{params.class_focus.lower()}-0",
    )
    accuracies = list(experiment.run())
    experiment.cleanup()

    test_acc = accuracies[0:5]
    train_acc = accuracies[5:10]
    return train_acc, test_acc


if __name__ == "__main__":
    import sys

    run_experiment_direct(sys.argv[1:])
    print("Process Completed.")
