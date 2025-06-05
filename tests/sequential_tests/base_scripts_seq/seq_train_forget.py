from typing import List, Tuple

from experiments.forget_softhebb_experiment import ForgetExperiment
from experiments.forget_softhebb_experiment_iid import ForgetExperimentIID
from interfaces.experiment import Experiment
from interfaces.network import Network
from models.MLP.baseline_mlp import MLPBaseline
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
        f"-{params.experiment_name}-{params.experiment_type.lower()}-{params.lr}--",
    )
    accuracies = list(experiment.run())
    experiment.cleanup()

    test_acc = accuracies[0:5]
    train_acc = accuracies[5:10]
    return train_acc, test_acc


def run_experiment_direct_iid(
    arg_list: List[str],
) -> None:
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

    experiment: Experiment = ForgetExperimentIID(
        model,
        params,
        f"-{params.experiment_name}-{params.experiment_type.lower()}-{params.lr}--",
    )
    accuracies = list(experiment.run())
    experiment.cleanup()

    return None
