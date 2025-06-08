import itertools
import logging
import traceback

from tqdm import tqdm

from tests.sequential_tests.base_scripts_seq.seq_train_forget import *
from utils.experiment_utils.experiment_logger import *
from utils.plotting.plot_accuracy_list import *
from utils.utils_root import *
from utils.utils_root import get_project_root

# Logging setup
results_log = configure_logger(
    "Experiement Log sequential", "./log/experiment_softhebb_results.log"
)
# Root folder
project_root = get_project_root(levels_up=1)

# Experiment parameters
batch_sizes = [16]
hidden_sizes = [1024]
parameter_pairs = [(0.5, 0.0005)]
K_values = [1]
epsilons = [0.01]
focuses = ["NEURON"]
growth_parameters = ["LINEAR"]


# Sequential execution
for K in tqdm(K_values, desc="Hyperparamater: K Values."):
    for epsilon in tqdm(epsilons, desc="Hyperparamater: epsilon."):
        for focus in tqdm(focuses, desc=f"Hyperparamater: focus."):
            tqdm.write(f"{focus}-wise focus.")
            for growth in tqdm(growth_parameters, desc="Hyperparmater: Growth."):
                for batch_size, hsize in itertools.product(batch_sizes, hidden_sizes):
                    for lmbda, lr in parameter_pairs:
                        exp_name = f"focus:{focus} || K{K} || SOFTHEBB_BATCH{batch_size} || HSIZE{hsize} || {growth.upper()}"
                        args_list = [
                            "--data_name=MNIST",
                            f"--experiment_name={exp_name}",
                            f"--train_data={project_root}/data/mnist/train-images.idx3-ubyte",
                            f"--train_label={project_root}/data/mnist/train-labels.idx1-ubyte",
                            f"--test_data={project_root}/data/mnist/test-images.idx3-ubyte",
                            f"--test_label={project_root}/data/mnist/test-labels.idx1-ubyte",
                            "--train_size=60000",
                            "--test_size=10000",
                            "--classes=10",
                            f"--train_fname={project_root}/data/mnist/mnist_train.csv",
                            f"--test_fname={project_root}/data/mnist/mnist_test.csv",
                            "--input_dim=784",
                            "--output_dim=10",
                            "--heb_gam=0.99",
                            "--heb_eps=0.0001",
                            "--sub_experiment_scope_list=[[0,1],[2,3],[4,5],[6,7],[8,9]]",
                            f"--lamb={lmbda}",
                            "--heb_act=normalized",
                            "--class_learn=OUTPUT_CONTRASTIVE",
                            "--class_bias=no_bias",
                            "--class_act=normalized",
                            "--alpha=0",
                            "--beta=0.01",
                            "--sigma=1",
                            "--mu=0",
                            f"--w_lr={lr}",
                            "--l_lr=0.001",
                            "--b_lr=0.001",
                            "--init=uniform",
                            f"--hsize={hsize}",
                            f"--batch_size={batch_size}",
                            "--epochs=500",
                            f"--device={'cuda'}",
                            "--local_machine=True",
                            "--experiment_type=forget",
                            f"--K={K}",
                            f"--focus={focus}",
                            f"--weight_growth={growth}",
                            f"--epsilon={epsilon}",
                            "--seed=42",
                        ]

                        try:
                            logging.info(f"Running sequential experiment: {exp_name}")
                            results_acc, exp_name = run_experiment_direct_iid(args_list)
                            plot_acc(results_acc[1], exp_name)
                            logging.info(f"Completed: {exp_name}")
                        except Exception as e:
                            logging.error(f"Error in {exp_name}: {e}")
                            print(traceback.format_exc())
