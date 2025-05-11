import itertools
import logging

from tests.sequential_tests.base_scripts_seq.seq_train_forget import *
from utils.experiment_utils.experiment_logger import *
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
parameter_pairs = [(0.5, 1, 0.003)]
other_parameters = [("sanger", "sigmoid", "sigmoid", "neuron", "RELU", "neuron")]
K = 1
focuses = ["NEURON"]
growth_parameters = ["LINEAR"]


# Sequential execution
for focus in focuses:
    for growth in growth_parameters:
        for batch_size, hsize in itertools.product(batch_sizes, hidden_sizes):
            for lmbda, rho, lr in parameter_pairs:
                for (
                    heb_learn,
                    heb_growth,
                    clas_growth,
                    heb_focus,
                    heb_inhib,
                    class_focus,
                ) in other_parameters:
                    exp_name = f"focus:{focus}_K{K}_SOFTHEBB_BATCH{batch_size}_HSIZE{hsize}_{growth.upper()}_{growth.upper()}"

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
                        f"--heb_dim={hsize}",
                        "--output_dim=10",
                        "--heb_gam=0.99",
                        "--heb_eps=0.0001",
                        "--sub_experiment_scope_list=[[0,1],[2,3],[4,5],[6,7],[8,9]]",
                        f"--heb_inhib={heb_inhib}",
                        f"--heb_focus={heb_focus}",
                        f"--heb_growth={heb_growth}",
                        f"--heb_learn={heb_learn}",
                        f"--heb_lamb={lmbda}",
                        f"--heb_rho={rho}",
                        "--heb_act=normalized",
                        "--class_learn=OUTPUT_CONTRASTIVE",
                        f"--class_growth={clas_growth}",
                        "--class_bias=no_bias",
                        f"--class_focus={class_focus}",
                        "--class_act=normalized",
                        f"--lr={lr}",
                        "--sigmoid_k=1",
                        "--alpha=0",
                        "--beta=0.01",
                        "--sigma=1",
                        "--mu=0",
                        "--w_lr=0.003",
                        "--l_lr=0.003",
                        "--b_lr=0.003",
                        "--init=uniform",
                        f"--hsize={hsize}",
                        f"--batch_size={batch_size}",
                        "--epochs=10",
                        f"--device={'cuda'}",
                        "--local_machine=True",
                        "--experiment_type=forget",
                        f"--K={K}",
                        f"--focus={focus}",
                        f"--weight_growth={growth}",
                    ]

                    try:
                        logging.info(f"Running sequential experiment: {exp_name}")
                        run_experiment_direct(args_list)
                        logging.info(f"Completed: {exp_name}")
                    except Exception as e:
                        logging.error(f"Error in {exp_name}: {e}")
