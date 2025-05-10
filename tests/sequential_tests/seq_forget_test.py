import itertools
import logging
import sys

from tests.sequential_tests.base_scripts_seq import
    run_experiment_direct

# Logging setup
logging.basicConfig(
    filename="experiment_softhebb_results.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
)

# Experiment parameters
available_gpus = [1]
batch_sizes = [16]
hidden_sizes = [1024]
parameter_pairs = [(0.5, 1, 0.003)]
other_parameters = [("sanger", "sigmoid", "sigmoid", "neuron", "RELU", "neuron")]
K_values = [0.03, 1, 100, 1000]
K = K_values[0]
focuses = ["NEURON", "SYNAPSE"]
growth_parameters = ["LINEAR", "SIGMOID", "EXPONENTIAL"]

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
                        "--train_data=data/mnist/train-images.idx3-ubyte",
                        "--train_label=data/mnist/train-labels.idx1-ubyte",
                        "--test_data=data/mnist/test-images.idx3-ubyte",
                        "--test_label=data/mnist/test-labels.idx1-ubyte",
                        "--train_size=60000",
                        "--test_size=10000",
                        "--classes=10",
                        "--train_fname=data/mnist/mnist_train.csv",
                        "--test_fname=data/mnist/mnist_test.csv",
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
                        f"--device=cuda:{available_gpus[0]}",
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
