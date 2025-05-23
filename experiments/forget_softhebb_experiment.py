# Built-in imports
import ast
import os
import shutil
import time
from argparse import Namespace
from collections import Counter
from typing import Tuple, Type, Union

import torch
from torch.nn import Module
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, TensorDataset

from interfaces.experiment import Experiment
from interfaces.network import Network
from layers.base.data_setup_layer import DataSetupLayer
from layers.input_layer import InputLayer
# Utils imports
from utils.experiment_utils.experiment_constants import (DataSets,
                                                         ExperimentPhases,
                                                         Purposes)
from utils.experiment_utils.experiment_logger import *
from utils.experiment_utils.experiment_parser import *
from utils.experiment_utils.experiment_timer import *


class ForgetExperiment(Experiment):
    """Stage 1: Experiement set-up."""

    def __init__(self, model: Network, args: argparse.Namespace, name: str) -> None:
        """
        CONTRUCTOR METHOD
        @param
            model: model to be trained and tested in experiment
            args: all arguments passed for experiment
            name: name of experiment
        @return
            None
        """
        super().__init__(model, args, name)

        dataset_mapping = {member.name.upper(): member for member in DataSets}
        self.dataset = dataset_mapping[self.data_name.upper()]

        self.train_data = args.train_data
        self.train_label = args.train_label
        self.test_data = args.test_data
        self.test_label = args.test_label
        self.train_size = args.train_size
        self.test_size = args.test_size
        self.classes = args.classes
        self.train_fname = args.train_fname
        self.test_fname = args.test_fname
        self.count = 0  # count which test we are at
        # Input layer class of model

        input_layer: Module = DataSetupLayer()
        self.input_class: Type[InputLayer] = globals()[input_layer.__class__.__name__]

        # Dataset setup
        self.train_data_set: TensorDataset = self.input_class.setup_data(
            self.train_data,
            self.train_label,
            self.train_fname,
            self.train_size,
            self.dataset,
        )
        self.test_data_set: TensorDataset = self.input_class.setup_data(
            self.test_data,
            self.test_label,
            self.test_fname,
            self.test_size,
            self.dataset,
        )

        # Subexperiment scope list set up
        # Convert the string argument to a list of lists
        print(args.sub_experiment_scope_list)
        self.sub_experiment_scope_list = ast.literal_eval(
            args.sub_experiment_scope_list
        )

        # Dataloader setup
        self.sub_experiemnts_train_dataloader_list: list[DataLoader] = (
            self._setup_dataloaders(self.train_data_set, self.sub_experiment_scope_list)
        )
        self.sub_experiemnts_test_dataloader_list: list[DataLoader] = (
            self._setup_dataloaders(self.test_data_set, self.sub_experiment_scope_list)
        )

        # Other attributes set up
        self.testing_test_dataloader_list: list[DataLoader] = []
        self.TOTAL_SAMPLES: int = 0
        self.SUB_EXP_SAMPLES: int = 0
        self.curr_folder_path: str = self.RESULT_PATH

        self._setup_result_folder(self.RESULT_PATH)

        self.test_dataloader_dictionary: dict[DataLoader, str] = (
            {}
        )  # Key is the test dataloader, value is the the sub experiment name
        self._setup_test_dataloader_dictionary()

        self.sub_experiment_train_timers: dict[str, float] = {}
        self.sub_experiment_test_timers: dict[str, float] = {}
        self._setup_timer_dictionaries()

        self.keep_training = True

    def _setup_dataloaders(
        self, input_dataset: TensorDataset, sub_experiment_scope_list: list[list[int]]
    ) -> list[DataLoader]:

        result: list[DataLoader] = []

        entire_dataloader: DataLoader = DataLoader(
            input_dataset, batch_size=self.batch_size, shuffle=True
        )

        for curr_subexperiment_labels in sub_experiment_scope_list:

            label_filter_dictionary = dict(
                zip(curr_subexperiment_labels, curr_subexperiment_labels)
            )

            curr_sub_experiment_dataloader = self.input_class.filter_data_loader(
                entire_dataloader, label_filter_dictionary
            )
            result.append(curr_sub_experiment_dataloader)

        return result

    def _setup_result_folder(self, result_path: str) -> None:

        try:
            shutil.rmtree(f"{self.RESULT_PATH}/Output")
            shutil.rmtree(f"{self.RESULT_PATH}/Hidden")
        except OSError as e:
            print(f"Error: {e.strerror}")

        for label_value_list in self.sub_experiment_scope_list:

            # Create the subdirectory name
            subdirectory_name = (
                f"{self.data_name}_{'_'.join(map(str, label_value_list))}"
            )
            subdirectory_path = os.path.join(result_path, subdirectory_name)

            # Create the main subdirectory
            os.makedirs(subdirectory_path, exist_ok=True)

            # Create the 'hidden' and 'output' subdirectories
            os.makedirs(os.path.join(subdirectory_path, "Hidden"), exist_ok=True)
            os.makedirs(os.path.join(subdirectory_path, "Output"), exist_ok=True)

    def _setup_test_dataloader_dictionary(self) -> None:

        for label_value_list, curr_test_dataloader in zip(
            self.sub_experiment_scope_list, self.sub_experiemnts_test_dataloader_list
        ):

            subdirectory_name = (
                f"{self.data_name}_{'_'.join(map(str, label_value_list))}"
            )

            self.test_dataloader_dictionary[curr_test_dataloader] = subdirectory_name

    def _setup_timer_dictionaries(self) -> None:

        for label_value_list in self.sub_experiment_scope_list:

            subdirectory_name = (
                f"{self.data_name}_{'_'.join(map(str, label_value_list))}"
            )

            self.sub_experiment_train_timers[subdirectory_name] = 0
            self.sub_experiment_test_timers[subdirectory_name] = 0

        print(self.sub_experiment_train_timers)
        print(self.sub_experiment_test_timers)

    """
    Stage 2: Training and evaluation
    """

    def _experiment(self) -> None:

        for step in range(len(self.sub_experiment_scope_list)):

            self.keep_training = True

            self.SUB_EXP_SAMPLES = 0

            curr_train_dataloader: DataLoader = (
                self.sub_experiemnts_train_dataloader_list[step]
            )
            curr_test_dataloader: DataLoader = (
                self.sub_experiemnts_test_dataloader_list[step]
            )
            self.curr_folder_path: str = os.path.join(
                self.RESULT_PATH,
                f"{self.data_name}_{'_'.join(map(str, self.sub_experiment_scope_list[step]))}",
            )

            self.testing_test_dataloader_list.append(curr_test_dataloader)

            # for epoch in range(self.epochs):
            epoch = 0
            max_epochs = 35
            self.count += 1
            while (self.keep_training) and (epoch <= max_epochs):

                self._training(
                    curr_train_dataloader,
                    epoch,
                    self.data_name,
                    ExperimentPhases.FORGET,
                )

                epoch = epoch + 1

    def _training(
        self,
        train_data_loader: DataLoader,
        epoch: int,
        dname: str,
        phase: ExperimentPhases,
        visualize: bool = False,
    ) -> None:

        sub_experiment_name = self.curr_folder_path.split("/")[
            -1
        ]  # Assumes '/' as the path separator.

        if visualize:
            self.model.visualize_weights(
                self.curr_folder_path, epoch, sub_experiment_name
            )

        # Start timer
        train_start: float = time.time()
        self.EXP_LOG.info(f"Started '_training' function with {dname.upper()}.")

        # Epoch and Batch set up
        train_batches_per_epoch: int = len(train_data_loader)
        self.EXP_LOG.info(
            f"This training batch is epoch #{epoch} with {train_batches_per_epoch} batches of size {self.batch_size} in this epoch."
        )

        need_test: bool = True

        for inputs, labels in train_data_loader:

            if need_test:
                # Pause train timer and add to total time
                train_pause_time: float = time.time()
                self.sub_experiment_train_timers[sub_experiment_name] += (
                    train_pause_time - train_start
                )

                self._testing(
                    train_data_loader,
                    Purposes.TRAIN_ACCURACY,
                    epoch,
                    self.data_name,
                    ExperimentPhases.FORGET,
                )

                for curr_test_dataloader in self.testing_test_dataloader_list:

                    self._testing(
                        curr_test_dataloader,
                        Purposes.TEST_ACCURACY,
                        epoch,
                        self.data_name,
                        ExperimentPhases.FORGET,
                    )

                need_test = False

                # Restart train timer
                train_start = time.time()

                if self.keep_training == False:
                    break

            # Move input and targets to device
            inputs, labels = (
                inputs.to(self.device).float(),
                one_hot(labels, self.model.output_dim)
                .squeeze()
                .to(self.device)
                .float(),
            )

            # Forward pass
            self.model.train()
            self.model(inputs, clamped=labels)
            # Increment samples seen
            self.TOTAL_SAMPLES += 1
            self.SUB_EXP_SAMPLES += 1

        train_end: float = time.time()
        total_added_train_time = train_end - train_start
        self.sub_experiment_train_timers[sub_experiment_name] += total_added_train_time

        self.EXP_LOG.info(
            f"Training of epoch #{epoch} took {time_to_str(total_added_train_time)}."
        )
        self.EXP_LOG.info("Completed '_training' function for forget experiment")
        for layer in self.model.modules():
            # Check if the layer is an instance of SoftHebbLayer
            if hasattr(layer, "plot_wn_distribution") and callable(
                layer.plot_wn_distribution
            ):
                layer.plot_wn_distribution(epoch, self.count)

    def _testing(
        self,
        test_data_loader: DataLoader,
        purpose: Purposes,
        epoch: int,
        dname: str,
        phase: ExperimentPhases,
        visualize: bool = False,
    ) -> Union[float, Tuple[float, ...]]:

        test_start: float = time.time()
        self.EXP_LOG.info(f"Started '_testing' function with {dname.upper()}.")

        sub_experiment_name = self.curr_folder_path.split("/")[
            -1
        ]  # Assumes '/' as the path separator.

        # Epoch and batch set up
        test_batches_per_epoch = len(test_data_loader)
        self.EXP_LOG.info(
            f"Sub-experiemnt to be tested is {sub_experiment_name} -- Number of current experiment samples seen is {self.SUB_EXP_SAMPLES} -- Number of total experiment samples seen is {self.TOTAL_SAMPLES}"
        )
        self.EXP_LOG.info(
            f"This testing is with {test_batches_per_epoch} batches of size {self.batch_size} in this epoch."
        )

        # Set the model to evaluation mode - important for layers with different training / inference behaviour
        self.model.eval()
        self.EXP_LOG.info("Set the model to testing mode.")

        final_accuracy: float = 0

        with torch.no_grad():

            correct_test_count: int = 0

            total_test_count: int = len(test_data_loader)

            for inputs, labels in test_data_loader:

                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Inference
                predictions: torch.Tensor = self.model(inputs)

                # Evaluates performance of model on testing dataset
                correct_test_count += (
                    (predictions.argmax(-1) == labels).type(torch.float).sum()
                )
                total_test_count += labels.size(0)

            final_accuracy = correct_test_count / total_test_count

            if (final_accuracy >= 0.85) and (purpose == Purposes.TRAIN_ACCURACY):
                self.keep_training = False

        test_end = time.time()
        testing_time = test_end - test_start

        self.DEBUG_LOG.info(f"Test start for this: {test_start}")
        self.DEBUG_LOG.info(f"Test end for this: {test_end}")
        self.DEBUG_LOG.info(f"Test duration: {testing_time}")

        if purpose == Purposes.TEST_ACCURACY:
            testing_subexperiment_name = self.test_dataloader_dictionary[
                test_data_loader
            ]
            self.sub_experiment_test_timers[testing_subexperiment_name] += testing_time
            self.TEST_LOG.info(
                f"Current Experiment: {sub_experiment_name} || Current Subexperiment Samples Seen: {self.SUB_EXP_SAMPLES} || Total Samples Seen: {self.TOTAL_SAMPLES} || Test Accuracy on {testing_subexperiment_name}: {final_accuracy}"
            )

        if purpose == Purposes.TRAIN_ACCURACY:
            self.sub_experiment_train_timers[sub_experiment_name] += testing_time
            self.TRAIN_LOG.info(
                f"Current Experiment: {sub_experiment_name} || Current Subexperiment Samples Seen: {self.SUB_EXP_SAMPLES} || Total Samples Seen: {self.TOTAL_SAMPLES} || Train Accuracy on {sub_experiment_name}: {final_accuracy}"
            )

        self.EXP_LOG.info(
            f"Completed testing with {correct_test_count} out of {total_test_count}."
        )
        self.EXP_LOG.info("Completed '_testing' function.")
        self.EXP_LOG.info(
            f"Testing ({purpose.value.lower()} acc) of sample #{self.SUB_EXP_SAMPLES} in current subexperiment took {time_to_str(testing_time)}."
        )

        if visualize:
            self.model.visualize_weights(
                self.curr_folder_path, self.SUB_EXP_SAMPLES, purpose.name.lower()
            )

        return final_accuracy

    """
    Stage 3: Final testing and report. 
    """

    def _final_test(self):

        list_of_train_accuracy: list = []
        list_of_test_accuracy: list = []

        for step in range(len(self.sub_experiment_scope_list)):

            curr_train_dataloader: DataLoader = (
                self.sub_experiemnts_train_dataloader_list[step]
            )
            curr_test_dataloader: DataLoader = (
                self.sub_experiemnts_test_dataloader_list[step]
            )
            self.curr_folder_path: str = os.path.join(
                self.RESULT_PATH,
                f"{self.data_name}_{'_'.join(map(str, self.sub_experiment_scope_list[step]))}",
            )

            temp_test_acc: Union[float, Tuple[float, ...]] = self._testing(
                curr_test_dataloader,
                Purposes.TEST_ACCURACY,
                0,
                self.data_name,
                ExperimentPhases.FORGET,
                visualize=False,
            )
            temp_train_acc: Union[float, Tuple[float, ...]] = self._testing(
                curr_train_dataloader,
                Purposes.TRAIN_ACCURACY,
                0,
                self.data_name,
                ExperimentPhases.FORGET,
                visualize=False,
            )

            list_of_test_accuracy.append(temp_test_acc)
            list_of_train_accuracy.append(temp_train_acc)

        full_list_of_accuracy = list_of_test_accuracy + list_of_train_accuracy
        return full_list_of_accuracy

    def _param_start_log(self):
        self.EXP_LOG.info("Started logging of experiment parameters.")

    def _param_end_log(self):
        total_train_time = sum(self.sub_experiment_train_timers.values())
        total_test_time = sum(self.sub_experiment_test_timers.values())

        self.PARAM_LOG.info(
            f"End time of experiment: {time.strftime('%Y-%m-%d %Hh:%Mm:%Ss', time.localtime(self.END_TIME))}"
        )
        self.PARAM_LOG.info(
            f"Runtime of experiment: {time_to_str(self.DURATION if self.DURATION is not None else 0)}"
        )
        self.PARAM_LOG.info(
            f"Total train time of experiment: {time_to_str(total_train_time)}"
        )
        self.PARAM_LOG.info(
            f"Total test time of experiment: {time_to_str(total_test_time)}"
        )
        for sub_experiment_name, curr_timer in self.sub_experiment_test_timers.items():
            self.PARAM_LOG.info(
                f"Total test time (test acc) of {sub_experiment_name} experiment: {time_to_str(curr_timer)}"
            )

        for sub_experiment_name, curr_timer in self.sub_experiment_train_timers.items():
            self.PARAM_LOG.info(
                f"Total train time of {sub_experiment_name} experiment: {time_to_str(curr_timer)}"
            )

    def _final_test_log(self, results) -> None:

        test_acc_digit_0_1 = results[0]
        test_acc_digit_2_3 = results[1]
        test_acc_digit_4_5 = results[2]
        test_acc_digit_6_7 = results[3]
        test_acc_digit_8_9 = results[4]
        train_acc_digit_0_1 = results[5]
        train_acc_digit_2_3 = results[6]
        train_acc_digit_4_5 = results[7]
        train_acc_digit_6_7 = results[8]
        train_acc_digit_8_9 = results[9]

        self.PARAM_LOG.info(
            f"Testing accuracy of model on digits 0 and 1 after training for {self.epochs} epochs: {test_acc_digit_0_1}"
        )
        self.PARAM_LOG.info(
            f"Testing accuracy of model on digits 2 and 3 after training for {self.epochs} epochs: {test_acc_digit_2_3}"
        )
        self.PARAM_LOG.info(
            f"Testing accuracy of model on digits 4 and 5 after training for {self.epochs} epochs: {test_acc_digit_4_5}"
        )
        self.PARAM_LOG.info(
            f"Testing accuracy of model on digits 6 and 7 after training for {self.epochs} epochs: {test_acc_digit_6_7}"
        )
        self.PARAM_LOG.info(
            f"Testing accuracy of model on digits 8 and 9 after training for {self.epochs} epochs: {test_acc_digit_8_9}"
        )
        self.PARAM_LOG.info(
            f"Training accuracy of model on digits 0 and 1 after training for {self.epochs} epochs: {train_acc_digit_0_1}"
        )
        self.PARAM_LOG.info(
            f"Training accuracy of model on digits 2 and 3 after training for {self.epochs} epochs: {train_acc_digit_2_3}"
        )
        self.PARAM_LOG.info(
            f"Training accuracy of model on digits 4 and 5 after training for {self.epochs} epochs: {train_acc_digit_4_5}"
        )
        self.PARAM_LOG.info(
            f"Training accuracy of model on digits 6 and 7 after training for {self.epochs} epochs: {train_acc_digit_6_7}"
        )
        self.PARAM_LOG.info(
            f"Training accuracy of model on digits 8 and 8 after training for {self.epochs} epochs: {train_acc_digit_8_9}"
        )
