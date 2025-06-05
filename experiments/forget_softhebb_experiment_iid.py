# Built-in imports
import os
import shutil
import time
from typing import Tuple, Type, Union

import torch
from torch.nn import Module
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, TensorDataset

from interfaces.experiment import Experiment
from interfaces.network import Network
from layers.base.data_setup_layer import DataSetupLayer
from layers.input_layer import InputLayer
from utils.experiment_utils.experiment_constants import (DataSets,
                                                         ExperimentPhases,
                                                         Purposes)
from utils.experiment_utils.experiment_logger import *
from utils.experiment_utils.experiment_parser import *
from utils.experiment_utils.experiment_timer import *


class ForgetExperimentIID(Experiment):
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

        self.experiments_train_dataloader: DataLoader = self._setup_dataloaders(
            self.train_data_set
        )
        self.experiments_test_dataloader: DataLoader = self._setup_dataloaders(
            self.test_data_set
        )

        self.testing_test_dataloader_list: list[DataLoader] = []
        self.TOTAL_SAMPLES: int = 0
        self.curr_folder_path: str = self.RESULT_PATH

        self._setup_result_folder(self.RESULT_PATH)

        self.sub_experiment_train_timers: dict[str, float] = {}
        self.sub_experiment_test_timers: dict[str, float] = {}
        self._setup_timer_dictionaries()

        self.keep_training = True

    def _setup_dataloaders(self, input_dataset: TensorDataset) -> DataLoader:

        entire_dataloader: DataLoader = DataLoader(
            input_dataset, batch_size=self.batch_size, shuffle=True
        )
        return entire_dataloader

    def _setup_result_folder(self, result_path: str) -> None:

        try:
            shutil.rmtree(f"{self.RESULT_PATH}/Output")
            shutil.rmtree(f"{self.RESULT_PATH}/Hidden")
        except OSError as e:
            print(f"Error: {e.strerror}")

        # Create the subdirectory name
        subdirectory_name = f"{self.data_name}_{'_iid'}"
        subdirectory_path = os.path.join(result_path, subdirectory_name)

        # Create the main subdirectory
        os.makedirs(subdirectory_path, exist_ok=True)

        # Create the 'hidden' and 'output' subdirectories
        os.makedirs(os.path.join(subdirectory_path, "Hidden"), exist_ok=True)
        os.makedirs(os.path.join(subdirectory_path, "Output"), exist_ok=True)

    def _setup_timer_dictionaries(self) -> None:

        subdirectory_name = f"{self.data_name}_{'_iid'}"

        self.sub_experiment_train_timers[subdirectory_name] = 0
        self.sub_experiment_test_timers[subdirectory_name] = 0

    def _experiment(self) -> None:

        self.keep_training = True

        self.curr_folder_path: str = os.path.join(
            self.RESULT_PATH,
            f"{self.data_name}_{'_iid'}",
        )

        self.testing_test_dataloader_list.append(self.experiments_test_dataloader)

        epoch = 0
        max_epochs = 35
        self.count += 1
        while (self.keep_training) and (epoch <= max_epochs):

            self._training(
                self.experiments_train_dataloader,
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

        experiment_name = self.curr_folder_path.split("/")[-1]

        train_start: float = time.time()
        self.EXP_LOG.info(f"Started '_training' function with {dname.upper()}.")

        train_batches_per_epoch: int = len(train_data_loader)
        self.EXP_LOG.info(
            f"This training batch is epoch #{epoch} with {train_batches_per_epoch} batches of size {self.batch_size} in this epoch."
        )

        need_test: bool = True

        for inputs, labels in train_data_loader:

            if need_test:
                train_pause_time: float = time.time()
                self.sub_experiment_train_timers[experiment_name] += (
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

                train_start = time.time()

                if self.keep_training:
                    break

            inputs, labels = (
                inputs.to(self.device).float(),
                one_hot(labels, self.model.output_dim)
                .squeeze()
                .to(self.device)
                .float(),
            )

            self.model.train()
            self.model(inputs, clamped=labels)
            self.TOTAL_SAMPLES += 1
        train_end: float = time.time()
        total_added_train_time = train_end - train_start
        self.sub_experiment_train_timers[experiment_name] += total_added_train_time

        self.EXP_LOG.info(
            f"Training of epoch #{epoch} took {time_to_str(total_added_train_time)}."
        )
        self.EXP_LOG.info("Completed '_training' function for forget experiment")
        total_norm = (
            torch.nn.utils.parameters_to_vector(self.model.parameters()).norm(2).item()
        )
        self.WEIGHT_LOG.info(
            f"Model weight L2 norm after epoch #{epoch}: {total_norm:.4f}"
        )

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

        # Epoch and batch set up
        test_batches_per_epoch = len(test_data_loader)
        self.EXP_LOG.info(
            f"Experiment to be tested is iid -- Number of total experiment samples seen is {self.TOTAL_SAMPLES}"
        )
        self.EXP_LOG.info(
            f"This testing is with {test_batches_per_epoch} batches of size {self.batch_size} in this epoch."
        )

        self.model.eval()
        self.EXP_LOG.info("Set the model to testing mode.")

        final_accuracy: float = 0

        with torch.no_grad():

            correct_test_count: int = 0

            total_test_count: int = len(test_data_loader)

            for inputs, labels in test_data_loader:

                inputs, labels = inputs.to(self.device), labels.to(self.device)

                predictions: torch.Tensor = self.model(inputs)

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
        experiment_name = self.curr_folder_path.split("/")[-1]

        if purpose == Purposes.TEST_ACCURACY:
            self.sub_experiment_test_timers[experiment_name] += testing_time
            self.TEST_LOG.info(
                f"Current Experiment: iid || Total Samples Seen: {self.TOTAL_SAMPLES} || Test Accuracy: {final_accuracy}"
            )

        if purpose == Purposes.TRAIN_ACCURACY:
            self.sub_experiment_train_timers[experiment_name] += testing_time
            self.TRAIN_LOG.info(
                f"Current Experiment: {experiment_name} || Total Samples Seen: {self.TOTAL_SAMPLES} || Train Accuracy on {experiment_name}: {final_accuracy}"
            )

        self.EXP_LOG.info(
            f"Completed testing with {correct_test_count} out of {total_test_count}."
        )
        self.EXP_LOG.info("Completed '_testing' function.")
        self.EXP_LOG.info(
            f"Testing ({purpose.value.lower()} acc) in current subexperiment took {time_to_str(testing_time)}."
        )

        return final_accuracy

    def _final_test(self) -> None:
        return None
    def _final_test_log(self, results) -> None:
        return None

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
        for experiment_name, curr_timer in self.sub_experiment_test_timers.items():
            self.PARAM_LOG.info(
                f"Total test time (test acc) of {experiment_name} experiment: {time_to_str(curr_timer)}"
            )

        for experiment_name, curr_timer in self.sub_experiment_train_timers.items():
            self.PARAM_LOG.info(
                f"Total train time of {experiment_name} experiment: {time_to_str(curr_timer)}"
            )
