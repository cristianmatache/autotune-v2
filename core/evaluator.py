import os
from os.path import join as join_path
from abc import abstractmethod
import torch
from torch.nn import Module
from typing import Tuple, Any

from benchmarks.model_builders import ModelBuilder


def ensure_dir(path: str) -> str:
    if not os.path.exists(path):
        os.makedirs(path)
    return path


class Evaluator:

    """ Deals with things that happen repeatedly for every arm (draw of hyperparameter values):
    - train model
    - evaluate model with respect to the test/validation set(s)
    - report performance
    by saving and restoring checkpoints
    """

    def __init__(self, model_builder: ModelBuilder, criterion: Module = torch.nn.CrossEntropyLoss(),
                 output_dir: str = ".", file_name: str = "model.pth"):
        """
        :param model_builder: builder of a machine learning model based on an arm
        :param criterion: loss function
        :param output_dir: directory where to save the arms and their progress so far (as checkpoints)
        :param file_name: file (at output_dir/arm<i>/file_name) which stores the progress of the evaluation of an arm
        """
        self.output_dir = output_dir
        subdirs = next(os.walk(output_dir))[1]
        last_arm_number = len(subdirs)

        self.directory = ensure_dir(join_path(output_dir, f"arm{last_arm_number + 1}"))
        self.file_path = join_path(self.directory, file_name)

        self.ml_model, self.optimizer = model_builder.construct_model()
        self.arm = model_builder.arm
        self.criterion = criterion

        # save first checkpoint to file file_path
        self._save_checkpoint(epoch=0, val_error=1, test_error=1)

        self.n_resources = 0

    @abstractmethod
    def evaluate(self, *args: Any, **kwargs: Any) -> Tuple[float, float]:
        """ Aggregate the steps:
            - train model (available through self._train)
            - evaluate model with respect to the test/validation set(s) (available through self._test)
            - report performance
        :return: validation error, test error
        """
        pass

    @abstractmethod
    def _train(self, *args: Any, **kwargs: Any) -> None:
        """ Train for one epoch
        """
        pass

    @abstractmethod
    def _test(self, *args: Any, **kwargs: Any) -> float:
        """ compare the outputs of the trained model versus a benchmark dataset (i.e. validation/test sets)
        :return: test/validation error
        """
        pass

    @abstractmethod
    def _save_checkpoint(self, epoch: int, val_error: float, test_error: float) -> None:
        """ Stores the progress of the evaluation of an arm (i.e. a checkpoint)
        :param epoch: the number of the latest epoch
        :param val_error: validation error
        :param test_error: test error
        """
        pass

    def __str__(self) -> str:
        return f"Evaluator of arm:\n{self.arm}"
