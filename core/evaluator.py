import os
from os.path import join as join_path
from abc import abstractmethod
from typing import Any, Tuple


from core.model_builder import ModelBuilder
from core.optimization_goals import OptimizationGoals


def ensure_dir(path: str) -> str:
    """ If the directory at given path doesn't exist, it will create it
    :param path: path to directory
    :return: path to directory
    """
    if not os.path.exists(path):
        os.makedirs(path)
    return path


class Evaluator:

    """ Deals with things that happen repeatedly for an arm (draw of hyperparameter values):
    - train model
    - evaluate model with respect to the test/validation set(s)
    - report performance
    by saving and restoring checkpoints
    """

    __slots__ = ("model_builder", "output_dir", "file_name", "directory", "file_path", "arm", "n_resources")

    def __init__(self, model_builder: ModelBuilder, output_dir: str = ".", file_name: str = "model.pth"):
        """
        :param model_builder: builder of a machine learning model based on an arm
        :param output_dir: directory where to save the arms and their progress so far (as checkpoints)
        :param file_name: file (at output_dir/arm<i>/file_name) which stores the progress of the evaluation of an arm
        """
        self.output_dir = output_dir
        subdirectories = next(os.walk(output_dir))[1]
        last_arm_number = len(subdirectories)
        self.directory = ensure_dir(join_path(output_dir, f"arm{last_arm_number + 1}"))
        self.file_path = join_path(self.directory, file_name)

        self.arm = model_builder.arm
        self.n_resources = 0

    @abstractmethod
    def evaluate(self, *args: Any, **kwargs: Any) -> OptimizationGoals:
        """ Aggregates the steps:
            - train model (available through self._train)
            - evaluate model with respect to the test/validation set(s) (available through self._test)
            - report performance
        :return: optimization goals - metrics in terms of which we perform optimization
                 Eg. validation error, test error
        """
        pass

    @abstractmethod
    def _train(self, *args: Any, **kwargs: Any) -> None:
        """ Train for one epoch
        """
        pass

    @abstractmethod
    def _test(self, *args: Any, **kwargs: Any) -> Tuple[float, ...]:
        """ Compare the outputs of the trained model versus a benchmark dataset (i.e. validation/test sets)
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
        """
        :return: human readable representation of an (arm) evaluator
        """
        return f"Evaluator of arm:\n{self.arm}"
