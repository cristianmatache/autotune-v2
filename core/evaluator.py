import os
from os.path import join as join_path
from abc import abstractmethod
from typing import Any, Tuple, Optional, List, Union, TypeVar
from pathlib import Path
import pandas as pd


from core.model_builder import ModelBuilder
from core.optimisation_goals import OptimisationGoals
from core.arm import Arm

TPath = TypeVar('TPath', str, Path)


def ensure_dir(path: TPath) -> TPath:
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

    __slots__ = ("model_builder", "output_dir", "file_name", "directory", "file_path", "arm", "n_resources",
                 "loss_progress_file", "loss_history")

    def __init__(self, model_builder: ModelBuilder,
                 output_dir: Optional[Union[str, Path]] = None, file_name: str = "model.pth"):
        """
        :param model_builder: builder of a machine learning model based on an arm
        :param output_dir: directory where to save the arms and their progress so far (as checkpoints)
        :param file_name: file (at output_dir/arm<i>/file_name) which stores the progress of the evaluation of an arm
        """
        self.output_dir = output_dir
        if self.output_dir is not None:
            arm_id = str(pd.Timestamp.utcnow()).replace(':', '-').replace(' ', '-').replace('.', '-').replace('+', '-')
            self.directory: Path = ensure_dir(Path(self.output_dir) / f'arm-{arm_id}')
            self.file_path: str = join_path(self.directory, file_name)
            self.loss_progress_file: str = join_path(self.directory, f"loss_progress.{file_name}")

        self.loss_history: List[float] = []
        self.arm: Arm = model_builder.arm
        self.n_resources: int = 0

    @abstractmethod
    def evaluate(self, *args: Any, **kwargs: Any) -> OptimisationGoals:
        """ Aggregates the steps:
            - train model (available through self._train)
            - evaluate model with respect to the test/validation set(s) (available through self._test)
            - report performance
        :return: optimisation goals - metrics in terms of which we perform optimization
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

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Evaluator) and self.arm == other.arm

    def __hash__(self) -> int:
        return hash(str(self))
