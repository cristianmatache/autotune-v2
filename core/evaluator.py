import os
from os.path import join as join_path
from abc import abstractmethod
import torch
from torch.nn import Module
from typing import Tuple

from benchmarks.model_builders import ModelBuilder


def ensure_dir(path: str) -> str:
    if not os.path.exists(path):
        os.makedirs(path)
    return path


class Evaluator:

    def __init__(self, model_builder: ModelBuilder, criterion: Module = torch.nn.CrossEntropyLoss(),
                 output_dir: str = ".", file_name: str = "model.pth"):
        """
        :param output_dir: directory where to save the arms and their progress so far
        :param file_name: file (at output_dir/arm<i>/file_name) which records the progress of the evaluation of an arm
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
        self.save_checkpoint(epoch=0, val_error=1, test_error=1)

        self.n_resources = 0

    @abstractmethod
    def save_checkpoint(self, epoch: int, val_error: float, test_error: float) -> None:
        pass

    @abstractmethod
    def _train(self, *args, **kwargs) -> float:
        pass

    @abstractmethod
    def _test(self, *args, **kwargs) -> float:
        pass

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> Tuple[float, float]:
        pass
