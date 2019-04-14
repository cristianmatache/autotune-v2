from torch.nn import Module
from torch.optim import Optimizer, SGD
from abc import abstractmethod
from typing import Tuple, Type

from core.arm import Arm


class ModelBuilder:

    """
    Given an arm (draw of hyperparameters), a machine learning model and an optimizer
    constructs the model from the given hyperparameters
    """

    def __init__(self, arm: Arm, ml_model: Type[Module], optimizer: Type[Optimizer] = SGD):
        """
        :param arm: hyperparameters and their values
        :param ml_model:  machine learning model (Note that it is not instantiated)
        :param optimizer:  optimizing method (Note that it is not instantiated)
        """
        self.arm = arm
        self.ml_model = ml_model
        self.optimizer = optimizer

    @abstractmethod
    def construct_model(self) -> Tuple[Module, Optimizer]:
        """ Constructs the model from the actual hyperparameter values (arm)
        :return: instances of (module, optimizer)
        """
        pass
