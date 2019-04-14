from torch.nn import Module
from torch.optim import Optimizer, SGD
from abc import abstractmethod
from typing import Tuple, Type

from core.arm import Arm


class ModelBuilder:

    def __init__(self, arm: Arm, ml_model: Type[Module], optimizer: Type[Optimizer] = SGD):
        self.arm = arm
        self.ml_model = ml_model
        self.optimizer = optimizer

    @abstractmethod
    def construct_model(self) -> Tuple[Module, Optimizer]:
        pass
