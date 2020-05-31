from abc import abstractmethod
from typing import TypeVar, Generic, Type, Optional, Tuple

from autotune.core.arm import Arm

TModelType = TypeVar('TModelType', covariant=True)
TOptimizerType = TypeVar('TOptimizerType', covariant=True)


class ModelBuilder(Generic[TModelType, TOptimizerType]):
    """
    Given an arm (draw of hyperparameters) and a machine learning model constructs the model
    from the given hyperparameters
    """

    __slots__ = 'arm', 'ml_model'

    def __init__(self, arm: Arm, ml_model: Optional[Type[TModelType]] = None):
        """
        :param arm: hyperparameters and their values
        :param ml_model:  machine learning model (Note that it is not instantiated)
        """
        self.arm = arm
        self.ml_model: Optional[Type[TModelType]] = ml_model

    @abstractmethod
    def construct_model(self) -> Optional[Tuple[TModelType, TOptimizerType]]:
        """ Constructs the model from the actual hyperparameter values (arm)
        :return: any further information that one might want to use in the evaluator
        """
