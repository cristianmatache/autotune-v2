from abc import abstractmethod
from typing import TypeVar, Generic, Type, Optional, Tuple

from core.arm import Arm


ML_MODEL_TYPE = TypeVar('ML_MODEL_TYPE', covariant=True)
OPTIMIZER_TYPE = TypeVar('OPTIMIZER_TYPE', covariant=True)


class ModelBuilder(Generic[ML_MODEL_TYPE, OPTIMIZER_TYPE]):

    """
    Given an arm (draw of hyperparameters) and a machine learning model constructs the model
    from the given hyperparameters
    """

    def __init__(self, arm: Arm, ml_model: Type[ML_MODEL_TYPE] = None):
        """
        :param arm: hyperparameters and their values
        :param ml_model:  machine learning model (Note that it is not instantiated)
        """
        self.arm = arm
        self.ml_model = ml_model

    @abstractmethod
    def construct_model(self) -> Optional[Tuple[ML_MODEL_TYPE, OPTIMIZER_TYPE]]:
        """ Constructs the model from the actual hyperparameter values (arm)
        :return: any further information that one might want to use in the evaluator
        """
        pass
