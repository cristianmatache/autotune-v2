from torch import cuda, backends
from torch.nn import Module, DataParallel
from torch.optim import Optimizer, SGD
from typing import Tuple, Type, Callable
from abc import abstractmethod

from core import Arm, ModelBuilder

from benchmarks.ml_models.cudaconvnet2 import CudaConvNet2
from benchmarks.ml_models.logistic_regression import LogisticRegression

CONSTRUCT_MODEL_FUNCTION_TYPE = Callable[[ModelBuilder], Tuple[Module, Optimizer]]


class TorchModelBuilder(ModelBuilder[Module, Optimizer]):

    """
    Given an arm (draw of hyperparameters) and a machine learning model
    constructs a pytorch model from the given hyperparameters (arm)
    """

    def __init__(self, arm: Arm, ml_model: Type[Module]):
        """
        :param arm: hyperparameters and their values
        :param ml_model:  machine learning model (Note that it is not instantiated)
        """
        super().__init__(arm=arm, ml_model=ml_model)

    @abstractmethod
    def construct_model(self) -> Tuple[Module, Optimizer]:
        """ Constructs the model from the actual hyperparameter values (arm)
        :return: instances of (module, optimiser)
        """
        pass


def _update_model_for_gpu(construct_model_function: CONSTRUCT_MODEL_FUNCTION_TYPE) \
        -> CONSTRUCT_MODEL_FUNCTION_TYPE:
    """ decorator to update the model under construction and allow it to work with GPUs
    """
    def wrapper(self: ModelBuilder) -> Tuple[Module, Optimizer]:
        model, optimiser = construct_model_function(self)
        if cuda.is_available():
            model.cuda()
            model = DataParallel(model, device_ids=range(cuda.device_count()))
            backends.cudnn.benchmark = True
        return model, optimiser
    return wrapper


class CNNBuilder(TorchModelBuilder):

    """ Torch CNN model
    """

    __slots__ = ("in_channels",)

    def __init__(self, arm: Arm, in_channels: int = 3):
        """
        :param arm: hyperparameters and their values
        :param in_channels: number of input channels in ml_model CudaConvNet2
        """
        super().__init__(arm=arm, ml_model=CudaConvNet2)
        self.in_channels = in_channels

    @_update_model_for_gpu
    def construct_model(self) -> Tuple[Module, Optimizer]:
        """ Construct model and optimiser based on hyperparameters
        :return: instances of each (model, optimiser) using the hyperparameters as specified by the arm
        """
        arm = self.arm
        base_lr = arm.learning_rate

        model = self.ml_model(self.in_channels, int(arm.n_units_1), int(arm.n_units_2), int(arm.n_units_3))
        optimiser = SGD(model.parameters(), lr=base_lr, momentum=arm.momentum, weight_decay=arm.weight_decay)
        return model, optimiser


class LogisticRegressionBuilder(TorchModelBuilder):

    def __init__(self, arm: Arm):
        """
        :param arm: hyperparameters and their values
        """
        super().__init__(arm=arm, ml_model=LogisticRegression)

    @_update_model_for_gpu
    def construct_model(self) -> Tuple[Module, Optimizer]:
        """ Construct model and optimiser based on hyperparameters
        :return: instances of each (model, optimiser) using the hyperparameters as specified by the arm
        """
        arm = self.arm
        base_lr = arm.learning_rate

        model = self.ml_model(input_size=784, num_classes=10)
        optimiser = SGD(model.parameters(), lr=base_lr, momentum=arm.momentum, weight_decay=arm.weight_decay)
        return model, optimiser
