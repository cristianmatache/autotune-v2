from torch import cuda, backends
from torch.nn import Module, DataParallel
from torch.optim import Optimizer, SGD
from typing import Tuple, Type, Callable

from core.arm import Arm
from core.model_builder import ModelBuilder
from benchmarks.ml_models.cudaconvnet2 import CudaConvNet2
from benchmarks.ml_models.logistic_regression import LogisticRegression


def update_model_for_gpu(construct_model_function: Callable) -> Callable:
    """ decorator to update the model under construction to work with GPUs
    """
    def wrapper(self: ModelBuilder) -> Tuple[Module, Optimizer]:
        model, optimizer = construct_model_function(self)
        if cuda.is_available():
            model.cuda()
            model = DataParallel(model, device_ids=range(cuda.device_count()))
            backends.cudnn.benchmark = True
        return model, optimizer
    return wrapper


class CNNBuilder(ModelBuilder):

    def __init__(self, arm: Arm, ml_model: Type[CudaConvNet2] = CudaConvNet2, optimizer: Type[SGD] = SGD,
                 in_channels: int = 3):
        super().__init__(arm, ml_model, optimizer)
        self.in_channels = in_channels

    @update_model_for_gpu
    def construct_model(self) -> Tuple[Module, Optimizer]:
        """ Construct model and optimizer based on hyperparameters
        :return: instances of each (model, optimizer) using the hyperparameters as specified by the arm
        """
        arm = self.arm
        base_lr = arm.learning_rate

        model = self.ml_model(self.in_channels, int(arm.n_units_1), int(arm.n_units_2), int(arm.n_units_3))
        optimizer = self.optimizer(model.parameters(), lr=base_lr, momentum=arm.momentum, weight_decay=arm.weight_decay)
        return model, optimizer


class LogisticRegressionBuilder(ModelBuilder):

    def __init__(self, arm: Arm, ml_model: Type[LogisticRegression] = LogisticRegression,
                 optimizer: Type[SGD] = SGD):
        super().__init__(arm, ml_model, optimizer)

    @update_model_for_gpu
    def construct_model(self) -> Tuple[Module, Optimizer]:
        """ Construct model and optimizer based on hyperparameters
        :return: instances of each (model, optimizer) using the hyperparameters as specified by the arm
        """
        arm = self.arm
        base_lr = arm.learning_rate

        model = self.ml_model(input_size=784, num_classes=10)
        optimizer = self.optimizer(model.parameters(), lr=base_lr, momentum=arm.momentum, weight_decay=arm.weight_decay)
        return model, optimizer
