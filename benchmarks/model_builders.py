from torch import cuda, backends
from torch.nn import Module, DataParallel
from torch.optim import Optimizer, SGD
from typing import Tuple, Type, Callable

from core.arm import Arm
from core.model_builder import ModelBuilder
from benchmarks.ml_models.cudaconvnet2 import CudaConvNet2
from benchmarks.ml_models.logistic_regression import LogisticRegression


class CNNArm(Arm):

    """
    This class is used to create the type and enforce stronger type/method checking.
    Note the attributes are dynamically created by Arm.draw_hp_val()
    """

    def __init__(self):
        self.learning_rate = None
        self.n_units_1 = None
        self.n_units_2 = None
        self.n_units_3 = None
        self.weight_decay = None
        self.momentum = None
        self.batch_size = None
        self.lr_step = None
        self.gamma = None


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

    def __init__(self, arm: CNNArm, ml_model: Type[CudaConvNet2] = CudaConvNet2, optimizer: Type[SGD] = SGD,
                 in_channels: int = 3):
        super().__init__(arm, ml_model, optimizer)
        self.in_channels = in_channels
        # the below are re-set for type/attribute checking
        self.arm = arm
        self.ml_model = ml_model
        self.optimizer = optimizer

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


class LogisticRegressionArm(Arm):

    """
    This class is used to create the type and enforce stronger type checking.
    Note the attributes are dynamically created by Arm.draw_hp_val()
    """

    def __init__(self):
        self.learning_rate = None
        self.momentum = None
        self.weight_decay = None
        self.batch_size = None


class LogisticRegressionBuilder(ModelBuilder):

    def __init__(self, arm: LogisticRegressionArm, ml_model: Type[LogisticRegression] = LogisticRegression,
                 optimizer: Type[SGD] = SGD):
        super().__init__(arm, ml_model, optimizer)
        # the below are re-set for type/attribute checking
        self.arm = arm
        self.ml_model = ml_model
        self.optimizer = optimizer

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
