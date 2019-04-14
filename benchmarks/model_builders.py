from torch import cuda, backends
from torch.nn import Module, DataParallel
from torch.optim import Optimizer, SGD
from typing import Tuple, Type

from core.arm import Arm
from core.model_builder import ModelBuilder
from benchmarks.ml_models.cudaconvnet2 import CudaConvNet2
from benchmarks.ml_models.logistic_regression import LogisticRegression


class CNNArm(Arm):

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


class CNNBuilder(ModelBuilder):

    def __init__(self, arm: CNNArm, ml_model: Type[CudaConvNet2] = CudaConvNet2, optimizer: Type[SGD] = SGD):
        super().__init__(arm, ml_model, optimizer)
        self.arm = arm
        self.ml_model = ml_model
        self.optimizer = optimizer

    def construct_model(self) -> Tuple[Module, Optimizer]:
        # Construct model and optimizer based on hyperparameters
        arm = self.arm
        base_lr = arm.learning_rate

        model = self.ml_model(3, int(arm.n_units_1), int(arm.n_units_2), int(arm.n_units_3))  # n_channels = 3

        if cuda.is_available():
            model.cuda()
            model = DataParallel(model, device_ids=range(cuda.device_count()))
            backends.cudnn.benchmark = True

        optimizer = self.optimizer(model.parameters(), lr=base_lr, momentum=arm.momentum, weight_decay=arm.weight_decay)
        return model, optimizer


class LogisticRegressionArm(Arm):

    def __init__(self):
        self.learning_rate = None
        self.momentum = None
        self.weight_decay = None
        self.batch_size = None


class LogisticRegressionBuilder(ModelBuilder):

    def __init__(self, arm: LogisticRegressionArm, ml_model: Type[LogisticRegression] = LogisticRegression,
                 optimizer: Type[SGD] = SGD):
        super().__init__(arm, ml_model, optimizer)
        self.arm = arm
        self.ml_model = ml_model
        self.optimizer = optimizer

    def construct_model(self) -> Tuple[Module, Optimizer]:
        # Construct model and optimizer based on hyperparameters
        arm = self.arm
        base_lr = arm.learning_rate

        model = self.ml_model(input_size=784, num_classes=10)

        if cuda.is_available():
            model.cuda()
            model = DataParallel(model, device_ids=range(cuda.device_count()))
            backends.cudnn.benchmark = True

        optimizer = self.optimizer(model.parameters(), lr=base_lr, momentum=arm.momentum, weight_decay=arm.weight_decay)
        return model, optimizer
