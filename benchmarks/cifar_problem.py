import numpy as np
from typing import Tuple, Dict, Type

from core.arm import Arm
from core.params import Param
from core.problem_def import HyperparameterOptimizationProblem
from core.optimization_goals import OptimizationGoals
from datasets.image_dataset_loaders import CIFARLoader, ImageDatasetLoader
from benchmarks.torch_evaluator import TorchEvaluator
from benchmarks.torch_model_builders import CNNBuilder
from util.io import print_evaluation


LEARNING_RATE = Param('learning_rate', np.log(10 ** -6), np.log(10 ** 0), distrib='uniform', scale='log')
N_UNITS_1 = Param('n_units_1', np.log(2 ** 4), np.log(2 ** 8), distrib='uniform', scale='log', interval=1)
N_UNITS_2 = Param('n_units_2', np.log(2 ** 4), np.log(2 ** 8), distrib='uniform', scale='log', interval=1)
N_UNITS_3 = Param('n_units_3', np.log(2 ** 4), np.log(2 ** 8), distrib='uniform', scale='log', interval=1)
BATCH_SIZE = Param('batch_size', 32, 512, distrib='uniform', scale='linear', interval=1)
LR_STEP = Param('lr_step', 1, 5, distrib='uniform', init_val=1, scale='linear', interval=1)
GAMMA = Param('gamma', np.log(10 ** -3), np.log(10 ** -1), distrib='uniform', init_val=0.1, scale='log')
WEIGHT_DECAY = Param('weight_decay', np.log(10 ** -6), np.log(10 ** -1), init_val=0.004, distrib='uniform', scale='log')
MOMENTUM = Param('momentum', 0.3, 0.999, init_val=0.9, distrib='uniform', scale='linear')
HYPERPARAMS_DOMAIN = {
    'learning_rate': LEARNING_RATE,
    'n_units_1': N_UNITS_1,
    'n_units_2': N_UNITS_2,
    'n_units_3': N_UNITS_3,
    'batch_size': BATCH_SIZE,
    'lr_step': LR_STEP,
    'gamma': GAMMA,
    'weight_decay': WEIGHT_DECAY,
    'momentum': MOMENTUM,
}

HYPERPARAMETERS_TO_OPTIMIZE = ('learning_rate', 'n_units_1', 'n_units_2', 'n_units_3', 'batch_size')


class CifarEvaluator(TorchEvaluator):

    def adjust_learning_rate(self, epoch: int, base_lr: float, gamma: float, step_size: int) -> None:
        """Sets the learning rate to the initial LR decayed by gamma every step_size epochs"""
        lr = base_lr * (gamma ** (epoch // step_size))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    @print_evaluation(verbose=False, goals_to_print=("validation_error", "test_error"))
    def evaluate(self, n_resources: int) -> OptimizationGoals:
        self.n_resources += n_resources
        arm = self.arm

        # Load model and optimiser from file to resume training
        start_epoch = self.resume_from_checkpoint()

        # Rest of the tunable hyperparameters
        batch_size = int(arm.batch_size)
        base_lr = arm.learning_rate
        lr_step = int(arm.lr_step)
        gamma = arm.gamma

        # Compute derived hyperparameters
        n_batches = int(n_resources * 10000 / batch_size)  # each unit of resource = 10,000 examples

        batches_per_epoch = len(self.dataset_loader.train_loader(batch_size))
        max_epochs = int(n_batches / batches_per_epoch) + 1
        step_size = int(max_epochs / lr_step) if lr_step <= max_epochs else max_epochs

        for epoch in range(start_epoch, start_epoch + max_epochs):
            # Adjust learning rate by decay schedule
            self.adjust_learning_rate(epoch, base_lr, gamma, step_size)
            # Train the net for one epoch
            self._train(epoch, min(n_batches, batches_per_epoch), batch_size=batch_size)
            # Decrement n_batches remaining
            n_batches -= batches_per_epoch

        # Evaluate trained net on val and test set
        val_error, val_correct, val_total = self._test(is_validation=True)
        test_error, test_correct, test_total = self._test(is_validation=False)

        self._save_checkpoint(start_epoch + max_epochs, val_error, test_error)
        return OptimizationGoals(validation_error=val_error, test_error=test_error, val_correct=val_correct,
                                 val_total=val_total, test_correct=test_correct, test_total=test_total)


class CifarProblem(HyperparameterOptimizationProblem):

    def __init__(self, data_dir: str, output_dir: str, dataset_loader: Type[ImageDatasetLoader] = CIFARLoader,
                 hyperparams_domain: Dict[str, Param] = HYPERPARAMS_DOMAIN,
                 hyperparams_to_opt: Tuple[str, ...] = HYPERPARAMETERS_TO_OPTIMIZE, in_channels: int = 3):
        dataset_loader = dataset_loader(data_dir)
        super().__init__(hyperparams_domain, dataset_loader, hyperparams_to_opt)
        self.output_dir = output_dir
        self.in_channels = in_channels
        self.dataset_loader = dataset_loader

    def get_evaluator(self) -> CifarEvaluator:
        arm = Arm()
        arm.draw_hp_val(domain=self.domain, hyperparams_to_opt=self.hyperparams_to_opt)
        model_builder = CNNBuilder(arm, in_channels=self.in_channels)
        return CifarEvaluator(model_builder, self.dataset_loader, output_dir=self.output_dir)
