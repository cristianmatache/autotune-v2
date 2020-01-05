import numpy as np
from typing import Tuple, Type, Optional

from core import HyperparameterOptimisationProblem, Arm, OptimisationGoals, Domain, Param

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
HYPERPARAMS_DOMAIN = Domain(
    learning_rate=LEARNING_RATE,
    n_units_1=N_UNITS_1,
    n_units_2=N_UNITS_2,
    n_units_3=N_UNITS_3,
    batch_size=BATCH_SIZE,
    lr_step=LR_STEP,
    gamma=GAMMA,
    weight_decay=WEIGHT_DECAY,
    momentum=MOMENTUM)

HYPERPARAMETERS_TO_OPTIMIZE = ('learning_rate', 'n_units_1', 'n_units_2', 'n_units_3', 'batch_size')


class CifarEvaluator(TorchEvaluator):

    def _adjust_learning_rate(self, epoch: int, base_lr: float, gamma: float, step_size: int) -> None:
        """ Sets the learning rate to the initial LR decayed by gamma every step_size epochs. """
        lr = base_lr * (gamma ** (epoch // step_size))
        for param_group in self.optimiser.param_groups:
            param_group['lr'] = lr

    @print_evaluation(verbose=True, goals_to_print=("validation_error", "test_error"))
    def evaluate(self, n_resources: int) -> OptimisationGoals:
        """ Aggregate the steps:
            - train model (available through self._train)
            - evaluate model with respect to the test/validation set(s) (available through self._test)
            - report performance
        :param n_resources: number of resources allocated for training (used by Hyperband methods)
        :return: optimisation goals - metrics in terms of which we can perform optimisation
                 Eg. validation error, test error
        """
        self.n_resources = n_resources
        arm = self.arm

        # Load model and optimiser from file to resume training
        start_epoch = self._resume_from_checkpoint()

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
            self._adjust_learning_rate(epoch, base_lr, gamma, step_size)
            # Train the net for one epoch
            self._train(epoch, min(n_batches, batches_per_epoch), batch_size=batch_size)
            # Save/Update intermediate loss function values
            val_error, val_correct, val_total = self._test(is_validation=True)
            self._save_checkpoint(epoch, val_error, None)
            # Decrement n_batches remaining
            n_batches -= batches_per_epoch

        # Evaluate trained net on val and test set
        val_error, val_correct, val_total = self._test(is_validation=True)
        test_error, test_correct, test_total = self._test(is_validation=False)

        self._save_checkpoint(start_epoch + max_epochs, val_error, test_error)
        return OptimisationGoals(validation_error=val_error, test_error=test_error, val_correct=val_correct,
                                 val_total=val_total, test_correct=test_correct, test_total=test_total)


class CifarProblem(HyperparameterOptimisationProblem):

    """
    Classification on CIFAR-10 dataset with a CNN
    """

    def __init__(self, data_dir: str, output_dir: str, dataset_loader: Optional[Type[ImageDatasetLoader]] = CIFARLoader,
                 hyperparams_domain: Domain = HYPERPARAMS_DOMAIN,
                 hyperparams_to_opt: Tuple[str, ...] = HYPERPARAMETERS_TO_OPTIMIZE, in_channels: int = 3):
        """
        :param data_dir: directory where the dataset is stored (or will be downloaded to if not already)
        :param output_dir: directory where to save the arms and their evaluation progress so far (as checkpoints)
        :param dataset_loader: dataset loader class. Note it is not instantiated (e.g. CIFARLoader)
        :param hyperparams_domain: names of the hyperparameters of a model along with their domain, that is
                                   ranges, distributions etc. (self.domain)
        :param hyperparams_to_opt: names of hyperparameters to be optimised, if () all params from domain are optimised
        :param in_channels: in_channels of CNNBuilder (for CudaConvNet2)
        """
        if dataset_loader is not None:
            dataset_loader = dataset_loader(data_dir)
        super().__init__(hyperparams_domain, hyperparams_to_opt, dataset_loader, output_dir)
        self.in_channels = in_channels

    def get_evaluator(self, arm: Optional[Arm] = None) -> CifarEvaluator:
        """
        :param arm: a combination of hyperparameters and their values
        :return: problem evaluator for an arm (given or random if not given)
        """
        if arm is None:  # if no arm is provided, generate a random arm
            arm = Arm()
            arm.draw_hp_val(domain=self.domain, hyperparams_to_opt=self.hyperparams_to_opt)
        model_builder = CNNBuilder(arm, in_channels=self.in_channels)
        return CifarEvaluator(model_builder, self.dataset_loader, output_dir=self.output_dir)
