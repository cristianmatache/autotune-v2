import numpy as np
from typing import Tuple

from core import HyperparameterOptimisationProblem, Arm, OptimisationGoals, Domain
from core.params import *

from datasets.image_dataset_loaders import MNISTLoader
from benchmarks.torch_evaluator import TorchEvaluator
from benchmarks.torch_model_builders import LogisticRegressionBuilder
from util.io import print_evaluation


LEARNING_RATE = Param('learning_rate', np.log(10 ** -6), np.log(10 ** 0), distrib='uniform', scale='log')
WEIGHT_DECAY = Param('weight_decay', np.log(10 ** -6), np.log(10 ** -1), distrib='uniform', scale='log')
MOMENTUM = Param('momentum', 0.3, 0.999, distrib='uniform', scale='linear')
BATCH_SIZE = Param('batch_size', 20, 2000, distrib='uniform', scale='linear', interval=1)
HYPERPARAMS_DOMAIN = Domain(
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    momentum=MOMENTUM,
    batch_size=BATCH_SIZE)


class MnistEvaluator(TorchEvaluator):

    @print_evaluation(verbose=True, goals_to_print=("test_correct", "validation_error"))
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
        # Compute derived hyperparameters
        n_batches = int(n_resources * 10000 / batch_size)  # each unit of resource = 10,000 examples

        batches_per_epoch = len(self.dataset_loader.train_loader(batch_size))
        max_epochs = int(n_batches / batches_per_epoch) + 1

        for epoch in range(start_epoch, start_epoch + max_epochs):
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


class MnistProblem(HyperparameterOptimisationProblem):

    """
    Classification on MNIST dataset with logistic regression
    """

    def __init__(self, data_dir: str, output_dir: str,
                 hyperparams_domain: Domain = HYPERPARAMS_DOMAIN, hyperparams_to_opt: Tuple[str, ...] = ()):
        """
        :param data_dir: directory where the dataset is stored (or will be downloaded to if not already)
        :param output_dir: directory where to save the arms and their evaluation progress so far (as checkpoints)
        :param hyperparams_domain: names of the hyperparameters of a model along with their domain, that is
                                   ranges, distributions etc. (self.domain)
        :param hyperparams_to_opt: names of hyperparameters to be optimised, if () all params from domain are optimised
        """
        dataset_loader = MNISTLoader(data_dir)
        super().__init__(hyperparams_domain, hyperparams_to_opt, dataset_loader, output_dir)

    def get_evaluator(self, arm: Arm = None) -> MnistEvaluator:
        """
        :param arm: a combination of hyperparameters and their values
        :return: problem evaluator for an arm (given or random if not given)
        """
        if arm is None:  # if no arm is provided, generate a random arm
            arm = Arm()
            arm.draw_hp_val(domain=self.domain, hyperparams_to_opt=self.hyperparams_to_opt)
        model_builder = LogisticRegressionBuilder(arm)
        return MnistEvaluator(model_builder, self.dataset_loader, output_dir=self.output_dir)
