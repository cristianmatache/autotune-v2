from __future__ import division

from typing import Any, Tuple, Optional, Dict, cast

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.ar_model import AR

from autotune.core import HyperparameterOptimisationProblem, Evaluator, Arm, OptimisationGoals, ModelBuilder, Domain, \
    Param
from autotune.datasets.bitmex_loader import BitmexLoader
from autotune.util.io import print_evaluation

HYPERPARAMS_DOMAIN = Domain(
    x=Param('x', -5, 10, distrib='uniform', scale='linear'),
    y=Param('y', 1, 15, distrib='uniform', scale='linear'))


class BitmexBuilder(ModelBuilder[Any, Any]):

    def __init__(self, arm: Arm):
        """
        :param arm: a combination of hyperparameters and their values
        """
        super().__init__(arm)

    def construct_model(self) -> None:
        pass  # TODO


class BitmexEvaluator(Evaluator):

    def __init__(self, model_builder: ModelBuilder, dataset_loader: BitmexLoader,
                 output_dir: str = ".", file_name: str = "model.pth"):
        """
        :param model_builder: builder of a machine learning model based on an arm
        :param dataset_loader: dataset loader
        :param output_dir: directory where to save the arms and their evaluation progress so far (as checkpoints)
        :param file_name: file (at output_dir/arm<i>/file_name) which stores the progress of the evaluation of an arm
        """
        super().__init__(model_builder, output_dir, file_name)
        self.todo = model_builder.construct_model()  # TODO
        self.dataset_loader = dataset_loader

        # save first checkpoint to file file_path
        self._save_checkpoint(epoch=0, val_error=1, test_error=1)

    @print_evaluation(verbose=True, goals_to_print=())
    def evaluate(self, n_resources: int) -> OptimisationGoals:
        """ Given an arm (draw of hyperparameter values), evaluate the Branin function on it
        :param n_resources: this parameter is not used in this function but all optimisers require this parameter
        :return: the function value for the current arm can be found in OptimisationGoals.fval, Note that test_error and
        validation_error attributes are mandatory for OptimisationGoals objects but Branin has no machine learning model
        """
        # training
        with BitmexLoader() as loader:
            train_set = loader.train_set
            n_seen, n_bins = train_set.shape

        bin_predictors = {time_bin: AR(list(train_set.iloc[:, time_bin])) for time_bin in range(n_bins)}

        # test - validation set
        val_error = self._test(n_bins, n_seen, bin_predictors, is_validation=True)
        # test_error = self._test(n_bins, n_seen, bin_predictors, is_validation=False)

        # return OptimisationGoals(test_error=test_error, validation_error=val_error)
        return OptimisationGoals(test_error=1, validation_error=val_error)

    def _train(self, epoch: int, max_batches: int, batch_size: int) -> float:
        pass

    def _test(  # type: ignore # pylint: disable=arguments-differ  # noqa # Not used
            self, n_bins: int, n_seen: int, bin_predictors: Dict[int, AR], is_validation: bool
    ) -> float:
        with BitmexLoader() as loader:
            val_set = loader.val_set
            n_predictions, _ = val_set.shape

        errors = []
        if is_validation:
            prediction_interval = (n_seen, n_seen+n_predictions-1)
        else:
            prediction_interval = (n_seen+n_predictions, n_seen+2*n_predictions-1)

        for time_bin in range(n_bins):
            if time_bin == 81:
                continue
            expected = np.array(val_set.iloc[:, time_bin])
            predicted = bin_predictors[time_bin].fit().predict(*prediction_interval)
            plt.plot(list(range(len(expected))), expected)
            plt.plot(list(range(len(expected))), predicted)
            error = np.mean((predicted - expected) ** 2)
            errors.append(error)
        print(np.sqrt(errors))
        res_error = np.sqrt(np.mean(errors))
        return cast(float, res_error)

    def _save_checkpoint(self, epoch: int, val_error: float, test_error: float) -> None:
        pass


class BitmexProblem(HyperparameterOptimisationProblem):
    """
    Predicting Bitcoin trade volumes based on data from BitMEX exchange
    """

    def __init__(self, output_dir: Optional[str] = None, hyperparams_domain: Domain = HYPERPARAMS_DOMAIN,
                 hyperparams_to_opt: Tuple[str, ...] = ()):
        """
        :param output_dir: directory where to save the arms and their evaluation progress so far (as checkpoints)
        :param hyperparams_domain: names of the hyperparameters of a model along with their domain, that is
                                   ranges, distributions etc. (self.domain)
        :param hyperparams_to_opt: names of hyperparameters to be optimised, if () all params from domain are optimised
        """
        super().__init__(hyperparams_domain, hyperparams_to_opt, output_dir=output_dir)

    def get_evaluator(self, arm: Optional[Arm] = None) -> BitmexEvaluator:
        """
        :param arm: a combination of hyperparameters and their values
        :return: problem evaluator for an arm (given or random if not given)
        """
        if arm is None:  # if no arm is provided, generate a random arm
            arm = Arm()
            arm.draw_hp_val(domain=self.domain, hyperparams_to_opt=self.hyperparams_to_opt)
        model_builder = BitmexBuilder(arm)
        assert self.output_dir is not None
        return BitmexEvaluator(model_builder, dataset_loader=BitmexLoader(), output_dir=self.output_dir)


if __name__ == "__main__":
    bitmex_evaluator = BitmexProblem().get_evaluator()
    bitmex_evaluator.evaluate(n_resources=0)
    plt.show()
