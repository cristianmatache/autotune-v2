from __future__ import division
import numpy as np
from typing import Any, Dict, Tuple, Optional

from core.problem_def import HyperparameterOptimizationProblem
from core.evaluator import Evaluator
from core.arm import Arm
from core.params import *
from core.optimization_goals import OptimizationGoals
from core.model_builder import ModelBuilder
from util.io import print_evaluation

HYPERPARAMS_DOMAIN = {
    'x': Param('x', -5, 10, distrib='uniform', scale='linear'),
    'y': Param('y', 1, 15, distrib='uniform', scale='linear')
}


class BraninBuilder(ModelBuilder[Any, Any]):

    def __init__(self, arm: Arm):
        """
        :param arm: hyperparameters and their values
        """
        super().__init__(arm)

    def construct_model(self) -> None:
        """ Branin is a known function (so it has no machine learning model associated with it)
        """
        pass


class BraninEvaluator(Evaluator):

    @print_evaluation(verbose=True, goals_to_print=())
    def evaluate(self, n_resources: int) -> OptimizationGoals:
        """ Given an arm (draw of hyperparameter values),
        :param n_resources: this parameter is not used in this function but all optimisers require this parameter
        :return: the function value for the current arm can be found in OptimizationGoals.fval, Note that test_error and
        validation_error attributes are mandatory for OptimizationGoals objects but Branin has no machine learning model
        """
        a = 1
        b = 5.1 / (4 * np.pi ** 2)
        c = 5 / np.pi
        r = 6
        s = 10
        t = 1 / (8 * np.pi)

        arm = self.arm
        x1 = arm.x
        x2 = arm.y

        f = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s
        return OptimizationGoals(fval=f, test_error=-1, validation_error=-1)

    def _train(self, *args: Any, **kwargs: Any) -> None:
        pass

    def _test(self, *args: Any, **kwargs: Any) -> None:
        pass

    def _save_checkpoint(self, epoch: int, val_error: float, test_error: float) -> None:
        pass


class BraninProblem(HyperparameterOptimizationProblem):

    """
    Canonical optimisation test problem
    See https://www.sfu.ca/~ssurjano/branin.html
    """

    def __init__(self, output_dir: str, hyperparams_domain: Dict[str, Param] = HYPERPARAMS_DOMAIN,
                 hyperparams_to_opt: Tuple[str, ...] = ()):
        """
        :param output_dir: directory where to save the arms and their evaluation progress so far (as checkpoints)
        :param hyperparams_domain: names of the hyperparameters of a model along with their domain, that is
                                   ranges, distributions etc. (self.domain)
        :param hyperparams_to_opt: names of hyperparameters to be optimized, if () all params from domain are optimized
        """
        super().__init__(hyperparams_domain, hyperparams_to_opt, output_dir=output_dir)

    def get_evaluator(self, arm: Optional[Arm] = None) -> BraninEvaluator:
        """
        :param arm: a draw of hyperparameters and their values
        :return: problem evaluator for an arm (given or random if not given)
        """
        if arm is None:  # if no arm is provided, generate a random arm
            arm = Arm()
            arm.draw_hp_val(domain=self.domain, hyperparams_to_opt=self.hyperparams_to_opt)
        model_builder = BraninBuilder(arm)
        return BraninEvaluator(model_builder, self.output_dir)
