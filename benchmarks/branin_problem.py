from __future__ import division
import numpy as np
from typing import Any, Dict, Tuple

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
        pass


class BraninEvaluator(Evaluator):

    @print_evaluation(verbose=True, goals_to_print=())
    def evaluate(self, n_resources: int) -> OptimizationGoals:
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
    @see https://www.sfu.ca/~ssurjano/branin.html
    """

    def __init__(self, output_dir: str, hyperparams_domain: Dict[str, Param] = HYPERPARAMS_DOMAIN,
                 hyperparams_to_opt: Tuple[str, ...] = ()):
        super().__init__(hyperparams_domain, hyperparams_to_opt, output_dir=output_dir)

    def get_evaluator(self, arm: Arm = None) -> BraninEvaluator:
        if arm is None:  # if no arm is provided, generate a random arm
            arm = Arm()
            arm.draw_hp_val(domain=self.domain, hyperparams_to_opt=self.hyperparams_to_opt)
        model_builder = BraninBuilder(arm)
        return BraninEvaluator(model_builder)
