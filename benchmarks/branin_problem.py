from __future__ import division
import numpy as np
from typing import Any, Tuple, Optional, Union
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from core import HyperparameterOptimisationProblem, Evaluator, Arm, OptimisationGoals, ModelBuilder, Domain
from core.params import *
from util.io import print_evaluation

HYPERPARAMS_DOMAIN = Domain(
    x=Param('x', -5, 10, distrib='uniform', scale='linear'),
    y=Param('y', 1, 15, distrib='uniform', scale='linear'))


def branin(x1: Union[int, np.ndarray], x2: Union[int, np.ndarray]) -> Union[int, np.ndarray]:
    """ Branin function
    :param x1: x
    :param x2: y
    :return: value of Branin function
    """
    a = 1
    b = 5.1 / (4 * np.pi ** 2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)

    f = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s
    return f


class BraninBuilder(ModelBuilder[Any, Any]):

    def __init__(self, arm: Arm):
        """
        :param arm: a combination of hyperparameters and their values
        """
        super().__init__(arm)

    def construct_model(self) -> None:
        """ Branin is a known function (so it has no machine learning model associated to it)
        """
        pass


class BraninEvaluator(Evaluator):

    @print_evaluation(verbose=True, goals_to_print=())
    def evaluate(self, n_resources: int) -> OptimisationGoals:
        """ Given an arm (draw of hyperparameter values), evaluate the Branin function on it
        :param n_resources: this parameter is not used in this function but all optimisers require this parameter
        :return: the function value for the current arm can be found in OptimisationGoals.fval, Note that test_error and
        validation_error attributes are mandatory for OptimisationGoals objects but Branin has no machine learning model
        """
        return OptimisationGoals(fval=branin(self.arm.x, self.arm.y), test_error=-1, validation_error=-1)

    def _train(self, *args: Any, **kwargs: Any) -> None:
        pass

    def _test(self, *args: Any, **kwargs: Any) -> None:
        pass

    def _save_checkpoint(self, epoch: int, val_error: float, test_error: float) -> None:
        pass


class BraninProblem(HyperparameterOptimisationProblem):

    """
    Canonical optimisation test problem
    See https://www.sfu.ca/~ssurjano/branin.html
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

    def get_evaluator(self, arm: Optional[Arm] = None) -> BraninEvaluator:
        """
        :param arm: a combination of hyperparameters and their values
        :return: problem evaluator for an arm (given or random if not given)
        """
        if arm is None:  # if no arm is provided, generate a random arm
            arm = Arm()
            arm.draw_hp_val(domain=self.domain, hyperparams_to_opt=self.hyperparams_to_opt)
        model_builder = BraninBuilder(arm)
        return BraninEvaluator(model_builder, self.output_dir)

    def plot_surface(self, n_simulations: int) -> None:
        xs, ys, zs = [], [], []
        for _ in range(n_simulations):
            evaluator = self.get_evaluator()
            xs.append(evaluator.arm.x)
            ys.append(evaluator.arm.y)
            zs.append(evaluator.evaluate(n_resources=0).fval)

        xs, ys, zs = [np.array(array, dtype="float64") for array in (xs, ys, zs)]
        fig = plt.figure()
        ax = fig.gca(projection=Axes3D.name)
        surf = ax.plot_trisurf(xs, ys, zs, cmap="coolwarm", antialiased=True)
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()


if __name__ == "__main__":
    BraninProblem().plot_surface(500)
