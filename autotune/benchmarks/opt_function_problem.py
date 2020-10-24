from __future__ import division

from typing import Any, Optional, Tuple, Union, cast

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from autotune.core import (
    Arm, Domain, Evaluator, HyperparameterOptimisationProblem, ModelBuilder, OptimisationGoals, Param)
from autotune.util.io import print_evaluation

AVAILABLE_OPT_FUNCTIONS = ('rastrigin', 'wave', 'branin', 'egg', 'camel')

HYPERPARAMS_DOMAIN_EGGHOLDER = Domain(
    x=Param('x', -512, 512, distrib='uniform', scale='linear'),
    y=Param('y', -512, 512, distrib='uniform', scale='linear'))

GLOBAL_MIN_EGGHOLDER = -959.6407

HYPERPARAMS_DOMAIN_BRANIN = Domain(
    x=Param('x', -5, 10, distrib='uniform', scale='linear'),
    y=Param('y', 1, 15, distrib='uniform', scale='linear'))

GLOBAL_MIN_BRANIN = 0.397887

HYPERPARAMS_DOMAIN_CAMEL = Domain(
    x=Param('x', -3, 3, distrib='uniform', scale='linear'),
    y=Param('y', -2, 2, distrib='uniform', scale='linear'))

GLOBAL_MIN_CAMEL = -1.0316

HYPERPARAMS_DOMAIN_WAVE = Domain(
    x=Param('x', -5.12, 5.12, distrib='uniform', scale='linear'),
    y=Param('y', -5.12, 5.12, distrib='uniform', scale='linear'))

GLOBAL_MIN_WAVE = -1
GLOBAL_MIN_RASTRIGIN = 0

DOMAINS = {
    'egg': HYPERPARAMS_DOMAIN_EGGHOLDER,
    'branin': HYPERPARAMS_DOMAIN_BRANIN,
    'camel': HYPERPARAMS_DOMAIN_CAMEL,
    'wave': HYPERPARAMS_DOMAIN_WAVE,
    'rastrigin': HYPERPARAMS_DOMAIN_WAVE,
}


def egg_holder(
        x1: Union[int, np.ndarray], x2: Union[int, np.ndarray], noise_variance: int = 0, scaled_noise: bool = False
) -> Union[int, np.ndarray]:
    """Egg Holder function.

    :param x1: x
    :param x2: y
    :param noise_variance: how noisy to make egg holder
    :param scaled_noise: whether to apply linear noise scaling
    :return: value of Egg Holder function
    """
    f = - (x2+47) * np.sin(np.sqrt(abs(x2 + 0.5*x1 + 47))) - x1 * np.sin(np.sqrt(abs(x1 - x2 - 47)))
    return f + noise_variance * np.random.randn(1)[0] * (1 if not scaled_noise else f - GLOBAL_MIN_EGGHOLDER)


def branin(
        x1: Union[int, np.ndarray], x2: Union[int, np.ndarray], noise_variance: float = 0, scaled_noise: bool = False
) -> Union[int, np.ndarray]:
    """Branin function.

    :param x1: x
    :param x2: y
    :param noise_variance: how noisy to make branin
    :param scaled_noise: whether to apply linear noise scaling
    :return: value of Branin function
    """
    a = 1
    b = 5.1 / (4 * np.pi ** 2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)

    f = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s
    return f + noise_variance * np.random.randn(1)[0] * (1 if not scaled_noise else f - GLOBAL_MIN_BRANIN)


def six_hump_camel(
        x1: Union[int, np.ndarray], x2: Union[int, np.ndarray], noise_variance: int = 0, scaled_noise: bool = False
) -> Union[int, np.ndarray]:
    """Six Hump Camel function.

    :param x1: x
    :param x2: y
    :param noise_variance: how noisy to make six hump camel
    :param scaled_noise:
    :return: value of Six Hump Camel function
    """
    f = (4 - (2.1*(x1**2)) + ((x1**4)/3)) * (x1**2)
    f += x1 * x2
    f += (-4 + 4*(x2**2)) * (x2**2)
    return f + noise_variance * np.random.randn(1)[0] * (1 if not scaled_noise else f - GLOBAL_MIN_CAMEL)


def drop_wave(
        x1: Union[int, np.ndarray], x2: Union[int, np.ndarray], noise_variance: int = 0, scaled_noise: bool = False
) -> Union[int, np.ndarray]:
    """Drop-wave function.

    :param x1: x
    :param x2: y
    :param noise_variance: how noisy to make drop-wave
    :param scaled_noise:
    :return: value of drop-wave function
    """
    f1 = 1 + np.cos(12*np.sqrt(x1**2 + x2**2))
    f2 = 0.5 * (x1**2 + x2**2) + 2
    f = - f1 / f2
    return f + noise_variance * np.random.randn(1)[0] * (1 if not scaled_noise else f - GLOBAL_MIN_WAVE)


def rastrigin(
        x1: Union[int, np.ndarray], x2: Union[int, np.ndarray], noise_variance: int = 0, scaled_noise: bool = False
) -> Union[int, np.ndarray]:
    """Rastrigin function.

    :param x1: x
    :param x2: y
    :param noise_variance: how noisy to make rastrigin
    :param scaled_noise:
    :return: value of rastrigin function
    """
    d = 2
    x = [x1, x2]

    def rastrigin_term(x_i: float) -> float:
        return cast(float, x_i**2 - 10 * np.cos(2*np.pi*x_i))

    f = 10*d + sum(rastrigin_term(x_i) for x_i in x)  # type: ignore  # sum() apparently is annotated with int only
    return f + noise_variance * np.random.randn(1)[0] * (1 if not scaled_noise else f - GLOBAL_MIN_RASTRIGIN)


OPT_FUNCTIONS = {
    'egg': egg_holder,
    'branin': branin,
    'camel': six_hump_camel,
    'wave': drop_wave,
    'rastrigin': rastrigin,
}


class OptFunctionBuilder(ModelBuilder[Any, Any]):

    def __init__(self, arm: Arm):
        """
        :param arm: a combination of hyperparameters and their values
        """
        super().__init__(arm, )

    def construct_model(self) -> None:
        """Opt function is a known function (so it has no machine learning model associated to it)"""


class OptFunctionEvaluator(Evaluator):

    def __init__(self, func_name: str, model_builder: ModelBuilder):
        super().__init__(model_builder)
        self.func_name = func_name

    @print_evaluation(verbose=False, goals_to_print=())
    def evaluate(self, n_resources: int) -> OptimisationGoals:
        """Given an arm (draw of hyperparameter values), evaluate the Optimization (Eg. Branin) function on it.

        :param n_resources: this parameter is not used in this function but all optimisers require this parameter
        :return: the function value for the current arm can be found in OptimisationGoals.fval, Note that test_error and
        validation_error attributes are mandatory for OptimisationGoals objects but 6HC has no machine learning model
        """
        return OptimisationGoals(fval=OPT_FUNCTIONS[self.func_name](self.arm.x, self.arm.y, noise_variance=0),
                                 test_error=-1, validation_error=-1)

    def _train(self, epoch: int, max_batches: int, batch_size: int) -> float:
        raise TypeError('Cannot call _train on well defined loss functions')

    def _test(self, is_validation: bool) -> Tuple[float, ...]:
        raise TypeError('Cannot call _test on well defined loss functions')

    def _save_checkpoint(self, epoch: int, val_error: float, test_error: float) -> None:
        raise TypeError('Cannot call _save_checkpoint on well defined loss functions')


class OptFunctionProblem(HyperparameterOptimisationProblem):

    """Canonical optimisation test problem See https://www.sfu.ca/~ssurjano/optimization.html."""

    def __init__(self, func_name: str, hyperparams_to_opt: Tuple[str, ...] = ()):
        """
        :param func_name: Name of the optimization function (Eg. branin, egg)
        :param hyperparams_to_opt: names of hyperparameters to be optimised, if () all params from domain are optimised
        """
        super().__init__(DOMAINS[func_name], hyperparams_to_opt)
        self.func_name = func_name

    def get_evaluator(self, arm: Optional[Arm] = None) -> OptFunctionEvaluator:
        """
        :param arm: a combination of hyperparameters and their values
        :return: problem evaluator for an arm (given or random if not given)
        """
        if arm is None:  # if no arm is provided, generate a random arm
            arm = Arm()
            arm.draw_hp_val(domain=self.domain, hyperparams_to_opt=self.hyperparams_to_opt)
        model_builder = OptFunctionBuilder(arm)
        return OptFunctionEvaluator(self.func_name, model_builder)

    def plot_surface(self, n_simulations: int) -> None:
        xs, ys, zs = [], [], []
        for _ in range(n_simulations):
            evaluator = self.get_evaluator()
            xs.append(evaluator.arm.x)
            ys.append(evaluator.arm.y)
            zs.append(evaluator.evaluate(n_resources=0).fval)

        xs, ys, zs = [np.array(array, dtype="float64") for array in (xs, ys, zs)]

        # plt.hist(zs, cumulative=False)
        # plt.xlabel(f"Values of {self.func_name} function")
        # plt.ylabel("Count")
        # plt.show()

        fig = plt.figure()
        ax = fig.gca(projection=Axes3D.name)
        surf = ax.plot_trisurf(xs, ys, zs, cmap="coolwarm", antialiased=True)
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()


if __name__ == "__main__":
    OptFunctionProblem('rastrigin').plot_surface(50000)
    OptFunctionProblem('wave').plot_surface(50000)
    OptFunctionProblem('egg').plot_surface(50000)
    OptFunctionProblem('branin').plot_surface(50000)
    OptFunctionProblem('camel').plot_surface(50000)
