from __future__ import division
import numpy as np
from typing import Any, Tuple, Optional, Union, List
import matplotlib.pyplot as plt

from core import HyperparameterOptimisationProblem, Evaluator, Arm, OptimisationGoals, ModelBuilder, Domain
from core.params import *
from util.io import print_evaluation

HYPERPARAMS_DOMAIN = Domain(
    x=Param('x', -5, 10, distrib='uniform', scale='linear'),
    y=Param('y', 1, 15, distrib='uniform', scale='linear'))


def branin(x1: Union[int, np.ndarray], x2: Union[int, np.ndarray]) -> Union[int, np.ndarray]:
    a = 1
    b = 5.1 / (4 * np.pi ** 2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)

    f = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s
    return f


def get_aggressiveness_from_gamma_distrib(time: int, n: int, k: int) -> float:
    sqrt_beta_component = np.sqrt(k**2 + 4 * (n - time))
    beta_t = (k + sqrt_beta_component) / (2*(n-time))  # beta is increasing in terms of time so variance is decreasing
    alpha_t = k * beta_t + 1  # mode is always k (threshold for 0 aggressiveness)

    shape = alpha_t
    scale = 1 / beta_t

    aggresiveness = np.random.gamma(shape, scale)
    # x = np.linspace(0, 20, 200)
    # y = stats.gamma.pdf(x, a=shape, scale=scale)
    # plt.plot(x, y, "y-", label=r'$\alpha=29, \beta=3$')
    return aggresiveness


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

    def __init__(self, model_builder: ModelBuilder, output_dir: Optional[str] = None, file_name: str = "model.pth",
                 ml_aggressiveness: float = 0,
                 necessary_aggressiveness: float = 0,
                 up_spikiness: float = 0):
        super().__init__(model_builder=model_builder, output_dir=output_dir, file_name=file_name)
        self.max_resources: int = 81
        self.fs: List[float] = []

        # Curve shape parameters DEPEND ON self.max_resources - feel free to add a schedule to them if you want
        # - ml aggressiveness = the higher h1 the more it bites from function debt - especially at the beginning
        self.ml_aggressiveness = ml_aggressiveness
        # - necessary aggressiveness = the higher h2 the later necessary aggressiveness starts - flattens tail later
        self.necessary_aggressiveness = necessary_aggressiveness
        # - up spikiniess = the lower h3 the smoother the function, the higher h3 the more upwards spikes
        #                   spikes are proportional with time left = bigger spikes at the beginning, smaller in the end
        self.up_spikiness = up_spikiness

    @print_evaluation(verbose=True, goals_to_print=())
    def evaluate(self, n_resources: int) -> OptimisationGoals:
        """ Given an arm (draw of hyperparameter values), evaluate the Branin function on it
        :param n_resources: this parameter is not used in this function but all optimisers require this parameter
        :return: the function value for the current arm can be found in OptimisationGoals.fval, Note that test_error and
        validation_error attributes are mandatory for OptimisationGoals objects but Branin has no machine learning model
        """
        time = n_resources
        n = self.max_resources
        branin_value = branin(self.arm.x, self.arm.y)
        k = 2  # mode of all gamma distributions - corresponds to 0 aggressiveness

        if time == 0:
            self.fs = [branin_value]
            return OptimisationGoals(fval=branin_value, test_error=-1, validation_error=-1)

        f_n = branin_value - 200
        print(f"Starting from: {branin_value}  and aiming to finish at: {f_n}")

        if not self.fs:
            self.fs = [branin_value]
        
        prev_time = len(self.fs)
        for t in range(prev_time, time):
            agg_level = get_aggressiveness_from_gamma_distrib(t, n + 1, k)
            f_time = self.fs[-1]
            if agg_level == k:  # be neutral
                f_next_time = f_time
            elif agg_level > k:  # be aggressive - go down with different aggressivenesses
                absolute_aggressiveness = agg_level - k
                function_debt = f_n - f_time
                ml_aggressed = f_time + absolute_aggressiveness * self.ml_aggressiveness * function_debt / 100
                time_aggressed = (f_n - ml_aggressed) * ((t / (n - 1)) ** self.necessary_aggressiveness)
                f_next_time = ml_aggressed + time_aggressed
            else:  # aggressiveness < k - go up
                time_left = n - t
                f_next_time = f_time + self.up_spikiness * time_left
                if time_left == 1:
                    f_next_time = f_n
            self.fs.append(f_next_time)

        if time == self.max_resources:
            plt.plot(list(range(time)), self.fs)
            assert self.fs[-1] == f_n

        return OptimisationGoals(fval=self.fs[-1], test_error=-1, validation_error=-1)

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

    def get_evaluator(self, arm: Optional[Arm] = None,
                      ml_aggressiveness: float = 0.9,
                      necessary_aggressiveness: float = 10,
                      up_spikiness: float = 0.1) -> BraninEvaluator:
        """
        :param arm: a combination of hyperparameters and their values
        :param ml_aggressiveness:
        :param necessary_aggressiveness:
        :param up_spikiness:
        :return: problem evaluator for an arm (given or random if not given)
        """
        if arm is None:  # if no arm is provided, generate a random arm
            arm = Arm()
            arm.draw_hp_val(domain=self.domain, hyperparams_to_opt=self.hyperparams_to_opt)
        model_builder = BraninBuilder(arm)
        # return BraninEvaluator(model_builder, self.output_dir)
        return BraninEvaluator(model_builder, ml_aggressiveness=ml_aggressiveness,
                               necessary_aggressiveness=necessary_aggressiveness, up_spikiness=up_spikiness)


if __name__ == "__main__":
    branin_problem = BraninProblem()
    schedule = [(None, 0.9, 10, 0.1), (None, 0.6, 5, 0.1), (None, 0.1, 6, 0.4)]
    [branin_problem.get_evaluator(*shape).evaluate(81) for _ in range(2) for shape in schedule]
    # evaluator = branin_problem.get_evaluator()
    # evaluator.evaluate(30)
    # evaluator.evaluate(50)
    # evaluator.evaluate(81)
    plt.show()
