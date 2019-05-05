from __future__ import division
import numpy as np
from typing import Optional, List, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from core import Arm, OptimisationGoals, ModelBuilder, Domain
from core.params import *
from util.io import print_evaluation
from benchmarks.branin_problem import BraninBuilder, BraninEvaluator, BraninProblem, branin

HYPERPARAMS_DOMAIN = Domain(
    x=Param('x', -5, 10, distrib='uniform', scale='linear'),
    y=Param('y', 1, 15, distrib='uniform', scale='linear'))


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


class BraninSimulationEvaluator(BraninEvaluator):

    def __init__(self, model_builder: ModelBuilder, output_dir: Optional[str] = None, file_name: str = "model.pth",
                 ml_aggressiveness: float = 0,
                 necessary_aggressiveness: float = 0,
                 up_spikiness: float = 0):
        super().__init__(model_builder=model_builder, output_dir=output_dir, file_name=file_name)
        self.max_resources: int = 81  # TODO
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


class BraninSimulationProblem(BraninProblem):

    """
    """

    def get_evaluator(self, arm: Optional[Arm] = None,
                      ml_aggressiveness: float = 0.9,
                      necessary_aggressiveness: float = 10,
                      up_spikiness: float = 0.1) -> BraninSimulationEvaluator:
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
        return BraninSimulationEvaluator(model_builder, ml_aggressiveness=ml_aggressiveness,
                                         necessary_aggressiveness=necessary_aggressiveness,
                                         up_spikiness=up_spikiness)

    def plot_surface(self, n_simulations: int = 500, n_resources: Optional[int] = None,
                     shape_families: Tuple[Tuple[Optional[Arm], float, float, float], ...] = ((None, 0.9, 10, 0.1),)) \
            -> None:
        xs, ys, zs = [], [], []
        for i in range(n_simulations):
            evaluator = self.get_evaluator(*shape_families[i % len(shape_families)])
            xs.append(evaluator.arm.x)
            ys.append(evaluator.arm.y)
            if n_resources is None:
                n_resources = evaluator.max_resources
            z = evaluator.evaluate(n_resources=n_resources).fval
            assert z == branin(evaluator.arm.x, evaluator.arm.y) - 200
            zs.append(z)

        xs, ys, zs = [np.array(array, dtype="float64") for array in (xs, ys, zs)]
        fig = plt.figure()
        ax = fig.gca(projection=Axes3D.name)
        surf = ax.plot_trisurf(xs, ys, zs, cmap="coolwarm", antialiased=True)
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()


if __name__ == "__main__":
    # function colors: 1 blue 2 green 3 orange 4 red 5 purple 6 brown 7 pink 8 grey
    branin_problem = BraninSimulationProblem()
    family_of_shapes = (
                        (None, 1.3, 10.0, 0.14),  # with aggressive start
                        (None, 0.6, 7.0, 0.1),    # with average aggressiveness at start and at the beginning
                        (None, 0.3, 3.0, 0.2),    # non aggressive start, aggressive end
                        )
    branin_problem.plot_surface(10, shape_families=family_of_shapes)
    # [branin_problem.get_evaluator(*shape).evaluate(81) for _ in range(2) for shape in schedule]

    # evaluator = branin_problem.get_evaluator()
    # evaluator.evaluate(30)
    # evaluator.evaluate(50)
    # evaluator.evaluate(81)
    plt.show()
