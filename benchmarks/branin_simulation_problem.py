from __future__ import division
import numpy as np
from typing import Optional, List, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as stats

from core import Arm, OptimisationGoals, ModelBuilder, Domain, RoundRobinShapeFamilyScheduler, SHAPE_FAMILY_TYPE, \
    SimulationProblem
from core.params import *
from util.io import print_evaluation
from benchmarks.branin_problem import BraninBuilder, BraninEvaluator, BraninProblem, branin

HYPERPARAMS_DOMAIN = Domain(
    x=Param('x', -5, 10, distrib='uniform', scale='linear'),
    y=Param('y', 1, 15, distrib='uniform', scale='linear'))


def _plot_gamma_process_distribs(n: int, k: int) -> None:
    """ Overlaps all Gamma distributions of the Gamma process on which the simulation is based.
    :param n: number of distributions
    :param k: mode
    """
    x = np.linspace(0, 20, 200)

    def plot_gamma_distrib(time: int) -> None:
        sqrt_beta_component = np.sqrt(k ** 2 + 4 * (n - time))
        beta_t = (k + sqrt_beta_component) / (2 * (n - time))  # beta increases in terms of time so variance decreases
        alpha_t = k * beta_t + 1  # mode is always k (threshold for 0 aggressiveness)

        shape = alpha_t
        scale = 1 / beta_t

        y = stats.gamma.pdf(x, a=shape, scale=scale)
        plt.plot(x, y, "y-", label=r'$\alpha=29, \beta=3$')

    for t in range(n):
        plot_gamma_distrib(t)

    plt.xlabel("level of aggressiveness")
    plt.ylabel("pdf")
    plt.show()


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
                 up_spikiness: float = 0,
                 max_resources: int = 81,
                 init_noise: int = 0):
        """
        :param model_builder:
        :param output_dir:
        :param file_name:
        :param ml_aggressiveness:
        :param necessary_aggressiveness:
        :param up_spikiness:
        :param max_resources:
        """
        super().__init__(model_builder=model_builder, output_dir=output_dir, file_name=file_name)
        self.max_resources = max_resources
        self.fs: List[float] = []

        # Curve shape parameters DEPEND ON self.max_resources - feel free to add a schedule to them if you want
        # - ml aggressiveness = the higher h1 the more it bites from function debt - especially at the beginning
        self.ml_aggressiveness = ml_aggressiveness
        # - necessary aggressiveness = the higher h2 the later necessary aggressiveness starts - flattens tail later
        self.necessary_aggressiveness = necessary_aggressiveness
        # - up spikiniess = the lower h3 the smoother the function, the higher h3 the more upwards spikes
        #                   spikes are proportional with time left = bigger spikes at the beginning, smaller in the end
        self.up_spikiness = up_spikiness
        self.init_noise = init_noise

    def simulate(self, time: int, n: int, f_n: float) -> None:
        """ X is time/any other resources, Y is f(X) - simulated function at time X. Note that this function can
        simulate any type of resource not only time but for simplicity in naming we use time for resource. This method,
        will simulate Ys up to the given time. If the function was already evaluated at some times, it will continue
        from there.
        :param time: generate new Xs to simulate Ys, where lastX < newXs <= time
        :param n: the last point that the simulated function will be evaluated at
        :param f_n: target, f(n), last value of f, usually = branin - 200
        """
        k = 2  # mode of all gamma distributions - corresponds to 0 aggressiveness

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
                # (time_left if time_left < self.max_resources / 2 else 2 * np.sqrt(time_left)) instead of time_left
                if time_left == 1:
                    f_next_time = f_n
            self.fs.append(f_next_time)

    @print_evaluation(verbose=False, goals_to_print=())
    def evaluate(self, n_resources: int) -> OptimisationGoals:
        """ Given an arm (draw of hyperparameter values), evaluate the Branin function on it
        :param n_resources: this parameter is not used in this function but all optimisers require this parameter
        :return: the function value for the current arm can be found in OptimisationGoals.fval, Note that test_error and
        validation_error attributes are mandatory for OptimisationGoals objects but Branin has no machine learning model
        """
        time = int(n_resources)
        n_resources_before_first_halving = 0
        n = self.max_resources + n_resources_before_first_halving
        branin_value = branin(self.arm.x, self.arm.y)
        branin_noisy_value = branin(self.arm.x, self.arm.y, self.init_noise)
        f_n = branin_value - 200  # target (last value of f)
        # print(f"Starting from: {branin_value} and aiming to finish at: {f_n}")

        if not self.fs:
            self.fs = [branin_noisy_value]
            self.simulate(n_resources_before_first_halving, n, f_n)

        if time == 0:
            self.fs = [branin_noisy_value]
            self.simulate(n_resources_before_first_halving, n, f_n)
            return OptimisationGoals(fval=branin_value, test_error=-1, validation_error=-1)

        total_time = time + n_resources_before_first_halving
        self.simulate(total_time, n, f_n)

        plt.plot(list(range(total_time)), self.fs)
        if time == self.max_resources:
            assert self.fs[-1] == f_n

        return OptimisationGoals(fval=self.fs[-1], test_error=-1, validation_error=-1)


class BraninSimulationProblem(BraninProblem, SimulationProblem):

    """
    """

    def get_evaluator(self, arm: Optional[Arm] = None,
                      ml_aggressiveness: float = 0.9,
                      necessary_aggressiveness: float = 10,
                      up_spikiness: float = 0.1,
                      max_resources: int = 81,
                      init_noise: int = 0) -> BraninSimulationEvaluator:
        """
        :param arm: a combination of hyperparameters and their values
        :param ml_aggressiveness:
        :param necessary_aggressiveness:
        :param up_spikiness:
        :param max_resources:
        :param init_noise:
        :return: problem evaluator for an arm (given or random if not given)
        """
        if arm is None:  # if no arm is provided, generate a random arm
            arm = Arm()
            arm.draw_hp_val(domain=self.domain, hyperparams_to_opt=self.hyperparams_to_opt)
        model_builder = BraninBuilder(arm)
        return BraninSimulationEvaluator(model_builder, ml_aggressiveness=ml_aggressiveness,
                                         necessary_aggressiveness=necessary_aggressiveness,
                                         up_spikiness=up_spikiness, max_resources=max_resources,
                                         init_noise=init_noise)

    def plot_surface(self, n_simulations: int, max_resources: int = 81, n_resources: Optional[int] = None,
                     shape_families: Tuple[SHAPE_FAMILY_TYPE, ...] = ((None, 0.9, 10, 0.1, 81),),
                     init_noise: float = 0) -> None:
        """ plots the surface of the values of simulated loss functions at n_resources, by default is plot the losses
        at max resources, in which case their values would be 200-branin (if necessary aggressiveness is not disabled)
        :param n_simulations: number of simulations
        :param max_resources: maximum number of resources
        :param n_resources: show the surface of the values of simulated loss functions at n_resources
                            Note that this is relative to max_resources
        :param shape_families: shape families
        :param init_noise:
        """
        if n_resources is None:
            n_resources = max_resources
        assert n_resources <= max_resources
        scheduler = RoundRobinShapeFamilyScheduler(shape_families, max_resources, init_noise)

        xs, ys, zs = [], [], []
        for i in range(n_simulations):
            evaluator = self.get_evaluator(*scheduler.get_family())
            xs.append(evaluator.arm.x)
            ys.append(evaluator.arm.y)
            z = evaluator.evaluate(n_resources=n_resources).fval
            zs.append(z)
            if n_resources == max_resources:
                assert z == branin(evaluator.arm.x, evaluator.arm.y) - 200

        xs, ys, zs = [np.array(array, dtype="float64") for array in (xs, ys, zs)]
        fig = plt.figure()
        ax = fig.gca(projection=Axes3D.name)
        surf = ax.plot_trisurf(xs, ys, zs, cmap="coolwarm", antialiased=True)
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()


if __name__ == "__main__":
    # function colors: 1 blue 2 green 3 orange 4 red 5 purple 6 brown 7 pink 8 grey
    branin_problem = BraninSimulationProblem()
    families_of_shapes = (
        (None, 1.3, 10.0, 0.14),  # with aggressive start
        (None, 0.6, 7.0, 0.1),    # with average aggressiveness at start and at the beginning
        (None, 0.3, 3.0, 0.2),    # non aggressive start, aggressive end
    )
    branin_problem.plot_surface(n_simulations=9, max_resources=81, n_resources=81, shape_families=families_of_shapes,
                                init_noise=0.2)

    # evaluator = branin_problem.get_evaluator()
    # evaluator.evaluate(30)
    # evaluator.evaluate(50)
    # evaluator.evaluate(81)
    plt.show()
    # _plot_gamma_process_distribs(50, 2)
