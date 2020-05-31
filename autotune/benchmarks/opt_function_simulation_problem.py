from __future__ import division

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from autotune.benchmarks.opt_function_problem import OptFunctionBuilder, OptFunctionEvaluator, OptFunctionProblem, \
    OPT_FUNCTIONS
from autotune.core import Arm, OptimisationGoals, ModelBuilder, RoundRobinShapeFamilyScheduler, ShapeFamily, \
    SimulationProblem, SimulationEvaluator
from autotune.util.io import print_evaluation


class OptFunctionSimulationEvaluator(OptFunctionEvaluator, SimulationEvaluator):
    EPSILON = 0.0001

    def __init__(self, func_name: str,
                 model_builder: ModelBuilder,
                 ml_aggressiveness: float = 0,
                 necessary_aggressiveness: float = 0,
                 up_spikiness: float = 0,
                 is_smooth: bool = False,
                 start_shift: int = 0,
                 end_shift: int = 200,
                 max_resources: int = 81,
                 init_noise: int = 0,
                 should_plot: bool = False):
        """
        :param model_builder:
        :param ml_aggressiveness:
        :param necessary_aggressiveness:
        :param up_spikiness:
        :param is_smooth:
        :param start_shift:
        :param end_shift:
        :param max_resources:
        :param init_noise:
        :param should_plot:
        """
        OptFunctionEvaluator.__init__(self, func_name=func_name, model_builder=model_builder)
        SimulationEvaluator.__init__(self, ml_aggressiveness, necessary_aggressiveness, up_spikiness, max_resources,
                                     is_smooth)
        self.init_noise = init_noise
        self.start_shift, self.end_shift = start_shift, end_shift
        self.should_plot = should_plot

    @print_evaluation(verbose=False, goals_to_print=())
    def evaluate(self, n_resources: int) -> OptimisationGoals:
        """
        Given an arm (draw of hyperparameter values), evaluate the Branin simulation function on it
        :param n_resources: this parameter is not used in this function but all optimisers require this parameter
        :return: the function value for the current arm can be found in OptimisationGoals.fval, Note that test_error and
        validation_error attributes are mandatory for OptimisationGoals objects but Branin has no machine learning model
        """
        time = int(n_resources)
        n_resources_before_first_halving = 0
        n = self.max_resources + n_resources_before_first_halving
        branin_value = OPT_FUNCTIONS[self.func_name](self.arm.x, self.arm.y, noise_variance=0, scaled_noise=True)
        branin_noisy_value = OPT_FUNCTIONS[self.func_name](self.arm.x, self.arm.y, self.init_noise, scaled_noise=False)
        f_n = branin_value - self.end_shift  # target (last value of f)
        f_1 = branin_noisy_value - self.start_shift  # first value of f
        # print(f"Starting from: {branin_value} and aiming to finish at: {f_n}")

        if not self.non_smooth_fs:
            self.non_smooth_fs = [f_1]
            self.simulate(n, n, f_n)

        if time == 1:
            self.non_smooth_fs = [f_1]
            self.simulate(n, n, f_n)
            return OptimisationGoals(fval=self.fs[time-1], test_error=-1, validation_error=-1)

        self.simulate(n, n, f_n)

        if self.should_plot:
            plt.plot(list(range(time)), self.fs[:time], linewidth=1.5)
            plt.xlabel("time/epoch/resources")
            plt.ylabel("error/loss")
            # plt.ylim(-self.end_shift-10, 210-self.start_shift)
        if time == self.max_resources and self.necessary_aggressiveness != np.inf:
            assert self.non_smooth_fs[n-1] - f_n < self.EPSILON

        return OptimisationGoals(fval=self.fs[time-1], test_error=-1, validation_error=-1)


class OptFunctionSimulationProblem(OptFunctionProblem, SimulationProblem):

    def get_evaluator(  # type: ignore # pylint: disable=arguments-differ  # FIXME
            self, arm: Optional[Arm] = None,
            ml_aggressiveness: float = 0.9, necessary_aggressiveness: float = 10, up_spikiness: float = 0.1,
            is_smooth: bool = False, start_shift: int = 0, end_shift: int = 200,
            max_resources: int = 81, init_noise: int = 0, should_plot: bool = False
    ) -> OptFunctionSimulationEvaluator:
        """
        :param arm: a combination of hyperparameters and their values
        :param ml_aggressiveness:
        :param necessary_aggressiveness:
        :param up_spikiness:
        :param is_smooth:
        :param start_shift:
        :param end_shift:
        :param max_resources:
        :param init_noise:
        :param should_plot:
        :return: problem evaluator for an arm (given or random if not given)
        """
        if arm is None:  # if no arm is provided, generate a random arm
            arm = Arm()
            arm.draw_hp_val(domain=self.domain, hyperparams_to_opt=self.hyperparams_to_opt)
        model_builder = OptFunctionBuilder(arm)
        return OptFunctionSimulationEvaluator(
            func_name=self.func_name, model_builder=model_builder, ml_aggressiveness=ml_aggressiveness,
            necessary_aggressiveness=necessary_aggressiveness, up_spikiness=up_spikiness, is_smooth=is_smooth,
            start_shift=start_shift, end_shift=end_shift, max_resources=max_resources, init_noise=init_noise,
            should_plot=should_plot)

    def plot_surface(  # pylint: disable=arguments-differ
            self, n_simulations: int, max_resources: int = 81, n_resources: Optional[int] = None,
            shape_families: Tuple[ShapeFamily, ...] = (ShapeFamily(None, 0.9, 10, 0.1),),
            init_noise: float = 0
    ) -> None:
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
        for _ in range(n_simulations):
            evaluator = self.get_evaluator(*scheduler.get_family(), should_plot=True)  # type: ignore  # FIXME
            xs.append(evaluator.arm.x)
            ys.append(evaluator.arm.y)
            z = evaluator.evaluate(n_resources=n_resources).fval
            zs.append(z)

        xs, ys, zs = [np.array(array, dtype="float64") for array in (xs, ys, zs)]
        fig = plt.figure()
        ax = fig.gca(projection=Axes3D.name)
        surf = ax.plot_trisurf(xs, ys, zs, cmap="coolwarm", antialiased=True)
        fig.colorbar(surf, shrink=0.3, aspect=5)

        plt.show()


if __name__ == "__main__":
    # function colors: 1 blue 2 green 3 orange 4 red 5 purple 6 brown 7 pink 8 grey
    opt_func_name = 'wave'
    problem = OptFunctionSimulationProblem(opt_func_name)
    families_of_shapes_egg = (
        ShapeFamily(None, 1.5, 10, 15, False, 0, 200),  # with aggressive start
        ShapeFamily(None, 0.5, 7, 10, False, 0, 200),  # with average aggressiveness at start and at the beginning
        ShapeFamily(None, 0.2, 4, 7, True, 0, 200),  # non aggressive start, aggressive end
    )
    families_of_shapes_general = (
        ShapeFamily(None, 1.5, 10, 15, False),  # with aggressive start
        ShapeFamily(None, 0.5, 7, 10, False),  # with average aggressiveness at start and at the beginning
        ShapeFamily(None, 0.2, 4, 7, True),  # non aggressive start, aggressive end
    )
    families_of_shapes = {
        'egg': families_of_shapes_egg,
        'wave': families_of_shapes_general,
    }.get(opt_func_name, families_of_shapes_general)
    problem.plot_surface(n_simulations=1000, max_resources=81, n_resources=81,
                         shape_families=families_of_shapes,
                         init_noise=0.1)

    # evaluator = branin_problem.get_evaluator(is_smooth=True, should_plot=True)
    # evaluator.evaluate(81)
    # evaluator.evaluate(50)
    # evaluator.evaluate(30)
    # plt.show()
    # _plot_gamma_process_distribs(50, 2)
