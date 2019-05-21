from __future__ import division
import numpy as np
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from core import Arm, OptimisationGoals, ModelBuilder, RoundRobinShapeFamilyScheduler, ShapeFamily, \
    SimulationProblem, SimulationEvaluator
from util.io import print_evaluation
from benchmarks.branin_problem import BraninBuilder, BraninEvaluator, BraninProblem, branin


class BraninSimulationEvaluator(BraninEvaluator, SimulationEvaluator):

    def __init__(self, model_builder: ModelBuilder, output_dir: Optional[str] = None, file_name: str = "model.pth",
                 ml_aggressiveness: float = 0,
                 necessary_aggressiveness: float = 0,
                 up_spikiness: float = 0,
                 start_shift: int = 0,
                 end_shift: int = 200,
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
        BraninEvaluator.__init__(self, model_builder=model_builder, output_dir=output_dir, file_name=file_name)
        SimulationEvaluator.__init__(self, ml_aggressiveness, necessary_aggressiveness, up_spikiness, max_resources)
        self.init_noise = init_noise
        self.start_shift, self.end_shift = start_shift, end_shift

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
        f_n = branin_value - self.end_shift  # target (last value of f)
        f_1 = branin_noisy_value - self.start_shift  # first value of f
        # print(f"Starting from: {branin_value} and aiming to finish at: {f_n}")

        if not self.fs:
            self.fs = [f_1]
            self.simulate(n_resources_before_first_halving, n, f_n)

        if time == 0:
            self.fs = [f_1]
            self.simulate(n_resources_before_first_halving, n, f_n)
            return OptimisationGoals(fval=f_1, test_error=-1, validation_error=-1)

        total_time = time + n_resources_before_first_halving
        self.simulate(total_time, n, f_n)

        # plt.plot(list(range(total_time)), self.fs)
        if time == self.max_resources and self.necessary_aggressiveness != np.inf:
            assert self.fs[-1] == f_n

        return OptimisationGoals(fval=self.fs[-1], test_error=-1, validation_error=-1)


class BraninSimulationProblem(BraninProblem, SimulationProblem):

    """
    """

    def get_evaluator(self, arm: Optional[Arm] = None,
                      ml_aggressiveness: float = 0.9,
                      necessary_aggressiveness: float = 10,
                      up_spikiness: float = 0.1,
                      start_shift: int = 0,
                      end_shift: int = 200,
                      max_resources: int = 81,
                      init_noise: int = 0) -> BraninSimulationEvaluator:
        """
        :param arm: a combination of hyperparameters and their values
        :param ml_aggressiveness:
        :param necessary_aggressiveness:
        :param up_spikiness:
        :param start_shift:
        :param end_shift:
        :param max_resources:
        :param init_noise:
        :return: problem evaluator for an arm (given or random if not given)
        """
        if arm is None:  # if no arm is provided, generate a random arm
            arm = Arm()
            arm.draw_hp_val(domain=self.domain, hyperparams_to_opt=self.hyperparams_to_opt)
        model_builder = BraninBuilder(arm)
        return BraninSimulationEvaluator(model_builder, ml_aggressiveness=ml_aggressiveness,
                                         necessary_aggressiveness=necessary_aggressiveness, up_spikiness=up_spikiness,
                                         start_shift=start_shift, end_shift=end_shift,
                                         max_resources=max_resources, init_noise=init_noise)

    def plot_surface(self, n_simulations: int, max_resources: int = 81, n_resources: Optional[int] = None,
                     shape_families: Tuple[ShapeFamily, ...] = (ShapeFamily(None, 0.9, 10, 0.1),),
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
        ShapeFamily(None, 1.3, 10.0, 0.14),  # with aggressive start
        ShapeFamily(None, 0.6, 7.0, 0.1),    # with average aggressiveness at start and at the beginning
        ShapeFamily(None, 0.3, 3.0, 0.2),    # non aggressive start, aggressive end
    )
    branin_problem.plot_surface(n_simulations=10, max_resources=81, n_resources=81, shape_families=families_of_shapes,
                                init_noise=0.3)

    # evaluator = branin_problem.get_evaluator()
    # evaluator.evaluate(30)
    # evaluator.evaluate(50)
    # evaluator.evaluate(81)
    plt.show()
    # _plot_gamma_process_distribs(50, 2)
