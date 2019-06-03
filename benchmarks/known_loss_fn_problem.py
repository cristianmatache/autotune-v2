from __future__ import division
import numpy as np
from typing import Optional, Dict, List
import matplotlib.pyplot as plt

from core import Arm, OptimisationGoals, SimulationProblem, SimulationEvaluator, HyperparameterOptimisationProblem
from util.io import print_evaluation


class KnownFnEvaluator(SimulationEvaluator):

    def __init__(self, known_fs: Dict[Arm, List[float]], should_plot: bool = False, proposed_arm: Arm = None):
        """
        :param known_fs:
        :param should_plot:
        """
        super().__init__(0, 0, 0, 0)
        self.should_plot = should_plot
        self.known_fs = known_fs
        self.proposed_arm = self.arm = proposed_arm

    @print_evaluation(verbose=False, goals_to_print=("fval",))
    def evaluate(self, n_resources: int) -> OptimisationGoals:
        """ Given an arm (draw of hyperparameter values), find the existing loss function that minimizes mean square
        error with the proposed arm. That is, the loss function that corresponds to the arm that is closest to the
        proposed arm (in terms of mean square error) will be returned.
        :param n_resources: this parameter is not used in this function but all optimisers require this parameter
        :return: the function value for the current arm can be found in OptimisationGoals.fval, Note that test_error and
        validation_error attributes are mandatory for OptimisationGoals objects but Branin has no machine learning model
        """
        time = int(n_resources)
        available_arms: List[Arm] = list(self.known_fs.keys())
        hyperparams_names = self.proposed_arm.__dict__.keys()
        min_sq_error = np.inf
        min_sq_error_arm = None
        for arm in available_arms:
            sq_error = sum([(float(self.proposed_arm[hp]) - float(arm[hp])) ** 2 for hp in hyperparams_names])
            if sq_error < min_sq_error:
                min_sq_error_arm = arm
                min_sq_error = sq_error
        # print("*"*50, "\n", "min_sq_error:", min_sq_error)

        if self.should_plot:
            plt.plot(list(range(time)), self.known_fs[min_sq_error_arm][:time], linewidth=1.5)
            plt.xlabel("time/epoch/resources")
            plt.ylabel("error/loss")

        return OptimisationGoals(fval=self.known_fs[min_sq_error_arm][time-1], test_error=-1, validation_error=-1)


class KnownFnProblem(HyperparameterOptimisationProblem, SimulationProblem):

    """
    """

    def __init__(self, known_fs: Dict[Arm, List[float]], real_problem: HyperparameterOptimisationProblem):
        HyperparameterOptimisationProblem.__init__(self, real_problem.domain, real_problem.hyperparams_to_opt)
        SimulationProblem.__init__(self)
        self.known_fs = known_fs

    def get_evaluator(self, arm: Optional[Arm] = None,
                      should_plot: bool = False) -> KnownFnEvaluator:
        """
        :param arm: a combination of hyperparameters and their values
        :param should_plot:
        :return: problem evaluator for an arm (given or random if not given)
        """
        if arm is None:  # if no arm is provided, generate a random arm
            arm = Arm()
            arm.draw_hp_val(domain=self.domain, hyperparams_to_opt=self.hyperparams_to_opt)
        return KnownFnEvaluator(known_fs=self.known_fs, proposed_arm=arm, should_plot=should_plot)


if __name__ == "__main__":
    from benchmarks.branin_problem import BraninProblem
    branin_problem = BraninProblem()
    known_fns: Dict[Arm, List[float]] = {}
    for _ in range(10):
        evaluator = branin_problem.get_evaluator()
        known_fns[evaluator.arm] = [evaluator.evaluate(1).fval for _ in range(10)]

    known_fn_problem = KnownFnProblem(known_fs=known_fns, real_problem=branin_problem)
    known_fn_problem.get_evaluator(should_plot=True)
