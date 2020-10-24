from __future__ import division

from typing import Optional, Dict, List

import matplotlib.pyplot as plt
import numpy as np

from autotune.core import Arm, OptimisationGoals, SimulationProblem, SimulationEvaluator, Domain, \
    HyperparameterOptimisationProblem
from autotune.util.io import print_evaluation


class KnownFnEvaluator(SimulationEvaluator):

    def __init__(self, known_fs: Dict[Arm, List[float]], proposed_arm: Arm, domain: Domain, should_plot: bool = False):
        super().__init__(0, 0, 0, 0)
        self.should_plot = should_plot
        self.known_fs = known_fs
        self.proposed_arm = self.arm = proposed_arm
        self.domain = domain

    @print_evaluation(verbose=False, goals_to_print=("fval",))
    def evaluate(self, n_resources: int) -> OptimisationGoals:
        """Given an arm (draw of hyperparameter values), find the existing loss function that minimizes mean square
        error with the proposed arm. That is, the loss function that corresponds to the arm that is closest to the
        proposed arm (in terms of mean square error) will be returned. Note that hyperparameter values of an arm are
        normalized before applying mean square error.

        :param n_resources: this parameter is not used in this function but all optimisers require this parameter
        :return: the function value for the current arm can be found in OptimisationGoals.fval, Note that test_error and
        validation_error attributes are mandatory for OptimisationGoals objects but Branin has no machine learning model
        """
        time = int(n_resources)
        available_arms: List[Arm] = list(self.known_fs.keys())
        hyperparams_names = self.proposed_arm.__dict__.keys()
        min_sq_error = np.inf
        min_sq_error_arm: Optional[Arm] = None

        def sq_error_term(arm1: Arm, arm2: Arm, hp: str) -> float:
            return (float(arm1[hp]) - float(arm2[hp])) ** 2

        for arm in available_arms:
            normalized_arm = Arm.normalize(arm, domain=self.domain)
            normalized_proposed_arm = Arm.normalize(self.proposed_arm, domain=self.domain)
            sq_error = sum(sq_error_term(normalized_proposed_arm, normalized_arm, hp_name)
                           for hp_name in hyperparams_names)

            if sq_error < min_sq_error:
                min_sq_error_arm = arm
                min_sq_error = sq_error

        # with open("min_sq_error_file.txt", "a") as f:
        #     f.write(f"{np.sqrt(min_sq_error/len(hyperparams_names))},")

        assert min_sq_error_arm is not None
        if self.should_plot:
            plt.plot(list(range(time)), self.known_fs[min_sq_error_arm][:time], linewidth=1.5)
            plt.xlabel("time/epoch/resources")
            plt.ylabel("error/loss")

        return OptimisationGoals(fval=self.known_fs[min_sq_error_arm][time-1], test_error=-1, validation_error=-1,
                                 fvals=self.known_fs[min_sq_error_arm])


class KnownFnProblem(HyperparameterOptimisationProblem, SimulationProblem):

    def __init__(self, known_fs: Dict[Arm, List[float]], real_problem: HyperparameterOptimisationProblem):
        HyperparameterOptimisationProblem.__init__(self, real_problem.domain, real_problem.hyperparams_to_opt)
        SimulationProblem.__init__(self)
        self.known_fs = known_fs

    def get_evaluator(  # type: ignore # pylint: disable=arguments-differ  # FIXME
            self, arm: Optional[Arm] = None, should_plot: bool = False
    ) -> KnownFnEvaluator:
        """
        :param arm: a combination of hyperparameters and their values
        :param should_plot:
        :return: problem evaluator for an arm (given or random if not given)
        """
        if arm is None:  # if no arm is provided, generate a random arm
            arm = Arm()
            arm.draw_hp_val(domain=self.domain, hyperparams_to_opt=self.hyperparams_to_opt)
        return KnownFnEvaluator(known_fs=self.known_fs, proposed_arm=arm, should_plot=should_plot, domain=self.domain)


if __name__ == "__main__":
    from autotune.benchmarks import OptFunctionProblem
    branin_problem = OptFunctionProblem('branin')
    known_fns: Dict[Arm, List[float]] = {}
    for _ in range(10):
        evaluator = branin_problem.get_evaluator()
        known_fns[evaluator.arm] = [evaluator.evaluate(1).fval for _ in range(10)]

    known_fn_problem = KnownFnProblem(known_fs=known_fns, real_problem=branin_problem)
    known_fn_problem.get_evaluator(should_plot=True)
