from typing import Dict, Union, Callable

from core.problem_def import HyperparameterOptimizationProblem
from core.arm import Arm
from core.optimiser import Optimiser


class RandomOptimiser(Optimiser):

    """ Note that in this class we will use the terms "evaluation" and "iteration" interchangeably.
    An evaluation means: trying a combination of hyperparameters (an arm) and getting the validation, test errors
    """

    def __init__(self, n_resources: int, max_iter: int = None, max_time: int = None,
                 optimization_goal: str = "test_error", min_or_max: Callable = min):
        """ Random search
        :param n_resources: number of resources per evaluation (of each arm)
        :param max_iter: max iteration (considered infinity if None)
        :param max_time: max time a user is willing to wait for (considered infinity if None)
        :param optimization_goal: what part of the OptimizationGoals the Optimiser will minimize/maximize eg. test_error
        :param min_or_max: min/max (built in functions) - whether to minimize or to maximize the optimization_goal
        """
        super().__init__(max_iter, max_time, optimization_goal, min_or_max)
        self.n_resources = n_resources
        self.name = "Random"

    def run_optimization(self, problem: HyperparameterOptimizationProblem, verbosity: bool = False) \
            -> Dict[str, Union[Arm, float]]:
        self._init_optimizer_metrics()

        while not self._needs_to_stop():
            # Draw random sample
            evaluator = problem.get_evaluator()
            opt_goals = evaluator.evaluate(self.n_resources)
            # Evaluate arm on problem
            # Update evaluation history: arms tried so far, validation and test errors so far
            self._update_evaluation_history(evaluator.arm, **opt_goals.__dict__)
            # Update evaluation metrics: time so far, number of evaluations so far, checkpoint times so far
            self._update_optimizer_metrics()

            if verbosity:
                self._print_evaluation(getattr(opt_goals, self.optimization_goal))

        return self.min_or_max(self.eval_history, key=lambda x: x[self.optimization_goal])
