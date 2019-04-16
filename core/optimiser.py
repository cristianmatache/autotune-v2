from abc import abstractmethod
import numpy as np
from typing import Dict, Union, Callable
import time

from core.problem_def import HyperparameterOptimizationProblem
from core.arm import Arm


class Optimiser:

    def __init__(self, n_resources: int, max_iter: int = None, max_time: int = None,
                 optimization_goal: str = "test_error", min_or_max: Callable = min):
        """ Every optimiser should have a stopping condition
        :param n_resources: max resources
        :param max_iter: max iteration (considered infinity if None)
        :param max_time: max time a user is willing to wait for (considered infinity if None)
        :param optimization_goal: what part of the OptimizationGoals the Optimiser will minimize/maximize eg. test_error
        :param min_or_max: min/max (built in functions) - whether to minimize or to maximize the optimization_goal
        """
        # stop conditions
        self.n_resources = n_resources
        self.max_time = np.inf if max_time is None else max_time
        self.max_iter = np.inf if max_iter is None else max_iter
        if (max_iter is None) and (max_time is None):
            raise ValueError("max_iter and max_time cannot be None simultaneously")
        self.eval_history = []
        self.optimization_goal = optimization_goal
        if min_or_max not in [min, max]:
            raise ValueError("optimization must be a built in function: min or max")
        self.min_or_max = min_or_max

    def _update_evaluation_history(self, arm: Arm, validation_error: float, test_error: float, **kwargs: float) -> None:
        latest_evaluation = {
            'arm': arm,                            # combinations of hyperparameters (arms) tried so far
            'validation_error': validation_error,  # validation errors so far
            'test_error': test_error               # test errors so far
        }
        latest_evaluation.update(kwargs)
        self.eval_history.append(latest_evaluation)

    @abstractmethod
    def run_optimization(self, problem: HyperparameterOptimizationProblem, verbosity: bool) \
            -> Dict[str, Union[Arm, float]]:
        """
        :param problem: optimization problem (eg. CIFAR, MNIST, SVHN, MRBI problems)
        :param verbosity: whether to print the results of every single evaluation/iteration
        :return: {'arm': Arm (best arm), 'test_error': float, 'validation_error': float}
        """
        pass

    def _init_optimizer_metrics(self) -> None:
        self.time_zero = time.time()  # start time of optimization
        self.cum_time = 0             # cumulative time = time of last evaluation/iteration - start time
        self.num_iterations = 0       #
        self.checkpoints = []         # list of cum_times of successful evaluations/iterations so far

    def _update_optimizer_metrics(self) -> None:
        self.cum_time = time.time() - self.time_zero
        self.num_iterations += 1
        self.checkpoints.append(self.cum_time)

    def _needs_to_stop(self) -> bool:
        def _is_time_over(): return self.max_time <= self.cum_time
        def _exceeded_iterations(): return self.max_iter <= self.num_iterations
        if _is_time_over():
            print("\nFINISHED: Time limit exceeded")
        elif _exceeded_iterations():
            print("\nFINISHED: Exceeded maximum number of iterations")
        return _is_time_over() or _exceeded_iterations()

    def __str__(self) -> str:
        return f"\n> Starting optimisation\n" \
               f"  Stop when:\n" \
               f"    Max iterations          = {self.max_iter}\n" \
               f"    Max time                = {self.max_time} seconds\n" \
               f"  Resource per iteration    = {self.n_resources}\n" \
               f"  Optimizing ({self.min_or_max.__name__}) {self.optimization_goal}"
