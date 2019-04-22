from abc import abstractmethod
import numpy as np
from typing import Dict, Union, Callable
import time
from colorama import Fore, Style

from core.problem_def import HyperparameterOptimizationProblem
from core.arm import Arm


class Optimiser:

    """
    Base class for all optimisers, results of evaluations should be stored in self.eval_history.
    eval_history is used to find the optimum arm.
    """

    def __init__(self, max_iter: int = None, max_time: int = None,
                 optimization_goal: str = "test_error", min_or_max: Callable = min):
        """ Every optimiser should have a stopping condition
        :param max_iter: max iteration (considered infinity if None)
        :param max_time: max time a user is willing to wait for (considered infinity if None)
        :param optimization_goal: what part of the OptimizationGoals the Optimiser will minimize/maximize eg. test_error
        :param min_or_max: min/max (built in functions) - whether to minimize or to maximize the optimization_goal
        """
        # stop conditions
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
        """
        Optimizer metrics are:
        - time when the optimization started
        - time elapsed since we started optimization
        - number of iterations
        - times at which each evaluation ended
        Note that these metrics have a canonical update method _update_optimizer_metrics but can also be updated  in a
        different way (for example in TPE)
        """
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
        """
        :return: human readable formatting of an optimizer
        """
        return f"\n> Starting optimisation\n" \
               f"  Stop when:\n" \
               f"    Max iterations          = {self.max_iter}\n" \
               f"    Max time                = {self.max_time} seconds\n" \
               f"  Optimizing ({self.min_or_max.__name__}) {self.optimization_goal}"

    def _print_evaluation(self, goal_value: float) -> None:
        """ Prints statistics for each evaluation, if the current evaluation is the best (optimal) so far, this will be
        printed in green, otherwise this will be printed in red.
        :param goal_value: value of the optimization goal for a certain evaluation
        """
        num_spaces = 8
        best_test_error_so_far = self.min_or_max([x[self.optimization_goal] for x in self.eval_history])
        opt_goal_str = str(self.optimization_goal).replace('_', ' ')

        print(f"{Fore.GREEN if goal_value == best_test_error_so_far else Fore.RED}"
              f"\n> SUMMARY: iteration number: {self.num_iterations},{num_spaces * ' '}"
              f"time elapsed: {self.cum_time:.2f}s,{num_spaces * ' '}"
              f"current {opt_goal_str}: {goal_value:.5f},{num_spaces * ' '}"
              f" best {opt_goal_str} so far: {best_test_error_so_far:.5f}"
              f"{Style.RESET_ALL}")
