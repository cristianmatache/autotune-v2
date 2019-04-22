from abc import abstractmethod
import numpy as np
from typing import Callable, List
import time
from colorama import Fore, Style

from core.problem_def import HyperparameterOptimizationProblem
from core.optimization_goals import OptimizationGoals
from core.evaluator import Evaluator
from core.evaluation import Evaluation


class Optimiser:

    """
    Base class for all optimisers, results of evaluations should be stored in self.eval_history which is used to
    find the optimum arm in terms of min/max self.optimization_func(evaluation) for each evaluation in self.eval_history
    """

    @staticmethod
    def default_optimization_func(opt_goals: OptimizationGoals) -> float:
        """validation_error (Default optimization_func)"""
        return opt_goals.validation_error

    def __init__(self, max_iter: int = None, max_time: int = None, min_or_max: Callable = min,
                 optimization_func: Callable[[OptimizationGoals], float] = default_optimization_func):
        """
        :param max_iter: max iteration (considered infinity if None) - stopping condition
        :param max_time: max time a user is willing to wait for (considered infinity if None) - stopping condition
        :param min_or_max: min/max (built in functions) - whether to minimize or to maximize the optimization_goal
        :param optimization_func: function in terms of which to perform optimization (can aggregate several optimization
                                  goals or can just return the value of one optimization goal)
        """
        # stop conditions
        if (max_iter is None) and (max_time is None):
            raise ValueError("max_iter and max_time cannot be None simultaneously")
        self.max_time = np.inf if max_time is None else max_time
        self.max_iter = np.inf if max_iter is None else max_iter

        if min_or_max not in [min, max]:
            raise ValueError(f"optimization must be a built in function: min or max, instead {min_or_max} was supplied")
        self.min_or_max = min_or_max
        self.optimization_func = optimization_func

        self.eval_history: List[Evaluation] = []

    def _update_evaluation_history(self, evaluator: Evaluator, opt_goals: OptimizationGoals) -> None:
        self.eval_history.append(Evaluation(evaluator, opt_goals))

    def _get_best_evaluation(self) -> Evaluation:
        return self.min_or_max(self.eval_history, key=lambda e: self.optimization_func(e.optimization_goals))

    @abstractmethod
    def run_optimization(self, problem: HyperparameterOptimizationProblem, verbosity: bool) -> Evaluation:
        """
        :param problem: optimization problem (eg. CIFAR, MNIST, SVHN, MRBI problems)
        :param verbosity: whether to print the results of every single evaluation/iteration
        :return: Evaluation of best arm (evaluator, optimization_goals)
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
               f"  Optimizing ({self.min_or_max.__name__}) {self.optimization_func.__name__}" \
               f" {self.optimization_func.__doc__}"

    def _print_evaluation(self, opt_func_value: float) -> None:
        """ Prints statistics for each evaluation, if the current evaluation is the best (optimal) so far, this will be
        printed in green, otherwise this will be printed in red.
        :param opt_func_value: value of the optimization_func for a certain evaluation
        """
        num_spaces = 8
        best_so_far = self.min_or_max(
            [self.optimization_func(evaluation.optimization_goals) for evaluation in self.eval_history]
        )

        print(f"{Fore.GREEN if opt_func_value == best_so_far else Fore.RED}"
              f"\n> SUMMARY: iteration number: {self.num_iterations},{num_spaces * ' '}"
              f"time elapsed: {self.cum_time:.2f}s,{num_spaces * ' '}"
              f"current {self.optimization_func.__doc__}: {opt_func_value:.5f},{num_spaces * ' '}"
              f" best {self.optimization_func.__doc__} so far: {best_so_far:.5f}"
              f"{Style.RESET_ALL}")
