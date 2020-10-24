import time
from abc import abstractmethod
from typing import Callable, List, Optional, Union, cast

import numpy as np
from colorama import Fore, Style

from autotune.core.evaluation import Evaluation
from autotune.core.evaluator import Evaluator
from autotune.core.optimisation_goals import OptimisationGoals
from autotune.core.problem_def import HyperparameterOptimisationProblem, SimulationProblem
from autotune.core.shape_family_scheduler import ShapeFamilyScheduler
from autotune.util.typing import MinOrMax


class Optimiser:
    """Base class for all optimisers, results of evaluations should be stored in self.eval_history which is used to find
    the optimum arm in terms of min/max self.optimisation_func(evaluation) for each evaluation in self.eval_history."""
    time_zero: float     # Start time of optimization
    cum_time: float      # Cumulative time = time of last evaluation/iteration - start time
    num_iterations: int  # Number of iterations
    checkpoints: List[float] = []  # List of cum_times of successful evaluations/iterations so far

    @staticmethod
    def default_optimisation_func(opt_goals: OptimisationGoals) -> float:
        """validation_error (Default optimisation_func)"""
        return cast(float, opt_goals.validation_error)

    def __init__(
            self, max_iter: Optional[int] = None, max_time: Optional[int] = None,
            min_or_max: MinOrMax = cast(MinOrMax, min),
            optimisation_func: Callable[[OptimisationGoals], float] = default_optimisation_func,
            is_simulation: bool = False, scheduler: Optional[ShapeFamilyScheduler] = None,
            plot_simulation: bool = False
    ):
        """
        :param max_iter: max iteration (considered infinity if None) - stopping condition
        :param max_time: max time a user is willing to wait for (considered infinity if None) - stopping condition
        :param min_or_max: min/max (built in functions) - whether to minimize or to maximize the optimization_goal
        :param optimisation_func: function in terms of which to perform optimization (can aggregate several optimization
                                  goals or can just return the value of one optimization goal)
        :param is_simulation: flag if the problem under optimisation is a real machine learning problem or a simulation
        :param scheduler: if the problem is a simulation, the scheduler provides the parameters for families of shapes
        :param plot_simulation: each simulated loss function will be added to plt.plot, use plt.show() to see results
        """
        # stop conditions
        if (max_iter is None) and (max_time is None):
            raise ValueError("max_iter and max_time cannot be None simultaneously")
        self.max_time: int = np.inf if max_time is None else max_time
        self.max_iter: int = np.inf if max_iter is None else max_iter

        if min_or_max not in [min, max]:
            raise ValueError(f"optimization must be a built in function: min or max, instead {min_or_max} was supplied")
        self.min_or_max = min_or_max
        self.optimisation_func = optimisation_func

        self.eval_history: List[Evaluation] = []

        # Simulation-specific attributes
        self.is_simulation = is_simulation
        self.scheduler = scheduler
        self.plot_simulation = plot_simulation

    def _update_evaluation_history(self, evaluator: Evaluator, opt_goals: OptimisationGoals) -> None:
        self.eval_history.append(Evaluation(evaluator, opt_goals))

    def _get_best_evaluation(self) -> Evaluation:
        return self.min_or_max(self.eval_history, key=lambda e: self.optimisation_func(e.optimisation_goals))

    @abstractmethod
    def run_optimisation(self, problem: HyperparameterOptimisationProblem, verbosity: bool) -> Evaluation:
        """
        :param problem: optimization problem (eg. CIFAR, MNIST, SVHN, MRBI problems)
        :param verbosity: whether to print the results of every single evaluation/iteration
        :return: Evaluation of best arm (evaluator, optimization_goals)
        """

    def init_optimiser_metrics(self) -> None:
        """Optimizer metrics are:

        - time when the optimization started
        - time elapsed since we started optimization
        - number of iterations
        - times at which each evaluation ended
        Note that these metrics have a canonical update method _update_optimiser_metrics but can also be updated in a
        different way (for example in TPE)
        """
        self.time_zero = time.time()        # start time of optimization
        self.cum_time: float = 0            # cumulative time = time of last evaluation/iteration - start time
        self.num_iterations = 0
        self.checkpoints: List[float] = []  # list of cum_times of successful evaluations/iterations so far

    def _update_optimiser_metrics(self) -> None:
        self.cum_time = time.time() - self.time_zero
        self.num_iterations += 1
        self.checkpoints.append(self.cum_time)

    def _needs_to_stop(self) -> bool:
        def _is_time_over() -> bool:
            return self.max_time <= self.cum_time

        def _exceeded_iterations() -> bool:
            return self.max_iter <= self.num_iterations

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
               f"  Optimizing ({self.min_or_max.__name__}) {self.optimisation_func.__name__}" \
               f" {self.optimisation_func.__doc__}"

    def _print_evaluation(self, opt_func_value: float) -> None:
        """Prints statistics for each evaluation, if the current evaluation is the best (optimal) so far, this will be
        printed in green, otherwise this will be printed in red.

        :param opt_func_value: value of the optimisation_func for a certain evaluation
        """
        num_spaces = 8
        best_so_far = self.min_or_max(
            [self.optimisation_func(evaluation.optimisation_goals) for evaluation in self.eval_history]
        )

        print(f"{Fore.GREEN if opt_func_value == best_so_far else Fore.RED}"
              f"\n> SUMMARY: iteration number: {self.num_iterations},{num_spaces * ' '}"
              f"time elapsed: {self.cum_time:.2f}s,{num_spaces * ' '}"
              f"current {self.optimisation_func.__doc__}: {opt_func_value:.5f},{num_spaces * ' '}"
              f" best {self.optimisation_func.__doc__} so far: {best_so_far:.5f}"
              f"{Style.RESET_ALL}")


Problem = Union[HyperparameterOptimisationProblem, SimulationProblem]
RunOptimisationType = Callable[[Optimiser, Problem, bool], Evaluation]


def optimisation_metric_user(run_optimisation: RunOptimisationType) -> RunOptimisationType:
    def wrapper(self: Optimiser, problem: Problem, verbosity: bool) -> Evaluation:
        self.init_optimiser_metrics()
        if self.is_simulation and not isinstance(problem, SimulationProblem):
            raise ValueError("You are trying to run a simulation but you have not provided a shape family schedule")
        return run_optimisation(self, problem, verbosity)
    return wrapper
