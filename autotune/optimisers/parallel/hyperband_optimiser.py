from math import ceil
from os.path import join as path_join
from typing import Callable, List, Optional

import mpmath
import pandas as pd
from colorama import Fore, Style

from autotune.core import (
    Evaluation, Evaluator, HyperparameterOptimisationProblem, OptimisationGoals, Optimiser, ShapeFamilyScheduler,
    SimulationProblem, optimisation_metric_user)

COL = Fore.MAGENTA

mpmath.mp.dps = 64


class ParallelHyperbandOptimiser(Optimiser):

    """ Examples of resources:
    1 Resource  = 10 000 training examples
    1 Resources = 1 epoch
    """

    def __init__(self, eta: int, max_iter: int = None, max_time: int = None, min_or_max: Callable = min,
                 optimisation_func: Callable[[OptimisationGoals], float] = Optimiser.default_optimisation_func,
                 is_simulation: bool = False, scheduler: Optional[ShapeFamilyScheduler] = None,
                 plot_simulation: bool = False):
        """
        :param eta: halving rate
        :param max_iter: max iteration (considered infinity if None) - stopping condition
        :param max_time: max time a user is willing to wait for (considered infinity if None) - stopping cond. NOT USED
        :param min_or_max: min/max (built in functions) - whether to minimize or to maximize the optimisation_goal
        :param optimisation_func: function in terms of which to perform optimisation (can aggregate several optimisation
                                  goals or can just return the value of one optimisation goal)
        :param is_simulation: flag if the problem under optimisation is a real machine learning problem or a simulation
        :param scheduler: if the problem is a simulation, the scheduler provides the parameters for families of shapes
        :param plot_simulation: each simulated loss function will be added to plt.plot, use plt.show() to see results
        """
        super().__init__(max_iter, max_time, min_or_max, optimisation_func, is_simulation, scheduler, plot_simulation)
        if max_iter is None:
            raise ValueError("For Hyperband max_iter cannot be None")
        self.eta = eta

    def _get_optimisation_func_val(self, evaluation: Evaluation) -> float:
        return self.optimisation_func(evaluation.optimisation_goals)

    def _get_best_n_evaluators(self, n: int, evaluations: List[Evaluation]) -> List[Evaluator]:
        """Note that for minimization we sort in ascending order while for maximization we sort in descending order by
        the value of the optimisation_func applied on evaluations.

        :param n: number of top "best evaluators" to retrieve
        :param evaluations: A list of ordered pairs (evaluator, result of evaluator's evaluate() method)
        :return: best n evaluators (those evaluators that gave the best n values on self.optimisation_goal)
        """
        is_descending = self.min_or_max == max
        sorted_evaluations_by_res = sorted(evaluations, key=self._get_optimisation_func_val, reverse=is_descending)
        sorted_evaluators = [evaluation.evaluator for evaluation in sorted_evaluations_by_res]
        return sorted_evaluators[:n]

    @optimisation_metric_user
    def run_optimisation(self, problem: HyperparameterOptimisationProblem, verbosity: bool = False) -> Evaluation:
        """
        :param problem: optimisation problem (eg. CIFAR, MNIST, SVHN, MRBI problems)
        :param verbosity: whether to print the results of every single evaluation/iteration
        :return: Evaluation of best arm (evaluator, optimisation_goals)
        """
        R = self.max_iter  # maximum amount of resource that can be allocated to a single hyperparameter configuration
        eta = self.eta     # halving rate

        def log_eta(x: int) -> int: return int(mpmath.log(x)/mpmath.log(eta))
        s_max = log_eta(R)              # number of unique executions of Successive Halving (minus one)
        s_min = 2 if s_max >= 2 else 0  # skip the rest of the brackets after s_min
        B = (s_max + 1) * R             # total/max resources (without reuse) per execution of Successive Halving

        # Exploration-exploitation trade-off management outer loop
        for s in reversed(range(s_min, s_max + 1)):
            self._run_bracket(problem, B, R, eta, s, verbosity)

        return self._get_best_evaluation()

    def _run_bracket(
            self, problem: HyperparameterOptimisationProblem, B: int, R: int, eta: float, s: int, verbosity: bool
    ) -> None:
        n = int(ceil(int(B / R / (s + 1)) * eta ** s))  # initial number of evaluators/configurations/arms
        r = R * eta ** (-s)  # initial resources allocated to each evaluator/arm

        evaluators = self._get_arms_for_bracket(problem, n)

        # Successive halving with rate eta - based on values of self.optimisation_func(opt goals of each evaluation)
        for i in range(s + 1):
            n_i = n * eta ** (-i)  # evaluate n_i evaluators/configurations/arms
            r_i = r * eta ** i  # each with r_i resources
            evaluations = [Evaluation(evaluator, evaluator.evaluate(n_resources=r_i)) for evaluator in evaluators]
            print(f"{COL}** Evaluated {int(n_i)} arms, each with {r_i:.2f} resources {Style.RESET_ALL}")

            # Halving: keep best 1/eta of them, which will be allocated more resources/iterations
            evaluators = self._get_best_n_evaluators(n=int(n_i / eta), evaluations=evaluations)

            best_evaluation_in_round = self.min_or_max(evaluations, key=self._get_optimisation_func_val)
            self._update_evaluation_history(*best_evaluation_in_round)

            self._update_optimiser_metrics()
            if verbosity:
                self._print_evaluation(self.optimisation_func(best_evaluation_in_round.optimisation_goals))

    def _get_arms_for_bracket(self, problem: HyperparameterOptimisationProblem, n: int) -> List[Evaluator]:
        if not self.is_simulation:
            evaluators = [problem.get_evaluator() for _ in range(n)]
            self._bracket_population_of_arms_to_csv(evaluators, n_arms=n)
        else:  # is simulation
            problem: SimulationProblem
            evaluators = [problem.get_evaluator(*self.scheduler.get_family() if self.scheduler else (),
                                                should_plot=self.plot_simulation)
                          for _ in range(n)]
        print(f"{COL}\n{'=' * 73}\n>> Generated {n} evaluators each with a random arm {Style.RESET_ALL}")
        return evaluators

    @staticmethod
    def _bracket_population_of_arms_to_csv(evaluators: List[Evaluator], n_arms: int) -> None:
        if evaluators:
            assert all(evaluator.output_dir == evaluators[0].output_dir for evaluator in evaluators)
            file_path = path_join(evaluators[0].output_dir, f'bracket-with-{n_arms}-arms.csv')
            pd.DataFrame(evaluator.arm.__dict__ for evaluator in evaluators).to_csv(file_path)

    def __str__(self) -> str:
        return f"\n> Starting Hyperband optimisation\n" \
               f"    Max iterations (R)      = {self.max_iter}\n" \
               f"    Halving rate (eta)      = {self.eta}\n" \
               f"  Optimizing ({self.min_or_max.__name__}) {self.optimisation_func.__doc__}"
