from math import log, ceil
from typing import Callable, List
from colorama import Style, Fore

from core.optimiser import Optimiser, Evaluation, Evaluator, HyperparameterOptimisationProblem, OptimisationGoals

COL = Fore.MAGENTA


class HyperbandOptimiser(Optimiser):

    """ Examples of resources:
    1 Resource  = 10 000 training examples
    1 Resources = 1 epoch
    """

    def __init__(self, eta: int, max_iter: int = None, max_time: int = None, min_or_max: Callable = min,
                 optimisation_func: Callable[[OptimisationGoals], float] = Optimiser.default_optimisation_func):
        """
        :param eta: halving rate
        :param max_iter: max iteration (considered infinity if None) - stopping condition
        :param max_time: max time a user is willing to wait for (considered infinity if None) - stopping cond. NOT USED
        :param min_or_max: min/max (built in functions) - whether to minimize or to maximize the optimization_goal
        :param optimisation_func: function in terms of which to perform optimization (can aggregate several optimization
                                  goals or can just return the value of one optimization goal)
        """
        super().__init__(max_iter, max_time, min_or_max, optimisation_func)
        if max_iter is None:
            raise ValueError("For Hyperband max_iter cannot be None")
        self.eta = eta

    def _get_optimization_func_val(self, evaluation: Evaluation) -> float:
        return self.optimisation_func(evaluation.optimisation_goals)

    def _get_best_n_evaluators(self, n: int, evaluations: List[Evaluation]) -> List[Evaluator]:
        """ Note that for minimization we sort in ascending order while for maximization we sort in descending order by
        the value of the optimization_func applied on evaluations
        :param n: number of top "best evaluators" to retrieve
        :param evaluations: A list of ordered pairs (evaluator, result of evaluator's evaluate() method)
        :return: best n evaluators (those evaluators that gave the best n values on self.optimization_goal)
        """
        is_descending = self.min_or_max == max
        sorted_evaluations_by_res = sorted(evaluations, key=self._get_optimization_func_val, reverse=is_descending)
        sorted_evaluators = [evaluation.evaluator for evaluation in sorted_evaluations_by_res]
        return sorted_evaluators[:n]

    def run_optimisation(self, problem: HyperparameterOptimisationProblem, verbosity: bool = False) -> Evaluation:
        """
        :param problem: optimization problem (eg. CIFAR, MNIST, SVHN, MRBI problems)
        :param verbosity: whether to print the results of every single evaluation/iteration
        :return: Evaluation of best arm (evaluator, optimization_goals)
        """
        self._init_optimiser_metrics()

        R = self.max_iter  # maximum amount of resource that can be allocated to a single hyperparameter configuration
        eta = self.eta     # halving rate

        def log_eta(x: int) -> int: return int(log(x)/log(eta))
        s_max = log_eta(R)              # number of unique executions of Successive Halving (minus one)
        s_min = 2 if s_max >= 2 else 0  # skip the rest of the brackets after s_min
        B = (s_max + 1) * R             # total/max resources (without reuse) per execution of Successive Halving

        # Exploration-exploitation trade-off management outer loop
        for s in reversed(range(s_min, s_max + 1)):
            n = int(ceil(int(B/R/(s+1))*eta**s))  # initial number of evaluators/configurations/arms
            r = R*eta**(-s)                       # initial resources allocated to each evaluator/arm

            evaluators = [problem.get_evaluator() for _ in range(n)]
            print(f"{COL}\n{'=' * 73}\n>> Generated {n} evaluators each with a random arm {Style.RESET_ALL}")

            # Successive halving with rate eta - based on values of self.optimisation_func(opt goals of each evaluation)
            for i in range(s+1):
                n_i = n*eta**(-i)  # evaluate n_i evaluators/configurations/arms
                r_i = r*eta**i     # each with r_i resources
                evaluations = [Evaluation(evaluator, evaluator.evaluate(n_resources=r_i)) for evaluator in evaluators]
                print(f"{COL}** Evaluated {int(n_i)} arms, each with {r_i:.2f} resources {Style.RESET_ALL}")

                # Halving: keep best 1/eta of them, which will be allocated more resources/iterations
                evaluators = self._get_best_n_evaluators(n=int(n_i/eta), evaluations=evaluations)

                best_evaluation_in_round = self.min_or_max(evaluations, key=self._get_optimization_func_val)
                self._update_evaluation_history(*best_evaluation_in_round)

                self._update_optimiser_metrics()
                if verbosity:
                    self._print_evaluation(self.optimisation_func(best_evaluation_in_round.optimization_goals))

        return self._get_best_evaluation()

    def __str__(self) -> str:
        return f"\n> Starting Hyperband optimisation\n" \
               f"    Max iterations (R)      = {self.max_iter}\n" \
               f"    Halving rate (eta)      = {self.eta}\n" \
               f"  Optimizing ({self.min_or_max.__name__}) {self.optimisation_func.__doc__}"
