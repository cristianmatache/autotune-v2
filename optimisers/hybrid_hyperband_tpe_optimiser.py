from math import log, ceil
from typing import Callable
from colorama import Style, Fore

from core import HyperparameterOptimisationProblem, Evaluation, OptimisationGoals, Optimiser

from optimisers.hyperband_optimiser import HyperbandOptimiser
from optimisers.tpe_optimiser import TpeOptimiser

COL = Fore.MAGENTA
END = Style.RESET_ALL


class HybridHyperbandTpeOptimiser(HyperbandOptimiser):

    """
    Hybrid method Hyperband-TPE adapted from https://arxiv.org/pdf/1801.01596.pdf
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
        super().__init__(eta, max_iter, max_time, min_or_max, optimisation_func)

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

            # Successive halving with rate eta - based on values of self.optimisation_func(opt goals of each evaluation)
            evaluators = []
            for i in range(s+1):
                n_i = n*eta**(-i)  # evaluate n_i evaluators/configurations/arms
                r_i = r*eta**i     # each with r_i resources

                if i == 0:  # Generate first n_i arms/evaluators with TPE
                    tpe_optimiser = TpeOptimiser(n_resources=r_i, max_iter=n_i,
                                                 optimisation_func=self.optimisation_func)
                    tpe_optimiser.run_optimisation(problem, verbosity=True)

                    # evaluators = [h.evaluator for h in tpe_optimiser.eval_history]
                    evaluations = [Evaluation(h.evaluator, h.optimisation_goals) for h in tpe_optimiser.eval_history]

                    print(f"{COL}\n{'=' * 73}\n>> Generated {n} evaluators and evaluated with TPE for {r_i} resources\n"
                          f"--- Starting halving ---{END}")

                else:        # Continue with halving as in Hyperband
                    evaluations = [Evaluation(evaluator, evaluator.evaluate(n_resources=r_i))
                                   for evaluator in evaluators]
                    print(f"{COL}** Evaluated {int(n_i)} arms, each with {r_i:.2f} resources {END}")

                # Halving: keep best 1/eta of them, which will be allocated more resources/iterations
                evaluators = self._get_best_n_evaluators(n=int(n_i/eta), evaluations=evaluations)

                best_evaluation_in_round = self.min_or_max(evaluations, key=self._get_optimisation_func_val)
                self._update_evaluation_history(*best_evaluation_in_round)

                self._update_optimiser_metrics()
                if verbosity:
                    self._print_evaluation(self.optimisation_func(best_evaluation_in_round.optimization_goals))

        return self._get_best_evaluation()
