from math import log, ceil
from typing import Callable
from colorama import Style, Fore

from core.problem_def import HyperparameterOptimizationProblem
from core.optimiser import Evaluation

from optimisers.hyperband_optimiser import HyperbandOptimiser
from optimisers.tpe_optimiser import TpeOptimiser

COL = Fore.MAGENTA
END = Style.RESET_ALL


class HybridHyperbandTpeOptimiser(HyperbandOptimiser):

    def __init__(self, eta: int, max_iter: int = None, max_time: int = None,
                 optimization_goal: str = "test_error", min_or_max: Callable = min):
        super().__init__(eta, max_iter, max_time, optimization_goal, min_or_max)
        if min_or_max == max:
            raise ValueError("Hybrid Hyperband-TPE supports minimization only, if you need maximization please "
                             "use minimization on the negative optimization goal")

    def run_optimization(self, problem: HyperparameterOptimizationProblem, verbosity: bool = False) -> Evaluation:
        self._init_optimizer_metrics()

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

            # Successive halving with rate eta - based on values of self.optimization_goal of each evaluation
            evaluators = []
            for i in range(s+1):
                n_i = n*eta**(-i)  # evaluate n_i evaluators/configurations/arms
                r_i = r*eta**i     # each with r_i resources

                if i == 0:  # Generate first n_i arms/evaluators with TPE
                    tpe_optimizer = TpeOptimiser(n_resources=r_i, max_iter=n_i,
                                                 optimization_goal=self.optimization_goal)
                    tpe_optimizer.run_optimization(problem, verbosity=True)

                    # evaluators = [h.evaluator for h in tpe_optimizer.eval_history]
                    evaluations = [Evaluation(h.evaluator, h.optimization_goals) for h in tpe_optimizer.eval_history]

                    print(f"{COL}\n{'=' * 73}\n>> Generated {n} evaluators and evaluated with TPE for {r_i} resources\n"
                          f"--- Starting halving ---{END}")

                else:        # Continue with halving as in Hyperband
                    evaluations = [Evaluation(evaluator, evaluator.evaluate(n_resources=r_i))
                                   for evaluator in evaluators]
                    print(f"{COL}** Evaluated {int(n_i)} arms, each with {r_i:.2f} resources {END}")

                # Halving: keep best 1/eta of them, which will be allocated more resources/iterations
                evaluators = self._get_best_n_evaluators(n=int(n_i/eta), evaluations=evaluations)

                best_evaluation_in_round = self.min_or_max(evaluations, key=self._get_opt_goal_val)
                self._update_evaluation_history(*best_evaluation_in_round)

                self._update_optimizer_metrics()
                if verbosity:
                    self._print_evaluation(getattr(best_evaluation_in_round.optimization_goals, self.optimization_goal))

        return self._get_best_evaluation()
