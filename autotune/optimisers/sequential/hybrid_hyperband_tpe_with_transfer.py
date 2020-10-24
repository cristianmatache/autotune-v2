import time
from abc import abstractmethod
from math import ceil
from typing import Callable, Dict, Optional, Tuple

import hyperopt
import mpmath
import pandas as pd
from colorama import Fore, Style
from hyperopt import Trials

from autotune.core import (
    Arm, Evaluation, Evaluator, HyperparameterOptimisationProblem, OptimisationGoals, Optimiser, ShapeFamilyScheduler,
    optimisation_metric_user)
from autotune.optimisers.sequential.hyperband_optimiser import HyperbandOptimiser
from autotune.optimisers.sequential.tpe_optimiser import TpeOptimiser

COL = Fore.MAGENTA
END = Style.RESET_ALL

# print = lambda *x: x

mpmath.mp.dps = 64


class HybridHyperbandTpeWithTransferOptimiser(HyperbandOptimiser):

    """Hybrid method Hyperband-TPE adapted from https://arxiv.org/pdf/1801.01596.pdf This method allows "history"
    transfer/bootstrapping from one bracket to another."""

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
        super().__init__(eta, max_iter, max_time, min_or_max, optimisation_func, is_simulation, scheduler,
                         plot_simulation)
        self.evaluations_by_resources: Dict[Evaluator, Tuple[int, OptimisationGoals, Arm]] = {}

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
            n = int(ceil(int(B/R/(s+1))*eta**s))  # initial number of evaluators/configurations/arms
            r = R*eta**(-s)                       # initial resources allocated to each evaluator/arm

            # Successive halving with rate eta - based on values of self.optimisation_func(opt goals of each evaluation)
            evaluators = []
            for i in range(s+1):
                n_i = n*eta**(-i)  # evaluate n_i evaluators/configurations/arms
                r_i = r*eta**i     # each with r_i resources

                if i == 0:  # Generate first n_i arms/evaluators with TPE
                    trials = self._get_trials(problem, r_i)
                    n_injected_trials = len(trials)
                    tpe_optimiser = TpeOptimiser(n_resources=r_i, max_iter=n_i + len(trials.trials),
                                                 optimisation_func=self.optimisation_func, min_or_max=self.min_or_max,
                                                 is_simulation=self.is_simulation, scheduler=self.scheduler,
                                                 trials_to_inject=trials, plot_simulation=self.plot_simulation)
                    tpe_optimiser.run_optimisation(problem, verbosity=True)

                    # evaluators = [h.evaluator for h in tpe_optimiser.eval_history]
                    evaluations = [Evaluation(h.evaluator, h.optimisation_goals) for h in tpe_optimiser.eval_history]

                    print(f"{COL}\n{'=' * 90}\n>> Generated {n} evaluators and evaluated with TPE for {r_i} resources "
                          f"with {n_injected_trials} trials injected \n--- Starting halving ---{END}")

                else:        # Continue with halving as in Hyperband
                    evaluations = [Evaluation(evaluator, evaluator.evaluate(n_resources=r_i))
                                   for evaluator in evaluators]
                    print(f"{COL}** Evaluated {int(n_i)} arms, each with {r_i:.2f} resources {END}")

                for evaluation in evaluations:
                    self.evaluations_by_resources[evaluation.evaluator] = (
                        int(r_i), evaluation.optimisation_goals, evaluation.evaluator.arm)

                # Halving: keep best 1/eta of them, which will be allocated more resources/iterations
                evaluators = self._get_best_n_evaluators(n=int(n_i/eta), evaluations=evaluations)

                best_evaluation_in_round = self.min_or_max(evaluations, key=self._get_optimisation_func_val)
                self._update_evaluation_history(*best_evaluation_in_round)

                self._update_optimiser_metrics()
                if verbosity:
                    self._print_evaluation(self.optimisation_func(best_evaluation_in_round.optimisation_goals))

        return self._get_best_evaluation()

    @abstractmethod
    def _is_transferable(self, evaluator_n_evaluated_resources: int, bracket_n_resources: int) -> bool:
        pass

    def _get_trials(self, problem: HyperparameterOptimisationProblem, n_resources: int) -> Trials:
        """Based on the method found on Github issues to inject trials into Hyperopt."""
        trials = Trials()
        if not self.eval_history:
            return trials

        hyperopt_selection = hyperopt.pyll.stochastic.sample(problem.get_hyperopt_space_from_hyperparams_to_opt())
        print(hyperopt_selection)

        df_dict = {
            'loss': [], **{hp_name: [] for hp_name in hyperopt_selection.keys()}, 'evaluator': [],
            'optimisation_goals': [], 'eval_time': []
        }
        for evaluator, (r, optimisation_goals, _) in self.evaluations_by_resources.items():
            if self._is_transferable(r, n_resources):
                sign = -1 if self.min_or_max == max else 1
                df_dict['loss'].append(sign * self.optimisation_func(optimisation_goals))
                for hp_name in hyperopt_selection.keys():
                    df_dict[hp_name].append(getattr(evaluator.arm, hp_name))
                df_dict['evaluator'].append(evaluator)
                df_dict['optimisation_goals'].append(optimisation_goals)
                df_dict['eval_time'] = time.time()
        df = pd.DataFrame(df_dict)

        test_trials = Trials()
        for tid, (index, row) in enumerate(df.iterrows()):
            hyperopt_trial = hyperopt.Trials().new_trial_docs(
                tids=[tid],
                specs=[None],
                results=[{'loss': row['loss'], 'status': hyperopt.STATUS_OK}],
                miscs=[{'tid': tid,
                        'cmd': ('domain_attachment', 'FMinIter_Domain'),
                        'idxs': {**{key: [tid] for key in hyperopt_selection.keys()}},
                        'vals': {**{key: [row[key]] for key in hyperopt_selection.keys()}},
                        'workdir': None
                        }]
            )
            hyperopt_trial[0]['state'] = hyperopt.JOB_STATE_DONE

            test_trials.insert_trial_docs(hyperopt_trial)
            test_trials.refresh()

        return test_trials


class HybridHyperbandTpeNoTransferOptimiser(HybridHyperbandTpeWithTransferOptimiser):

    def _is_transferable(self, evaluator_n_evaluated_resources: int, bracket_n_resources: int) -> bool:
        """No history transfer from previous brackets."""
        return False


class HybridHyperbandTpeTransferLongestOptimiser(HybridHyperbandTpeWithTransferOptimiser):

    def _is_transferable(self, evaluator_n_evaluated_resources: int, bracket_n_resources: int) -> bool:
        """Only transfer history that has been evaluated for max_iter resources (the survivors of previous brackets)"""
        return evaluator_n_evaluated_resources >= self.max_iter


class HybridHyperbandTpeTransferAllOptimiser(HybridHyperbandTpeWithTransferOptimiser):

    def _is_transferable(self, evaluator_n_evaluated_resources: int, bracket_n_resources: int) -> bool:
        """Transfer history that has been evaluated for at least the number of resources that we intend to start with.

        in the new bracket (that is, all comparable evaluations from previous brackets - where comparable means that the
        number of resources is not less that the number of resources that we want to generate)
        """
        return evaluator_n_evaluated_resources >= bracket_n_resources


class HybridHyperbandTpeTransferSameOptimiser(HybridHyperbandTpeWithTransferOptimiser):

    def _is_transferable(self, evaluator_n_evaluated_resources: int, bracket_n_resources: int) -> bool:
        """Transfer history that has been evaluated for exactly the number of resources that we intend to start with.

        in the new bracket (that is, all comparable evaluations from previous brackets - where comparable means that the
        number of resources is not less that the number of resources that we want to generate)
        """
        return evaluator_n_evaluated_resources == bracket_n_resources


class HybridHyperbandTpeTransferThresholdOptimiser(HybridHyperbandTpeWithTransferOptimiser):

    def __init__(self, eta: int, max_iter: int = None, max_time: int = None, min_or_max: Callable = min,
                 optimisation_func: Callable[[OptimisationGoals], float] = Optimiser.default_optimisation_func,
                 is_simulation: bool = False, scheduler: Optional[ShapeFamilyScheduler] = None,
                 n_res_transfer_threshold: Optional[int] = None):
        super().__init__(eta, max_iter, max_time, min_or_max, optimisation_func, is_simulation, scheduler)
        self.evaluations_by_resources: Dict[Evaluator, Tuple[int, OptimisationGoals, Arm]] = {}
        self.n_res_transfer_threshold = n_res_transfer_threshold

    def _is_transferable(self, evaluator_n_evaluated_resources: int, bracket_n_resources: int) -> bool:
        return (evaluator_n_evaluated_resources >= self.n_res_transfer_threshold and
                evaluator_n_evaluated_resources >= bracket_n_resources)  # Otherwise history would not be comparable
