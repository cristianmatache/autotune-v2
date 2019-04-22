import numpy as np
from sigopt import Connection
from sigopt.objects import Assignments
from typing import Callable, Tuple

from core import Optimiser, Evaluation, HyperparameterOptimizationProblem, Arm, OptimizationGoals, Evaluator

SIGOPT_API_KEY = "RAGFJSAISOJGFQOXCAVIVQRNNGOQNYGDEYISHTETQZCNWJNA"


class SigOptimiser(Optimiser):

    def __init__(self, n_resources: int, max_iter: int = None, max_time: int = None,
                 optimization_goal: str = "validation_error", min_or_max: Callable = min):
        super().__init__(max_iter, max_time, optimization_goal, min_or_max)

        # SigOpt supports maximization only, so if the problem is minimization, maximize -1 * optimization goal
        self.sign = -1 if min_or_max == max else 1
        self.n_resources = n_resources

    def run_optimization(self, problem: HyperparameterOptimizationProblem, verbosity: bool) -> Evaluation:
        self._init_optimizer_metrics()

        # Wrap parameter space
        space = problem.get_sigopt_space_from_hyperparams_to_opt()

        # Create SigOpt experiment
        conn = Connection(client_token=SIGOPT_API_KEY)
        experiment = conn.experiments().create(name=type(problem).__name__, parameters=space,
                                               observation_budget=self.max_iter)
        print(f"Created experiment: https://sigopt.com/experiment/{experiment.id}")

        # Clear any open suggestions
        conn.experiments(experiment.id).suggestions().delete(state="open")

        while not self._needs_to_stop():
            # Draw sample using SigOpt suggestion service
            suggestion = conn.experiments(experiment.id).suggestions().create()
            arm_dict = suggestion.assignments

            # Evaluate arm on problem
            evaluator, opt_goals = self.sigopt_objective_function(problem, arm_dict)

            # Update history
            self._update_evaluation_history(evaluator, opt_goals)

            # Add observation to SigOpt history
            conn.experiments(experiment.id).observations().create(
                suggestion=suggestion.id,
                value=self.sign * getattr(opt_goals, self.optimization_goal)  # sign is needed because SigOpt maximizes
            )

            # Update current evaluation time and function evaluations
            self._update_optimizer_metrics()

            if verbosity:
                self._print_evaluation(getattr(opt_goals, self.optimization_goal))

        return self._get_best_evaluation()

    def sigopt_objective_function(self, problem: HyperparameterOptimizationProblem, arm_dict: Assignments) \
            -> Tuple[Evaluator, OptimizationGoals]:

        def apply_logarithms() -> Assignments:
            for p_name in list(arm_dict.keys()):  # TODO: ask Jonathan about this without list, I got an error
                if p_name[:8] == "int_log_":
                    arm_dict[p_name[8:]] = round(np.exp(arm_dict[p_name]))
                elif p_name[:4] == "log_":
                    arm_dict[p_name[4:]] = np.exp(arm_dict[p_name])
            return arm_dict

        arm_dict = apply_logarithms()  # Apply transformations to log params
        arm = Arm(**arm_dict)          # create Arm from arm_dict for hyperparams that we want to optimize
        # hyperparameters that do not need to be optimized should be added to the Arm with their default values
        arm.set_default_values(domain=problem.domain, hyperparams_to_opt=problem.hyperparams_to_opt)

        evaluator = problem.get_evaluator(arm=arm)
        return evaluator, evaluator.evaluate(self.n_resources)
