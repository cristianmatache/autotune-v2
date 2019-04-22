import time
from functools import partial
from hyperopt import fmin, tpe, Trials, STATUS_OK
from typing import Callable, Dict, Union

from core.optimiser import Optimiser, Evaluation
from core.arm import Arm
from core.problem_def import HyperparameterOptimizationProblem
from core.evaluator import Evaluator
from core.optimization_goals import OptimizationGoals


class TpeOptimiser(Optimiser):

    def __init__(self, n_resources: int, max_iter: int = None, max_time: int = None,
                 optimization_goal: str = "test_error", min_or_max: Callable = min):
        """ TPE Bayesian optimization
        :param n_resources: number of resources per evaluation (of each arm)
        :param max_iter: max iteration (considered infinity if None)
        :param max_time: max time a user is willing to wait for (considered infinity if None)
        :param optimization_goal: what part of the OptimizationGoals the Optimiser will minimize/maximize eg. test_error
        :param min_or_max: min/max (built in functions) - whether to minimize or to maximize the optimization_goal
        """
        super().__init__(max_iter, max_time, optimization_goal, min_or_max)

        # TPE Hyperopt supports minimization only, so if the problem is maximization, minimize -1 * optimization goal
        self.sign = -1 if min_or_max == max else 1
        self.n_resources = n_resources

    def run_optimization(self, problem: HyperparameterOptimizationProblem, verbosity: bool = False) -> Evaluation:
        self._init_optimizer_metrics()

        # Wrap parameter space
        param_space = problem.get_hyperopt_space_from_hyperparams_to_opt()

        # Run TPE
        trials = Trials()
        fmin(lambda arm_dict: self.tpe_objective_function(arm_dict, problem), param_space,
             max_evals=self.max_iter, algo=partial(tpe.suggest, n_startup_jobs=10), trials=trials, verbose=verbosity)

        # Compute statistics
        for t in trials.trials:
            self._update_evaluation_history(t["result"]["evaluator"], t["result"]["optimization_goals"])
            self.checkpoints.append(t['result']['eval_time'] - self.time_zero)

        return self._get_best_evaluation()

    def tpe_objective_function(self, arm_dict: Dict[str, float], problem: HyperparameterOptimizationProblem)\
            -> Dict[str, Union[float, Evaluator, OptimizationGoals]]:
        """
        :param arm_dict: values for each hyperparameters to optimized populated by hyperopt TPE
        :param problem: eg. MnistProblem (provides an evaluator)
        :return: dictionary that must contain keys 'loss' and 'status' (as required by Hyperopt) plus any additional
        information that we may require after evaluation
        """
        arm = Arm(**arm_dict)  # set values for hyperparams that we want to optimize from arm_dict (fed by hyperopt TPE)
        # hyperparameters that do not need to be optimized should be added to the Arm with their default values
        arm.set_default_values(domain=problem.domain, hyperparams_to_opt=problem.hyperparams_to_opt)

        evaluator = problem.get_evaluator(arm=arm)
        opt_goals = evaluator.evaluate(self.n_resources)
        return {
            # TPE will minimize with respect to the value of 'loss'
            'loss': self.sign * getattr(opt_goals, self.optimization_goal),
            'status': STATUS_OK,             # mandatory for Hyperopt
            'eval_time': time.time(),        # timestamp when evaluation is finished
            'evaluator': evaluator,
            'optimization_goals': opt_goals
        }
