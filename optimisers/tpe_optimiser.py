import time
from functools import partial
from hyperopt import fmin, tpe, Trials, STATUS_OK
from typing import Callable, Dict, Union

from core.optimiser import Optimiser
from core.arm import Arm
from core.problem_def import HyperparameterOptimizationProblem


class TpeOptimiser(Optimiser):

    def __init__(self, n_resources: int, max_iter: int = None, max_time: int = None,
                 optimization_goal: str = "test_error", min_or_max: Callable = min):
        if min_or_max == max:
            raise ValueError("TPE supports minimization only, if you need maximization please "
                             "use minimization on the negative optimization goal")
        super().__init__(max_iter, max_time, optimization_goal, min_or_max)
        self.n_resources = n_resources
        self.name = "TPE"

    def run_optimization(self, problem: HyperparameterOptimizationProblem, verbosity: bool = False) \
            -> Dict[str, Union[Arm, float]]:
        self._init_optimizer_metrics()

        # Wrap parameter space
        param_space = problem.get_hyperopt_space_from_hyperparams_to_opt()

        # Run optimiser
        trials = Trials()
        best = fmin(lambda arm_dict: self.tpe_objective_function(arm_dict, problem, self.n_resources), param_space,
                    max_evals=self.max_iter, algo=partial(tpe.suggest, n_startup_jobs=10), trials=trials,
                    verbose=verbosity)
        # max_time=self.max_time, if self._needs_to_stop() kill fmin

        # Compute statistics
        for t in trials.trials:
            used_arm = Arm(**t['misc']['vals'])
            self._update_evaluation_history(used_arm, **t['result'])
            self.checkpoints.append(t['result']['eval_time'] - self.time_zero)

        return self.min_or_max(self.eval_history, key=lambda x: x[self.optimization_goal])

    def tpe_objective_function(self, arm_dict: Dict[str, float], problem: HyperparameterOptimizationProblem,
                               n_resources: int) -> Dict[str, float]:
        """
        :param arm_dict: values for each hyperparameters to optimized populated by hyperopt TPE
        :param problem: eg. MnistProblem (provides an evaluator)
        :param n_resources: number of resources allocated to the evaluator
        :return:
        """
        arm = Arm(**arm_dict)  # set values for hyperparams that we want to optimize from arm_dict (fed by hyperopt TPE)
        # hyperparameters that do not need to be optimized should be added to the Arm with their default values
        arm.set_default_values(domain=problem.domain, hyperparams_to_opt=problem.hyperparams_to_opt)

        evaluator = problem.get_evaluator(arm=arm)
        opt_goals = evaluator.evaluate(n_resources)
        return {
            'loss': getattr(opt_goals, self.optimization_goal),  # TPE will minimize with respect to the value of 'loss'
            'status': STATUS_OK,                                 # mandatory for Hyperopt
            **opt_goals.__dict__,
            'eval_time': time.time()
        }
