import time
from colorama import Fore, Style
from typing import Dict, Union

from core.problem_def import HyperparameterOptimizationProblem
from core.arm import Arm
from core.optimiser import Optimiser


class RandomOptimiser(Optimiser):

    """ Note that in this class we will use the terms "evaluation" and "iteration" interchangeably.
    An evaluation means: trying a combination of hyperparameters (an arm) and getting the validation, test errors
    """

    def __init__(self, n_resources: int, max_iter: int = None, max_time: int = None):
        super().__init__(n_resources, max_iter, max_time)
        self.name = "Random"
        self.eval_history = []

    def _init_optimization_metrics(self) -> None:
        self.time_zero = time.time()  # start time of optimization
        self.cum_time = 0             # cumulative time = time of last evaluation/iteration - start time
        self.num_iterations = 0       #
        self.checkpoints = []         # list of cum_times of successful evaluations/iterations so far

    def _needs_to_stop(self) -> bool:
        def _is_time_over(): return self.max_time <= self.cum_time
        def _exceeded_iterations(): return self.max_iter <= self.num_iterations
        if _is_time_over():
            print("\nFINISHED: Time limit exceeded")
        elif _exceeded_iterations():
            print("\nFINISHED: Exceeded maximum number of iterations")
        return _is_time_over() or _exceeded_iterations()

    def _update_evaluation_history(self, arm: Arm, val_error: float, test_error: float) -> None:
        self.eval_history.append({
            'arm': arm,                     # combinations of hyperparameters (arms) tried so far
            'validation_error': val_error,  # validation errors so far
            'test_error': test_error        # test errors so far
        })

    def _update_evaluation_metrics(self) -> None:
        self.cum_time = time.time() - self.time_zero
        self.num_iterations += 1
        self.checkpoints.append(self.cum_time)

    def run_optimization(self, problem: HyperparameterOptimizationProblem, verbosity: bool = False) \
            -> Dict[str, Union[Arm, float]]:
        self._init_optimization_metrics()

        while not self._needs_to_stop():
            # Draw random sample
            evaluator = problem.get_evaluator()
            val_error, test_error = evaluator.evaluate(self.n_resources)
            # Evaluate arm on problem
            # Update evaluation history: arms tried so far, validation and test errors so far
            self._update_evaluation_history(evaluator.arm, val_error, test_error)
            # Update evaluation metrics: time so far, number of evaluations so far, checkpoint times so far
            self._update_evaluation_metrics()

            if verbosity:
                self._print_evaluation(test_error)

        return min(self.eval_history, key=lambda x: x['test_error'])

    def _print_evaluation(self, test_error: float) -> None:
        num_spaces = 8
        best_test_error_so_far = min([x['test_error'] for x in self.eval_history])
        print(f"{Fore.GREEN if test_error == best_test_error_so_far else Fore.RED}"
              f"\n> SUMMARY: iteration_number: {self.num_iterations},{num_spaces * ' '}"
              f"time_elapsed: {self.cum_time:.2f}s,{num_spaces * ' '}"
              f"current_test_error: {test_error:.5f},{num_spaces * ' '}"
              f" best_test_error_so_far: {best_test_error_so_far:.5f}"
              f"{Style.RESET_ALL}")
