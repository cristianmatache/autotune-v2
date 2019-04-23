from typing import Callable

from core import HyperparameterOptimisationProblem, Optimiser, Evaluation, OptimisationGoals


class RandomOptimiser(Optimiser):

    """ Random search
    Note that in this class we will use the terms "evaluation" and "iteration" interchangeably.
    An evaluation means: trying a combination of hyperparameters (an arm) and getting the validation, test errors
    """

    def __init__(self, n_resources: int, max_iter: int = None, max_time: int = None, min_or_max: Callable = min,
                 optimisation_func: Callable[[OptimisationGoals], float] = Optimiser.default_optimisation_func):
        """
        :param n_resources: number of resources per evaluation (of each arm)
        :param max_iter: max iteration (considered infinity if None) - stopping condition
        :param max_time: max time a user is willing to wait for (considered infinity if None) - stopping condition
        :param min_or_max: min/max (built in functions) - whether to minimize or to maximize the optimization_goal
        :param optimisation_func: function in terms of which to perform optimization (can aggregate several optimization
                                  goals or can just return the value of one optimization goal)
        """
        super().__init__(max_iter, max_time, min_or_max, optimisation_func)
        self.n_resources = n_resources

    def run_optimisation(self, problem: HyperparameterOptimisationProblem, verbosity: bool = False) -> Evaluation:
        """
        :param problem: optimization problem (eg. CIFAR, MNIST, SVHN, MRBI problems)
        :param verbosity: whether to print the results of every single evaluation/iteration
        :return: Evaluation of best arm (evaluator, optimization_goals)
        """
        self._init_optimizer_metrics()

        while not self._needs_to_stop():
            # Draw random sample
            evaluator = problem.get_evaluator()
            opt_goals = evaluator.evaluate(self.n_resources)
            # Evaluate arm on problem
            # Update evaluation history: arms tried so far, validation and test errors so far
            self._update_evaluation_history(evaluator, opt_goals)
            # Update evaluation metrics: time so far, number of evaluations so far, checkpoint times so far
            self._update_optimizer_metrics()

            if verbosity:
                self._print_evaluation(self.optimisation_func(opt_goals))

        return self._get_best_evaluation()
