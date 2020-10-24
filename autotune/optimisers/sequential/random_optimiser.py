from typing import Callable, Optional

from autotune.core import (
    Evaluation, HyperparameterOptimisationProblem, OptimisationGoals, Optimiser, ShapeFamilyScheduler,
    SimulationProblem, optimisation_metric_user)


class RandomOptimiser(Optimiser):

    """Random search Note that in this class we will use the terms "evaluation" and "iteration" interchangeably.

    An evaluation means: trying a combination of hyperparameters (an arm) and getting the validation, test errors
    """

    def __init__(self, n_resources: int, max_iter: int = None, max_time: int = None, min_or_max: Callable = min,
                 optimisation_func: Callable[[OptimisationGoals], float] = Optimiser.default_optimisation_func,
                 is_simulation: bool = False, scheduler: Optional[ShapeFamilyScheduler] = None,
                 plot_simulation: bool = False):
        """
        :param n_resources: number of resources per evaluation (of each arm)
        :param max_iter: max iteration (considered infinity if None) - stopping condition
        :param max_time: max time a user is willing to wait for (considered infinity if None) - stopping condition
        :param min_or_max: min/max (built in functions) - whether to minimize or to maximize the optimisation_goal
        :param optimisation_func: function in terms of which to perform optimisation (can aggregate several optimisation
                                  goals or can just return the value of one optimisation goal)
        :param is_simulation: flag if the problem under optimisation is a real machine learning problem or a simulation
        :param scheduler: if the problem is a simulation, the scheduler provides the parameters for families of shapes
        :param plot_simulation: each simulated loss function will be added to plt.plot, use plt.show() to see results
        """
        super().__init__(max_iter, max_time, min_or_max, optimisation_func, is_simulation, scheduler, plot_simulation)
        self.n_resources = n_resources

    @optimisation_metric_user
    def run_optimisation(self, problem: HyperparameterOptimisationProblem, verbosity: bool = False) -> Evaluation:
        """
        :param problem: optimisation problem (eg. CIFAR, MNIST, SVHN, MRBI problems)
        :param verbosity: whether to print the results of every single evaluation/iteration
        :return: Evaluation of best arm (evaluator, optimisation_goals)
        """
        while not self._needs_to_stop():
            # Draw random sample
            if not self.is_simulation:
                evaluator = problem.get_evaluator()
            else:  # is simulation
                problem: SimulationProblem
                evaluator = problem.get_evaluator(*self.scheduler.get_family() if self.scheduler else (),
                                                  should_plot=self.plot_simulation)
            opt_goals = evaluator.evaluate(self.n_resources)
            # Evaluate arm on problem
            # Update evaluation history: arms tried so far, validation and test errors so far
            self._update_evaluation_history(evaluator, opt_goals)
            # Update evaluation metrics: time so far, number of evaluations so far, checkpoint times so far
            self._update_optimiser_metrics()

            if verbosity:
                self._print_evaluation(self.optimisation_func(opt_goals))

        return self._get_best_evaluation()
