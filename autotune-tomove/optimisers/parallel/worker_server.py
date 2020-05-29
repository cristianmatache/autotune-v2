from dataclasses import dataclass
from typing import Type, List, Callable, Optional, Union

from colorama import Fore, Style
from flask import Flask
import dill

from core import HyperparameterOptimisationProblem, Evaluation, OptimisationGoals, ShapeFamilyScheduler
from optimisers import TpeOptimiser, SigOptimiser, RandomOptimiser
from optimisers.parallel.block import Block

COL = Fore.MAGENTA
END = Style.RESET_ALL


@dataclass
class Worker:
    eta: int
    optimisation_func: Callable[[OptimisationGoals], float]
    sampler: Type[Union[TpeOptimiser, SigOptimiser, RandomOptimiser]]
    min_or_max: Callable = min
    is_simulation: bool = False
    scheduler: Optional[ShapeFamilyScheduler] = None

    def _get_optimisation_func_val(self, evaluation: Evaluation) -> float:
        return self.optimisation_func(evaluation.optimisation_goals)

    def _get_best_n_evaluations(self, n: int, evaluations: List[Evaluation]) -> List[Evaluation]:
        """ Note that for minimization we sort in ascending order while for maximization we sort in descending order by
        the value of the optimisation_func applied on evaluations
        :param n: number of top "best evaluators" to retrieve
        :param evaluations: A list of ordered pairs (evaluator, result of evaluator's evaluate() method)
        :return: best n evaluators (those evaluators that gave the best n values on self.optimisation_goal)
        """
        is_descending = self.min_or_max == max
        sorted_evaluations_by_res = sorted(evaluations, key=self._get_optimisation_func_val, reverse=is_descending)
        return sorted_evaluations_by_res[:n]

    def consume_block(self, block: Block, problem: HyperparameterOptimisationProblem) -> Block:
        if block.evaluations is None:
            optimiser = self.sampler(
                n_resources=block.r_i, max_iter=block.n_i, optimisation_func=self.optimisation_func,
                min_or_max=self.min_or_max, is_simulation=self.is_simulation, scheduler=self.scheduler
            )
            optimiser.run_optimisation(problem, verbosity=True)
            evaluations = optimiser.eval_history
        else:
            evaluations = [Evaluation(e.evaluator, e.evaluator.evaluate(n_resources=block.r_i))
                           for e in block.evaluations]
        print(f"{COL}** {'Generated' if block.evaluations is None else 'Evaluated'}: {block} {END}")
        return Block(
            bracket=block.bracket,
            i=block.i+1,
            max_i=block.max_i,
            n_i=int(block.n_i/self.eta),  # is this okay?
            r_i=block.r_i*self.eta,
            evaluations=self._get_best_n_evaluations(int(block.n_i / self.eta), evaluations)
        )


WORKER: Worker


app = Flask(__name__)


@app.route('/init')
def init() -> None:
    global WORKER
    WORKER = dill.load()
