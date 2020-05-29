from dataclasses import dataclass, field
from queue import Queue
from typing import Sequence, Type, List, Callable, Optional, Union

import mpmath
from colorama import Fore, Style

from core import Optimiser, HyperparameterOptimisationProblem, Evaluation, OptimisationGoals, ShapeFamilyScheduler, \
    ShapeFamily, UniformShapeFamilyScheduler
from optimisers import TpeOptimiser, SigOptimiser, RandomOptimiser

COL = Fore.MAGENTA
END = Style.RESET_ALL


@dataclass
class Block:
    """To be consumed by workers. Corresponds to (Ni, Ri) pairs of a bracket."""
    bracket: int
    i: int
    max_i: int
    n_i: int
    r_i: int
    evaluations: Optional[Sequence[Evaluation]] = None  # one evaluator for each arm


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


@dataclass
class Master:
    n_workers: int
    eta: int
    sampler: Type[Union[TpeOptimiser, SigOptimiser, RandomOptimiser]]
    max_iter: int = None
    max_time: int = None
    min_or_max: Callable = min
    optimisation_func: Callable[[OptimisationGoals], float] = Optimiser.default_optimisation_func
    is_simulation: bool = False
    scheduler: Optional[ShapeFamilyScheduler] = None
    plot_simulation: bool = False

    _queue: 'Queue[Block]' = field(init=False, default_factory=Queue)
    _workers: List[Worker] = field(init=False)

    def __post_init__(self) -> None:
        self._workers = [Worker(self.eta, self.optimisation_func, self.sampler, self.min_or_max,
                         self.is_simulation, self.scheduler) for _ in range(self.n_workers)]

    def run_optimisation(self, problem: HyperparameterOptimisationProblem) -> None:
        R = self.max_iter  # maximum amount of resource that can be allocated to a single hyperparameter configuration
        eta = self.eta  # halving rate

        def log_eta(x: int) -> int:
            return int(mpmath.log(x) / mpmath.log(eta))

        s_max = log_eta(R)  # number of unique executions of Successive Halving (minus one)
        s_min = 2 if s_max >= 2 else 0  # skip the rest of the brackets after s_min
        B = (s_max + 1) * R  # total/max resources (without reuse) per execution of Successive Halving

        for s in reversed(range(s_min, s_max + 1)):
            n = int(mpmath.ceil(int(B / R / (s + 1)) * eta ** s))  # initial number of evaluators/configurations/arms
            r = R * eta ** (-s)  # initial resources allocated to each evaluator/arm
            self._queue.put(Block(bracket=s, i=1, max_i=s+1, n_i=n, r_i=r))  # initial blocks

        worker = self._workers[0]
        while not self._queue.empty():
            block = self._queue.get()
            new_block = worker.consume_block(block, problem)
            if new_block.i <= new_block.max_i:
                self._queue.put(new_block)
            else:
                best_in_bracket = self.min_or_max(block.evaluations, key=self._get_optimisation_func_val)
                print(f'Finished bracket {block.bracket}:\n{block}\n',
                      best_in_bracket.evaluator.arm, best_in_bracket.optimisation_goals)

    def _get_optimisation_func_val(self, evaluation: Evaluation) -> float:
        return self.optimisation_func(evaluation.optimisation_goals)


if __name__ == '__main__':
    # INPUT_DIR = "D:/workspace/python/datasets/"
    # OUTPUT_DIR = "D:/workspace/python/datasets/output"
    # from benchmarks import MnistProblem
    # PROBLEM = MnistProblem(INPUT_DIR, OUTPUT_DIR)

    from benchmarks import OptFunctionSimulationProblem
    PROBLEM = OptFunctionSimulationProblem('rastrigin')
    families_of_shapes_general = (
        ShapeFamily(None, 1.5, 10, 15, False),  # with aggressive start
        ShapeFamily(None, 0.5, 7, 10, False),  # with average aggressiveness at start and at the beginning
        ShapeFamily(None, 0.2, 4, 7, True),  # non aggressive start, aggressive end
        ShapeFamily(None, 1.5, 10, 15, False, 200, 400),  # with aggressive start
        ShapeFamily(None, 0.5, 7, 10, False, 200, 400),  # with average aggressiveness at start and at the beginning
        ShapeFamily(None, 0.2, 4, 7, True, 200, 400),  # non aggressive start, aggressive end

        # ShapeFamily(None, 0, 1, 0, False, 0, 0),  # flat
    )
    master = Master(n_workers=2, eta=3, sampler=TpeOptimiser, max_iter=81, min_or_max=min, is_simulation=True,
                    scheduler=UniformShapeFamilyScheduler(families_of_shapes_general, max_resources=81, init_noise=10))
    master.run_optimisation(PROBLEM)
