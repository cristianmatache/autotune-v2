from dataclasses import dataclass, field
from queue import Queue
from typing import Type, List, Callable, Optional, Union

import mpmath
from colorama import Fore, Style
from flask import Flask

from autotune.core import Optimiser, HyperparameterOptimisationProblem, Evaluation, OptimisationGoals, \
    ShapeFamilyScheduler, ShapeFamily, UniformShapeFamilyScheduler
from autotune.optimisers.sequential.tpe_optimiser import TpeOptimiser
from autotune.optimisers.sequential.sigopt_optimiser import SigOptimiser
from autotune.optimisers.sequential.random_optimiser import RandomOptimiser
from autotune.optimisers.parallel.block import Block
from autotune.optimisers.parallel.worker_server import Worker

COL = Fore.MAGENTA
END = Style.RESET_ALL


@dataclass
class Master:
    n_workers: int
    eta: int
    sampler: Type[Union[TpeOptimiser, SigOptimiser, RandomOptimiser]]
    max_iter: Optional[int] = None
    max_time: Optional[int] = None
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


app = Flask(__name__)


MASTER: Master
PROBLEM: HyperparameterOptimisationProblem


@app.route('/set-master/n-workers=<n_workers>/eta=<eta>/max-iter=<max_iter>/sampler=<sampler>')
def set_master(n_workers: str = '2', eta: str = '3', max_iter: str = '81', _: str = ...) -> None:
    from autotune.benchmarks import OptFunctionSimulationProblem
    global PROBLEM
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
    global MASTER
    MASTER = Master(n_workers=int(n_workers), eta=int(eta), sampler=TpeOptimiser, max_iter=int(max_iter),
                    min_or_max=min, is_simulation=True,
                    scheduler=UniformShapeFamilyScheduler(families_of_shapes_general, max_resources=81, init_noise=10))


if __name__ == '__main__':
    INPUT_DIR = "D:/workspace/python/datasets/"
    OUTPUT_DIR = "D:/workspace/python/datasets/output"
    from autotune.benchmarks import MnistProblem
    PROBLEM = MnistProblem(INPUT_DIR, OUTPUT_DIR)

    MASTER.run_optimisation(PROBLEM)
