from abc import abstractmethod
from pprint import PrettyPrinter
from typing import Dict, Tuple

from benchmarks.data.dataset_loader import DatasetLoader
from benchmarks.evaluator import Evaluator
from core.params import Param


class HyperparameterOptimizationProblem:

    def __init__(self, hyperparams_domain: Dict[str, Param], dataset_loader: DatasetLoader,
                 hyperparams_to_opt: Tuple[str, ...] = ()):
        self.domain = hyperparams_domain
        self.hyperparams_to_opt = hyperparams_to_opt if hyperparams_to_opt else list(self.domain.keys())
        print(f"\n> Hyperparameters to optimize:\n    {'' if hyperparams_to_opt else 'ALL:'} {self.hyperparams_to_opt}")

        self.dataset_loader = dataset_loader

    def print_domain(self):
        print(f"\n> Problem {type(self).__name__} hyperparameters domain:")
        PrettyPrinter(indent=4).pprint(self.domain)

    @abstractmethod
    def get_evaluator(self) -> Evaluator:
        pass
