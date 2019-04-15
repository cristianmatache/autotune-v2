from abc import abstractmethod
from pprint import PrettyPrinter
from typing import Dict, Tuple

from datasets.dataset_loader import DatasetLoader
from benchmarks.torch_evaluator import Evaluator
from core.params import Param


class HyperparameterOptimizationProblem:

    def __init__(self, hyperparams_domain: Dict[str, Param], dataset_loader: DatasetLoader,
                 hyperparams_to_opt: Tuple[str, ...] = ()):
        """
        :param hyperparams_domain: names of the hyperparameters of a model along with their domain, that is
                                   ranges, distributions etc. (self.domain)
        :param dataset_loader: loads a dataset
        :param hyperparams_to_opt: names of hyperparameters to be optimized, if () all params from domain are optimized
        """
        self.domain = hyperparams_domain
        self.hyperparams_to_opt = hyperparams_to_opt if hyperparams_to_opt else list(self.domain.keys())
        print(f"\n> Hyperparameters to optimize:\n    {'' if hyperparams_to_opt else 'ALL:'} {self.hyperparams_to_opt}")

        self.dataset_loader = dataset_loader

    def print_domain(self) -> None:
        """ Pretty prints the domain of the problem
        """
        print(f"\n> Problem {type(self).__name__} hyperparameters domain:")
        PrettyPrinter(indent=4).pprint(self.domain)

    @abstractmethod
    def get_evaluator(self) -> Evaluator:
        """ An evaluator must:
        - generate random arm(s)
        - build the model based on this arm
        - train it
        - report the model's performance (eg. test_error)
        :return: evaluator
        """
        pass
