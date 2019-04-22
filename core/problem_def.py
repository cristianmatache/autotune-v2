from abc import abstractmethod
from pprint import PrettyPrinter
from typing import Dict, Tuple, List, Union, Optional
from hyperopt import hp
from hyperopt.pyll import Apply
import numpy as np

from datasets.dataset_loader import DatasetLoader
from core.evaluator import Evaluator
from core.params import Param
from core.arm import Arm
from core.hyperparams_domain import Domain


class HyperparameterOptimizationProblem:

    __slots__ = ("domain", "hyperparams_to_opt", "dataset_loader", "output_dir")

    def __init__(self, hyperparams_domain: Domain, hyperparams_to_opt: Tuple[str, ...] = (),
                 dataset_loader: Optional[DatasetLoader] = None, output_dir: Optional[str] = None):
        """
        :param hyperparams_domain: names of the hyperparameters of a model along with their domain, that is
                                   ranges, distributions etc. (self.domain)
        :param hyperparams_to_opt: names of hyperparameters to be optimized
                                   if set to () all params from domain are optimized
        :param dataset_loader: loads a dataset (e.g. CIFARLoader)
        :param output_dir: directory where to save the arms and their evaluation progress so far (as checkpoints)
        """
        self.domain = hyperparams_domain
        if hyperparams_to_opt:  # if any hyperparams_to_opt are given
            self.hyperparams_to_opt = tuple(set(hyperparams_to_opt) & set(self.domain.hyperparams_names()))
        else:                   # if no hyperparams_to_opt are given, optimize all from domain
            self.hyperparams_to_opt = tuple(self.domain.hyperparams_names())
        print(f"\n> Hyperparameters to optimize:\n    {'' if hyperparams_to_opt else 'ALL:'} {self.hyperparams_to_opt}")

        self.dataset_loader = dataset_loader
        self.output_dir = output_dir

    def print_domain(self) -> None:
        """ Pretty prints the domain of the problem
        """
        print(f"\n> Problem {type(self).__name__} hyperparameters domain:")
        PrettyPrinter(indent=4).pprint(self.domain.__dict__)

    @abstractmethod
    def get_evaluator(self, arm: Arm = None) -> Evaluator:
        """ An evaluator must:
        - generate random arm(s) if None is given
        - build the model based on this arm
        - train it
        - report the model's performance (eg. test_error)
        :return: evaluator
        """
        pass

    def get_hyperopt_space_from_hyperparams_to_opt(self) -> Dict[str, Apply]:
        """ Converts the problem's domain to hyperopt format
        :return: dict {hyperparameter_name: Apply (hyperparameter space from hyperopt)}
        """

        def convert_to_hyperopt(param: Param) -> Apply:
            if param.scale == "log":
                assert param.logbase == np.e
                if param.interval:
                    return hp.qloguniform(param.name, param.min_val, param.max_val, param.interval)
                else:
                    return hp.loguniform(param.name, param.min_val, param.max_val)
            else:
                if param.interval:
                    return hp.quniform(param.name, param.min_val, param.max_val, param.interval)
                else:
                    return hp.uniform(param.name, param.min_val, param.max_val)

        return {hp_name: convert_to_hyperopt(self.domain[hp_name]) for hp_name in self.hyperparams_to_opt}

    def get_sigopt_space_from_hyperparams_to_opt(self) -> List[Dict[str, Union[str, Dict[str, float]]]]:
        """ Converts the problem's domain to sigopt format
        :return: dict {'name': hyperparam name, 'type': eg. int, 'bounds': {'min': min bound, 'max': max bound} ...}
                 hyperparameter space as required by sigopt
        """

        def convert_to_sigopt(param: Param) -> Dict[str, Union[str, Dict[str, float]]]:
            name = param.name
            if param.scale == "log":
                assert param.logbase == np.e
                name = f"log_{name}"
                if param.interval:
                    name = "int_" + name
            param_type = 'double'
            if param.interval and param.scale != "log":
                param_type = 'int'
            return dict(name=name, type=param_type, bounds=dict(min=param.min_val, max=param.max_val))

        return [convert_to_sigopt(self.domain[hp_name]) for hp_name in self.hyperparams_to_opt]
