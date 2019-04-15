from typing import Tuple, Dict

from benchmarks.cifar_problem import CifarProblem, HYPERPARAMS_DOMAIN, DEF_HYPERPARAMETERS_TO_OPTIMIZE
from core.params import Param
from datasets.image_dataset_loaders import MRBILoader


class MrbiProblem(CifarProblem):

    def __init__(self, data_dir: str, output_dir: str,
                 hyperparams_domain: Dict[str, Param] = HYPERPARAMS_DOMAIN,
                 hyperparams_to_opt: Tuple[str, ...] = DEF_HYPERPARAMETERS_TO_OPTIMIZE):
        super().__init__(data_dir, output_dir, MRBILoader, hyperparams_domain, hyperparams_to_opt, in_channels=1)
