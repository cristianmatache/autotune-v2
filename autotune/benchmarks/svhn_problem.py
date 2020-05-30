from typing import Tuple

from benchmarks.cifar_problem import CifarProblem, HYPERPARAMS_DOMAIN, HYPERPARAMETERS_TO_OPTIMIZE
from core import Domain
from datasets.image_dataset_loaders import SVHNLoader


class SvhnProblem(CifarProblem):

    """
    Classification on SVHN dataset with a CNN
    """

    def __init__(self, data_dir: str, output_dir: str,
                 hyperparams_domain: Domain = HYPERPARAMS_DOMAIN,
                 hyperparams_to_opt: Tuple[str, ...] = HYPERPARAMETERS_TO_OPTIMIZE):
        """
        :param data_dir: directory where the dataset is stored (or will be downloaded to if not already)
        :param output_dir: directory where to save the arms and their evaluation progress so far (as checkpoints)
        :param hyperparams_domain: names of the hyperparameters of a model along with their domain, that is
                                   ranges, distributions etc. (self.domain)
        :param hyperparams_to_opt: names of hyperparameters to be optimized, if () all params from domain are optimized
        """
        super().__init__(data_dir, output_dir, SVHNLoader, hyperparams_domain, hyperparams_to_opt)
