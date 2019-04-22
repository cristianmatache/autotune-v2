from typing import Union
from types import SimpleNamespace

from core.params import Param, PairParam, CategoricalParam


class Domain(SimpleNamespace):

    """
    Dictionary wrapper to store the domain of hyperparameters
    """

    def __init__(self, **kwargs: Union[Param, PairParam, CategoricalParam]):
        """
        :param kwargs: domain of hyperparameters with names, ranges, distributions etc
                       Eg. {'momentum': Param(...), 'learning_rate': Param(...)}
        """
        super().__init__(**kwargs)

