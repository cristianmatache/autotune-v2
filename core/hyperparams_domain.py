from typing import Union, KeysView
from types import SimpleNamespace

from core.params import Param, PairParam, CategoricalParam
from util.frozen_class import frozen_class

PARAM_TYPE = Union[Param, PairParam, CategoricalParam]


@frozen_class
class Domain(SimpleNamespace):

    """
    Dictionary wrapper to store the domain of hyperparameters
    """

    def __init__(self, **kwargs: PARAM_TYPE):
        """
        :param kwargs: domain of hyperparameters with names, ranges, distributions etc
                       Eg. {'momentum': Param(...), 'learning_rate': Param(...)}
        """
        super().__init__(**kwargs)

    def __getitem__(self, item: str) -> PARAM_TYPE:
        return getattr(self, item)

    def hyperparams_names(self) -> KeysView[str]:
        return self.__dict__.keys()
