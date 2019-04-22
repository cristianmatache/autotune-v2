from typing import Union, KeysView
from types import SimpleNamespace

from core.params import Param, PairParam, CategoricalParam
from util.frozen_class import frozen_class

PARAM_TYPE = Union[Param, PairParam, CategoricalParam]


@frozen_class
class Domain(SimpleNamespace):

    """
    Dictionary wrapper to store the domain of hyperparameters. For example:
    self.momentum = Param('momentum', 0.3, 0.999, distrib='uniform', scale='linear')
    """

    def __init__(self, **kwargs: PARAM_TYPE):
        """
        :param kwargs: domain of hyperparameters with names, ranges, distributions etc
                       Eg. {'momentum': Param(...), 'learning_rate': Param(...)}
        """
        super().__init__(**kwargs)

    def __getitem__(self, item: str) -> PARAM_TYPE:
        """ Allow dictionary-like access to attributes. That is:
        instead of getattr(domain, item), one can use domain[item]
        """
        return getattr(self, item)

    def hyperparams_names(self) -> KeysView[str]:
        """
        :return: generator over all hyperparameter names from the domain
        """
        return self.__dict__.keys()
