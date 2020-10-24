from dataclasses import dataclass
from types import SimpleNamespace
from typing import KeysView, cast

from autotune.core.params import Param

# ParamType = Union[Param, PairParam, CategoricalParam]
ParamType = Param


@dataclass(init=False, frozen=True)
class Domain(SimpleNamespace):

    """Dictionary wrapper to store the domain of hyperparameters.

    For example:
    self.momentum = Param('momentum', 0.3, 0.999, distrib='uniform', scale='linear')
    """

    def __init__(self, **kwargs: ParamType):  # pylint: disable=useless-super-delegation  # Used for type hints only
        """
        :param kwargs: domain of hyperparameters with names, ranges, distributions etc
                       Eg. {'momentum': Param(...), 'learning_rate': Param(...)}
        """
        super().__init__(**kwargs)

    def __getitem__(self, item: str) -> ParamType:
        """Allow dictionary-like access to attributes.

        That is: instead of getattr(domain, item), one can use domain[item]
        """
        return cast(ParamType, getattr(self, item))

    def hyperparams_names(self) -> KeysView[str]:
        """
        :return: generator over all hyperparameter names from the domain
        """
        return self.__dict__.keys()
