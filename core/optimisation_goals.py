from typing import Tuple
from types import SimpleNamespace

from util.frozen_class import frozen_class


@frozen_class
class OptimisationGoals(SimpleNamespace):

    """
    Metrics in terms of which we can perform optimization (individually or by aggregation). That is, the optimizers will
    minimize/maximize one of their attributes or even aggregations (Eg. weighted sum) of attributes, as indicated by
    an optimization function (optimisation_func).
    """

    def __init__(self, **kwargs: float):
        """
        :param validation_error: error on validation set
        :param test_error: error on test set
        :param kwargs: other metrics in terms of which we can perform optimization
        """
        if ("validation_error" not in kwargs) or ("test_error" not in kwargs):
            print("WARNING: validation_error or test_error or both are not included in the Optimization Goals")
        super().__init__(**kwargs)

    def goals_to_str(self, goals_to_print: Tuple[str, ...] = ()) -> str:
        """ Formats  names and values for some/all attributes and formats them into a string
        :param goals_to_print: the attributes that will be formatted to a string, () or None to print all attributes
        :return: string with names and values for some/all attributes
        """
        if not goals_to_print:
            goals_to_print = tuple(self.__dict__.keys())
        goals = {g: self.__dict__[g] for g in goals_to_print if g in self.__dict__}
        try:
            max_len = max([len(g) for g in goals.keys()]) + 1
            def padding(word) -> int: return max_len - len(word)
            goals_strings = [f"  {g}:{padding(g) * ' '}{val}" for g, val in goals.items()]
            return "\n".join(["> Optimization goals:", *goals_strings])
        except ValueError:
            return ""

    def __str__(self) -> str:
        """
        :return: as goals_to_str but always for all attributes
        """
        return self.goals_to_str()
