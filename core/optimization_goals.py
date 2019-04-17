from typing import Tuple


class OptimizationGoals:

    """
    Metrics in terms of which we can perform optimization, that is, the optimisers will minimize one of its attributes
    """

    def __init__(self, validation_error: float, test_error: float, **kwargs: float):
        """
        :param validation_error: error on validation set
        :param test_error: error on test set
        :param kwargs: other metrics in terms of which we can perform optimization
        """
        self.validation_error = validation_error
        self.test_error = test_error
        self.__dict__.update(kwargs)

    def goals_to_str(self, goals_to_print: Tuple[str] = ()) -> str:
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
