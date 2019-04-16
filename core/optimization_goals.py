

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

    def __str__(self) -> str:
        max_len = max([len(g) for g in self.__dict__.keys()]) + 1
        def padding(word) -> int: return max_len - len(word)
        goals = [f"  {g}{padding(g) * ' '}{val}" for g, val in self.__dict__.items()]
        return "\n".join(["> Optimization goals:", *goals])
