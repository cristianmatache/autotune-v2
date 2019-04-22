from typing import Dict, Tuple

from core.params import Param


class Arm:

    """ Records hyperparameters and random/default values for each
    Example:
    arm.batch_size = 100
    arm.learning_rate = 0.123
    These attributes are set dynamically in draw_hp_val but, for type consistency reasons,it is recommended to inherit
    from this class and to set the hyperparameters as None attributes (Eg. CNNArm, LogisticRegressionArm)
    """

    def __init__(self, **kwargs: float):
        """ If a custom arm is needed, the hyperparameter values can be set from a dictionary. For example, for all
        informed optimization methods creating Arm this way is required. If one needs random values for each hyperparam,
        please use draw_hp_val.
        :param kwargs: {"hyperparameter_name": hyperparameter value}
        """
        self.__dict__.update(kwargs)

    def set_default_values(self, *, domain: Dict[str, Param], hyperparams_to_opt: Tuple[str, ...]) -> None:
        """ Sets the hyperparameters that appear in the domain but we don't want to optimize to their default values
        if they are not already set
        :param domain: domain of hyperparameters with names, ranges, distributions etc
                       Eg. {'momentum': Param(...), 'learning_rate': Param(...)}
        :param hyperparams_to_opt: hyperparameters to optimize
        """
        for hp_name in domain.keys():
            if not hasattr(self, hp_name) and hp_name not in hyperparams_to_opt:
                # if we do not need to optimize hp_name, set Arm value to its default (if not already set)
                hp_val = domain[hp_name].init_val
                if hp_val is None:
                    raise ValueError(f"No default value for param {hp_name} was supplied")
                setattr(self, hp_name, hp_val)

    def draw_hp_val(self, *, domain: Dict[str, Param], hyperparams_to_opt: Tuple[str, ...]) -> None:
        """ Draws random values for the hyperparameters that we want to optimize
        :param domain: domain of hyperparameters with names, ranges, distributions etc
                       Eg. {'momentum': Param(...), 'learning_rate': Param(...)}
        :param hyperparams_to_opt: hyperparameters to optimize
        """
        self.set_default_values(domain=domain, hyperparams_to_opt=hyperparams_to_opt)
        for hp_name in domain.keys():
            if hp_name in hyperparams_to_opt:  # draw random value if we need to optimize the current hyperparameter
                hp_val = domain[hp_name].get_param_range(1, stochastic=True)[0]
                setattr(self, hp_name, hp_val)

    def __str__(self) -> str:
        """
        :return: human readable string representation of an Arm
        """
        longest_hp = max([len(hp_name) for hp_name in self.__dict__.keys()])

        def padding(hp_name: str) -> int:
            return longest_hp - len(hp_name) + 1

        return '\n'.join(
            [f"   - {hp_name}:{' '*padding(hp_name)}{hp_val}" for hp_name, hp_val in self.__dict__.items()])
