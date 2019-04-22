from typing import NamedTuple

from core.evaluator import Evaluator
from core.optimization_goals import OptimizationGoals


class Evaluation(NamedTuple):

    """
    The result of one evaluation:
        i.e. evaluating an evaluator based on an arm
    """

    evaluator: Evaluator  # Note that the evaluator contains the arm it evaluated
    optimization_goals: OptimizationGoals
