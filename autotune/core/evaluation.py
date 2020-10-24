from typing import NamedTuple

from autotune.core.evaluator import Evaluator
from autotune.core.optimisation_goals import OptimisationGoals


class Evaluation(NamedTuple):
    """The result of one evaluation:

    i.e. evaluating an evaluator based on an arm - evaluator.evaluate()
    """

    evaluator: Evaluator  # Note that the evaluator contains the arm it evaluated
    optimisation_goals: OptimisationGoals
