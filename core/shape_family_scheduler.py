from typing import Optional, Tuple
from abc import abstractmethod

from core.arm import Arm

ML_AGGRESSIVENESS_TYPE = NECESSARY_AGGRESSIVENESS_TYPE = UP_SPIKINESS_TYPE = float
SHAPE_FAMILY_TYPE = Tuple[Optional[Arm], ML_AGGRESSIVENESS_TYPE, NECESSARY_AGGRESSIVENESS_TYPE, UP_SPIKINESS_TYPE]
EVAL_PARAMS_TYPE = Tuple[Optional[Arm], ML_AGGRESSIVENESS_TYPE, NECESSARY_AGGRESSIVENESS_TYPE, UP_SPIKINESS_TYPE, int]


class ShapeFamilyScheduler:

    def __init__(self, shape_families: Tuple[SHAPE_FAMILY_TYPE, ...], max_resources: int):
        """
        :param shape_families:
        :param max_resources:
        """
        self.shape_families = shape_families
        self.max_resources = max_resources

    @abstractmethod
    def get_family(self, arm: Optional[Arm] = None) -> EVAL_PARAMS_TYPE:
        pass


class RoundRobinShapeFamilyScheduler(ShapeFamilyScheduler):

    def __init__(self, shape_families: Tuple[SHAPE_FAMILY_TYPE, ...], max_resources: int):
        super().__init__(shape_families, max_resources)
        self.index = 0

    def get_family(self, arm: Optional[Arm] = None) -> EVAL_PARAMS_TYPE:
        """
        :param arm: if not None replaces the default arm that has been supplied in self.shape_families
        :return: parameters ready to be passed to BraninSimulationProblem.get_evaluator
        """
        shape_family = self.shape_families[self.index % len(self.shape_families)]
        default_arm, ml_aggressiveness, necessary_aggressiveness, up_spikiness = shape_family
        arm = arm if arm is not None else default_arm
        self.index += 1
        return arm, ml_aggressiveness, necessary_aggressiveness, up_spikiness, self.max_resources
