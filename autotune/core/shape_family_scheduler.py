from __future__ import annotations
from typing import Optional, Tuple, NamedTuple
from abc import abstractmethod
import random

from autotune.core import Arm


class ShapeFamily(NamedTuple):
    arm: Optional[Arm]
    ml_agg: float            # ML aggressiveness
    necessary_agg: float     # Necessary aggressiveness
    up_spikiness: float      # Up spikiness
    is_smooth: bool = False  # Whether to apply smoothing on the family or not
    start_shift: float = 0   # Shift f(0) downwards (before noise is applied)
    end_shift: float = 200   # Shift f(n) downwards


class EvaluatorParams(NamedTuple):
    """Like ShapeFamily plus max_res and noise, also presents defaults."""
    arm: Optional[Arm]
    ml_agg: float            # ML aggressiveness
    necessary_agg: float     # Necessary aggressiveness
    up_spikiness: float      # Up spikiness
    is_smooth: bool = False  # Whether to apply smoothing on the family or not
    start_shift: float = 0   # Shift f(0) downwards (before noise is applied)
    end_shift: float = 200   # Shift f(n) downwards

    max_res: int = 81        # maximum resources
    noise: float = 0.3       # initial noise


class ShapeFamilyScheduler:

    def __init__(self, shape_families: Tuple[ShapeFamily, ...], max_resources: int, init_noise: float):
        """
        :param shape_families:
        :param max_resources:
        """
        self.shape_families = shape_families
        self.max_resources = max_resources
        self.init_noise = init_noise

    @abstractmethod
    def get_family(self, arm: Optional[Arm] = None) -> EvaluatorParams:
        pass


class RoundRobinShapeFamilyScheduler(ShapeFamilyScheduler):

    def __init__(self, shape_families: Tuple[ShapeFamily, ...], max_resources: int, init_noise: float):
        super().__init__(shape_families, max_resources, init_noise)
        self.index = 0

    def get_family(self, arm: Optional[Arm] = None) -> EvaluatorParams:
        """
        :param arm: if not None replaces the default arm that has been supplied in self.shape_families
        :return: parameters ready to be passed to OptFunctionSimulationProblem.get_evaluator
        """
        shape_family = self.shape_families[self.index % len(self.shape_families)]
        default_arm, *rest_family = shape_family
        arm = arm if arm is not None else default_arm
        self.index += 1
        return EvaluatorParams(arm, *rest_family, self.max_resources, self.init_noise)  # type: ignore


class UniformShapeFamilyScheduler(ShapeFamilyScheduler):

    def get_family(self, arm: Optional[Arm] = None) -> EvaluatorParams:
        """
        :param arm: if not None replaces the default arm that has been supplied in self.shape_families
        :return: parameters ready to be passed to OptFunctionSimulationProblem.get_evaluator
        """
        default_arm, *rest_family = random.choice(self.shape_families)
        arm = arm if arm is not None else default_arm
        return EvaluatorParams(arm, *rest_family, self.max_resources, self.init_noise)  # type: ignore
