from dataclasses import dataclass
from typing import Sequence, Optional

from colorama import Fore, Style

from autotune.core import Evaluation

COL = Fore.MAGENTA
END = Style.RESET_ALL


@dataclass
class Block:
    """To be consumed by workers. Corresponds to (Ni, Ri) pairs of a bracket."""
    bracket: int
    i: int
    max_i: int
    n_i: int
    r_i: int
    evaluations: Optional[Sequence[Evaluation]] = None  # one evaluator for each arm
