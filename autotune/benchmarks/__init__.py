from autotune.benchmarks.cifar_problem import CifarProblem
from autotune.benchmarks.known_loss_fn_problem import KnownFnProblem
from autotune.benchmarks.mnist_problem import MnistProblem
from autotune.benchmarks.mrbi_problem import MrbiProblem
from autotune.benchmarks.opt_function_problem import AVAILABLE_OPT_FUNCTIONS, OptFunctionProblem
from autotune.benchmarks.opt_function_simulation_problem import OptFunctionSimulationProblem
from autotune.benchmarks.svhn_problem import SvhnProblem

__all__ = [
    'CifarProblem', 'MnistProblem', 'MrbiProblem', 'SvhnProblem',
    'OptFunctionSimulationProblem', 'OptFunctionProblem', 'AVAILABLE_OPT_FUNCTIONS', 'KnownFnProblem'
]
