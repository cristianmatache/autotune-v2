from autotune.benchmarks.cifar_problem import CifarProblem
from autotune.benchmarks.mnist_problem import MnistProblem
from autotune.benchmarks.mrbi_problem import MrbiProblem
from autotune.benchmarks.svhn_problem import SvhnProblem
from autotune.benchmarks.opt_function_simulation_problem import OptFunctionSimulationProblem
from autotune.benchmarks.opt_function_problem import OptFunctionProblem, AVAILABLE_OPT_FUNCTIONS
from autotune.benchmarks.known_loss_fn_problem import KnownFnProblem

__all__ = [
    'CifarProblem', 'MnistProblem', 'MrbiProblem', 'SvhnProblem',
    'OptFunctionSimulationProblem', 'OptFunctionProblem', 'AVAILABLE_OPT_FUNCTIONS', 'KnownFnProblem'
]
