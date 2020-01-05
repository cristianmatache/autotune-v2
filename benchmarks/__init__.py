from benchmarks.cifar_problem import CifarProblem
from benchmarks.mnist_problem import MnistProblem
from benchmarks.mrbi_problem import MrbiProblem
from benchmarks.svhn_problem import SvhnProblem
from benchmarks.opt_function_simulation_problem import OptFunctionSimulationProblem
from benchmarks.opt_function_problem import OptFunctionProblem, AVAILABLE_OPT_FUNCTIONS
from benchmarks.known_loss_fn_problem import KnownFnProblem

__all__ = [
    'CifarProblem', 'MnistProblem', 'MrbiProblem', 'SvhnProblem',
    'OptFunctionSimulationProblem', 'OptFunctionProblem', 'AVAILABLE_OPT_FUNCTIONS', 'KnownFnProblem'
]
