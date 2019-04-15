import pickle
import argparse
from os.path import join as join_path

from core.optimiser import Optimiser
from optimisers.random_optimizer import RandomOptimiser

from core.problem_def import HyperparameterOptimizationProblem
from benchmarks.mnist_problem import MnistProblem
from benchmarks.cifar_problem import CifarProblem
from benchmarks.svhn_problem import SvhnProblem
from benchmarks.mrbi_problem import MrbiProblem

INPUT_DIR = "D:/datasets/"
OUTPUT_DIR = "D:/datasets/output"

N_RESOURCES = 3
MAX_TIME = None
MAX_ITER = 2

PROBLEM = "mnist"
METHOD = "random"


def _get_args():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-i', '--input_dir', default=INPUT_DIR, type=str, help='input dir')
    parser.add_argument('-o', '--output_dir', default=OUTPUT_DIR, type=str, help='output dir')
    parser.add_argument('-res', '--n_resources', default=N_RESOURCES, type=int, help='n_resources')
    parser.add_argument('-time', '--max_time', default=MAX_TIME, type=int, help='max time (stop if exceeded)')
    parser.add_argument('-iter', '--max_iter', default=MAX_ITER, type=int, help='max iterations (stop if exceeded')
    parser.add_argument('-p', '--problem', default=PROBLEM, type=str, help='problem (eg. cifar, mnist, svhn)')
    parser.add_argument('-m', '--method', default=METHOD, type=str, help='method (eg. random, hyperband)')
    arguments = parser.parse_args()
    print(f"""\n> ARGS random run:
    Input directory:  {arguments.input_dir}
    Output directory: {arguments.output_dir}

    Problem:          {arguments.problem.upper()}
    Method:           {arguments.method.upper()}
    """)
    return arguments


def get_problem() -> HyperparameterOptimizationProblem:
    problem_class = {
        "cifar": CifarProblem,
        "mnist": MnistProblem,
        "svhn": SvhnProblem,
        "mrbi": MrbiProblem
    }[args.problem.lower()]
    problem_instance = problem_class(args.input_dir, args.output_dir)
    problem_instance.print_domain()
    return problem_instance


def get_optimiser() -> Optimiser:
    optimiser_class = {
        "random": RandomOptimiser
    }[args.method.lower()]
    # Define maximum units of resource assigned to each optimisation iteration
    optimiser_instance = optimiser_class(n_resources=args.n_resources, max_iter=args.max_iter, max_time=args.max_time)
    return optimiser_instance


if __name__ == "__main__":
    args = _get_args()
    problem = get_problem()
    optimiser = get_optimiser()

    print(optimiser)
    optimum = optimiser.run_optimization(problem, verbosity=True)
    print(f"Best hyperparams:\n{optimum['arm']}\nwith\n  - test error: {optimum['test_error']}\n"
          f"  - validation error: {optimum['validation_error']}")

    output_file_path = join_path(args.output_dir, "results.pkl")
    with open(output_file_path, 'wb') as f:
        pickle.dump([optimiser], f)
