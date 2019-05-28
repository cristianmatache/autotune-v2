import pickle
import argparse
from argparse import Namespace
from os.path import join as join_path

# Optimisers
from core import Optimiser
from optimisers import HybridHyperbandTpeOptimiser, HyperbandOptimiser, RandomOptimiser, SigOptimiser, TpeOptimiser, \
    HybridHyperbandSigoptOptimiser

# Problems
from core import HyperparameterOptimisationProblem, OptimisationGoals
from benchmarks import MnistProblem, CifarProblem, SvhnProblem, MrbiProblem, BraninProblem

# Set random seeds
import random
import torch


INPUT_DIR = "D:/datasets/"
OUTPUT_DIR = "D:/datasets/output"

N_RESOURCES = 3
MAX_TIME = None
MAX_ITER = 27
ETA = 3

PROBLEM = "branin"
METHOD = "hb+sigopt"
MIN_OR_MAX = "min"
RANDOM_SEED = 42


def optimisation_func(opt_goals: OptimisationGoals) -> float:
    """validation_error"""
    return opt_goals.validation_error


def optimisation_func_branin(opt_goals: OptimisationGoals) -> float:
    """fval"""
    return opt_goals.fval


def _get_args() -> Namespace:
    parser = argparse.ArgumentParser(description='Running optimisations')
    parser.add_argument('-i', '--input-dir', default=INPUT_DIR, type=str, help='input dir')
    parser.add_argument('-o', '--output-dir', default=OUTPUT_DIR, type=str, help='output dir')
    parser.add_argument('-time', '--max-time', default=MAX_TIME, type=int, help='max time (stop if exceeded)')
    parser.add_argument('-iter', '--max-iter', default=MAX_ITER, type=int, help='max iterations (stop if exceeded')
    parser.add_argument('-p', '--problem', default=PROBLEM, type=str, help='problem (eg. cifar, mnist, svhn)')
    parser.add_argument('-m', '--method', default=METHOD, type=str, help='method (eg. random, hyperband, tpe)')
    parser.add_argument('-opt', '--min-or-max', default=MIN_OR_MAX, type=str, help="min or max")
    parser.add_argument('-res', '--n-resources', default=N_RESOURCES, type=int, help='n_resources', required=False)
    parser.add_argument('-eta', default=ETA, type=int, help='halving rate for Hyperband', required=False)
    parser.add_argument('-s', '--seed', default=RANDOM_SEED, type=int, help='Random seed for all frameworks', required=False)
    arguments = parser.parse_args()
    print(f"""\n
    Input directory:  {arguments.input_dir}
    Output directory: {arguments.output_dir}

    Problem:          {arguments.problem.upper()}
    Method:           {arguments.method.upper()}

    Func {arguments.min_or_max.upper()}imizes:   {optimisation_func.__doc__}
    """)
    return arguments


def get_problem(arguments: Namespace) -> HyperparameterOptimisationProblem:
    problem_name = arguments.problem.lower()
    if problem_name == "cifar":
        problem_instance = CifarProblem(arguments.input_dir, arguments.output_dir)
    elif problem_name == "mnist":
        problem_instance = MnistProblem(arguments.input_dir, arguments.output_dir)
    elif problem_name == "svhn":
        problem_instance = SvhnProblem(arguments.input_dir, arguments.output_dir)
    elif problem_name == "mrbi":
        problem_instance = MrbiProblem(arguments.input_dir, arguments.output_dir)
    elif problem_name == "branin":
        problem_instance = BraninProblem()
        optimisation_func = optimisation_func_branin
    else:
        raise ValueError(f"Supplied problem {problem_name} does not exist")
    problem_instance.print_domain()
    return problem_instance


def get_optimiser() -> Optimiser:
    method = args.method.lower()
    min_or_max = min if args.min_or_max == 'min' else max

    if method == "random":
        return RandomOptimiser(n_resources=args.n_resources, max_iter=args.max_iter, max_time=args.max_time,
                               min_or_max=min_or_max, optimisation_func=optimisation_func)
    elif method == "hyperband":
        return HyperbandOptimiser(eta=args.eta, max_iter=args.max_iter, max_time=args.max_time, min_or_max=min_or_max,
                                  optimisation_func=optimisation_func)
    elif method == "tpe":
        return TpeOptimiser(n_resources=args.n_resources, max_iter=args.max_iter, max_time=args.max_time,
                            min_or_max=min_or_max, optimisation_func=optimisation_func)
    elif method == "hb+tpe":
        return HybridHyperbandTpeOptimiser(eta=args.eta, max_iter=args.max_iter, max_time=args.max_time,
                                           min_or_max=min_or_max, optimisation_func=optimisation_func)
    elif method == "sigopt":
        return SigOptimiser(n_resources=args.n_resources, max_iter=args.max_iter, max_time=args.max_time,
                            min_or_max=min_or_max, optimisation_func=optimisation_func)
    elif method == "hb+sigopt":
        return HybridHyperbandSigoptOptimiser(eta=args.eta, max_iter=args.max_iter, max_time=args.max_time,
                                              min_or_max=min_or_max, optimisation_func=optimisation_func)
    else:
        raise ValueError(f"Supplied problem {method} does not exist")


if __name__ == "__main__":
    args = _get_args()

    # FIXME move framework specifics under corresponding problems
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    problem = get_problem(args)
    optimiser = get_optimiser()

    print(optimiser)
    optimum_evaluation = optimiser.run_optimisation(problem, verbosity=True)
    print(f"Best hyperparams:\n{optimum_evaluation.evaluator.arm}\n"
          f"with:\n"
          f"  - {optimisation_func.__doc__}: {optimisation_func(optimum_evaluation.optimisation_goals)}\n"
          f"Total time:\n  - {optimiser.checkpoints[-1]} seconds")

    output_file_path = join_path(args.output_dir, f"results-{args.problem}-{args.method}.pkl")
    with open(output_file_path, 'wb') as f:
        pickle.dump((optimum_evaluation, optimiser.eval_history, optimiser.checkpoints), f)
