import argparse
from argparse import Namespace
import matplotlib.pyplot as plt
import numpy as np
import pickle
from os.path import join as join_path
from typing import Dict, List
import pandas as pd

# Optimisers
from core import Optimiser, Arm
from optimisers import HybridHyperbandTpeOptimiser, HyperbandOptimiser, RandomOptimiser, SigOptimiser, TpeOptimiser, \
    HybridHyperbandSigoptOptimiser, HybridHyperbandTpeTransferAllOptimiser, HybridHyperbandTpeNoTransferOptimiser, \
    HybridHyperbandTpeTransferLongestOptimiser, HybridHyperbandTpeTransferSameOptimiser

# Problems
from core import HyperparameterOptimisationProblem, OptimisationGoals
from benchmarks import MnistProblem, CifarProblem, SvhnProblem, MrbiProblem, OptFunctionProblem, KnownFnProblem, \
    AVAILABLE_OPT_FUNCTIONS

from util import flatten

KNOWN_FUNCTIONS_DIR = "../../loss_functions/"
INPUT_DIR = "D:/workspace/python/datasets/"
OUTPUT_DIR = "D:/workspace/python/datasets/output"

N_RESOURCES = 18
MAX_TIME = None
MAX_ITER = 6
ETA = 3

PROBLEM = "mnist"
METHOD = "sim(hb+tpe+transfer+none)"
MIN_OR_MAX = "min"

N_SIMULATIONS = 7000

PLOT_EACH = False


def optimisation_func(opt_goals: OptimisationGoals) -> float:
    """fval"""
    return opt_goals.fval


def optimisation_func_opt_function(opt_goals: OptimisationGoals) -> float:
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
    arguments = parser.parse_args()
    print(f"""\n
    Problem:          {arguments.problem.upper()}
    Method:           {arguments.method.upper()}
    Func {arguments.min_or_max.upper()}imizes:   {optimisation_func.__doc__}
    """)
    return arguments


def get_real_problem(arguments: Namespace) -> HyperparameterOptimisationProblem:
    problem_name = arguments.problem.lower()
    optimisation_id = str(pd.Timestamp.utcnow()).replace(':', '-').replace(' ', '-').replace('.', '-').replace('+', '-')
    output_dir = f'{arguments.output_dir}/{problem_name}/{optimisation_id}'
    if problem_name == "cifar":
        problem_instance = CifarProblem(arguments.input_dir, output_dir, dataset_loader=None)
    elif problem_name == "mnist":
        problem_instance = MnistProblem(arguments.input_dir, output_dir)
    elif problem_name == "svhn":
        problem_instance = SvhnProblem(arguments.input_dir, output_dir)
    elif problem_name == "mrbi":
        problem_instance = MrbiProblem(arguments.input_dir, output_dir)
    elif problem_name in AVAILABLE_OPT_FUNCTIONS:
        problem_instance = OptFunctionProblem(problem_name)
        optimisation_func = optimisation_func_opt_function
    else:
        raise ValueError(f"Supplied problem {problem_name} does not exist")
    problem_instance.print_domain()
    return problem_instance


def get_known_functions(arguments: Namespace, known_functions_dir: str = KNOWN_FUNCTIONS_DIR) -> Dict[Arm, List[float]]:
    problem_name = arguments.problem.lower()
    if problem_name == "cifar":
        arm_to_loss_function = {}
        files = [join_path(known_functions_dir, "cifar", f"best_loss_functions-{n}.pkl") for n in range(1, 2)]
        for file in files:
            with open(file, "rb") as f_:
                data = pickle.load(f_)
                for _, evals in data.items():
                    for e in evals:
                        loss_func, arm = e
                        if len(loss_func[1:]) > 300:
                            arm_to_loss_function[arm] = loss_func[1:]
        return arm_to_loss_function
    elif problem_name == "mnist":
        all_evaluators = []
        files = [join_path(known_functions_dir, "mnist", f"results-{PROBLEM}-random-{n}.pkl")
                 for n in (10, 20, 15, 19, 1, 100, 30)] + \
                [join_path(known_functions_dir, "mnist", f"results-{PROBLEM}-tpe-{n}.pkl") for n in (30, 200)]
        for file in files:
            with open(file, "rb") as f_:
                _, eval_history, checkpoints = pickle.load(f_)
                all_evaluators.append([evaluator_t for evaluator_t, _ in eval_history])
        return {e.arm: e.loss_history[1:] for e in flatten(all_evaluators)}
    elif problem_name == "svhn":
        return {}
    elif problem_name == "mrbi":
        return {}
    elif problem_name in AVAILABLE_OPT_FUNCTIONS:
        branin_problem = OptFunctionProblem(problem_name)
        known_fs: Dict[Arm, List[float]] = {}
        for _ in range(1000):
            evaluator = branin_problem.get_evaluator()
            known_fs[evaluator.arm] = [evaluator.evaluate(1).fval for _ in range(arguments.max_iter)]
        return known_fs
    else:
        raise ValueError(f"Supplied problem {problem_name} does not exist")


def get_optimiser() -> Optimiser:
    method = args.method.lower()
    min_or_max = min if args.min_or_max == 'min' else max

    # SIMULATIONS
    if method == "sim(random)":
        return RandomOptimiser(
            n_resources=args.n_resources, max_iter=args.max_iter, max_time=args.max_time, min_or_max=min_or_max,
            optimisation_func=optimisation_func, is_simulation=True, plot_simulation=PLOT_EACH)
    elif method == "sim(hb)":
        return HyperbandOptimiser(
            eta=args.eta, max_iter=args.max_iter, max_time=args.max_time, min_or_max=min_or_max,
            optimisation_func=optimisation_func, is_simulation=True, plot_simulation=PLOT_EACH)
    elif method == "sim(tpe)":
        return TpeOptimiser(
            n_resources=args.n_resources, max_iter=args.max_iter, max_time=args.max_time, min_or_max=min_or_max,
            optimisation_func=optimisation_func, is_simulation=True, plot_simulation=PLOT_EACH)
    elif method == "sim(hb+tpe)":
        return HybridHyperbandTpeOptimiser(
            eta=args.eta, max_iter=args.max_iter, max_time=args.max_time, min_or_max=min_or_max,
            optimisation_func=optimisation_func, is_simulation=True, plot_simulation=PLOT_EACH)
    elif "sim(hb+tpe+transfer" in method:
        hybrid_transfer = {
            "sim(hb+tpe+transfer+none)": HybridHyperbandTpeNoTransferOptimiser,
            "sim(hb+tpe+transfer+all)": HybridHyperbandTpeTransferAllOptimiser,
            "sim(hb+tpe+transfer+longest)": HybridHyperbandTpeTransferLongestOptimiser,
            "sim(hb+tpe+transfer+same)": HybridHyperbandTpeTransferSameOptimiser
        }[method]
        return hybrid_transfer(
            eta=args.eta, max_iter=args.max_iter, max_time=args.max_time, min_or_max=min_or_max,
            optimisation_func=optimisation_func, is_simulation=True, plot_simulation=PLOT_EACH)
    elif method == "sim(sigopt)":
        return SigOptimiser(
            n_resources=args.n_resources, max_iter=args.max_iter, max_time=args.max_time, min_or_max=min_or_max,
            optimisation_func=optimisation_func, is_simulation=True, plot_simulation=PLOT_EACH)
    elif method == "sim(hb+sigopt)":
        return HybridHyperbandSigoptOptimiser(
            eta=args.eta, max_iter=args.max_iter, max_time=args.max_time, min_or_max=min_or_max,
            optimisation_func=optimisation_func, is_simulation=True, plot_simulation=PLOT_EACH)
    else:
        raise ValueError(f"Supplied problem {method} does not exist")


if __name__ == "__main__":
    args = _get_args()
    optimums = []
    known_fns = get_known_functions(args)

    # plt.hist([loss_fn[-1] for arm, loss_fn in known_fns.items()])
    # plt.show()

    for _ in range(N_SIMULATIONS):
        print("********iteration:", _, "avg so far:", np.mean(optimums))
        real_problem = get_real_problem(args)
        problem = KnownFnProblem(known_fs=known_fns, real_problem=real_problem)
        optimiser = get_optimiser()
        optimum_evaluation = optimiser.run_optimisation(problem, verbosity=True)
        print(f"Best hyperparams:\n{optimum_evaluation.evaluator.arm}\n"
              f"with:\n"
              f"  - {optimisation_func.__doc__}: {optimisation_func(optimum_evaluation.optimisation_goals)}\n"
              f"Total time:\n  - {optimiser.checkpoints[-1]} seconds")
        res = optimisation_func(optimum_evaluation.optimisation_goals)
        optimums.append(res)
        if PLOT_EACH:
            plt.show()

    print(f"""\n\n------------- OPTIMUM STATISTICS OVER {N_SIMULATIONS} SIMULATIONS -------------
    average optimum: {np.mean(optimums)}
    sample std dev:  {np.std(optimums, ddof=1)}
    avg top 25%:     {np.mean(sorted(optimums)[:int(N_SIMULATIONS/4)])}
    top quartile:    {sorted(optimums)[:int(N_SIMULATIONS/4)]}
    """)
    print(len(set(optimums)))
    print(set(optimums))

    with open(join_path(OUTPUT_DIR, f"known-fns-hist-{METHOD}-{N_SIMULATIONS}.pkl"), "wb") as f:
        pickle.dump({
            "method": METHOD,

            "n_simulations": N_SIMULATIONS,
            "all_norm_optimums": optimums,

            "avg_optimum": np.mean(optimums),
            "sample_std_dev": np.std(optimums, ddof=1),
            "avg_top_25%": np.mean(sorted(optimums)[:int(N_SIMULATIONS / 4)]),
        }, f)

    increasing_errors = sorted([fn[-1] for arm, fn in known_fns.items()])
    step_size = increasing_errors[1] - increasing_errors[0]
    bins = [0.074 + i * 0.001 for i in range(50)]

    print(increasing_errors)
    print(bins)
    plt.hist(optimums, bins=bins)
    plt.show()
