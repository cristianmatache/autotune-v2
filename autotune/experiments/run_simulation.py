import argparse
import pickle
from argparse import Namespace
from os.path import join as join_path

import matplotlib.pyplot as plt
import numpy as np

# Optimisers
from autotune.core import Optimiser, RoundRobinShapeFamilyScheduler, ShapeFamily, UniformShapeFamilyScheduler
from autotune.optimisers import HybridHyperbandTpeOptimiser, HyperbandOptimiser, RandomOptimiser, SigOptimiser, \
    TpeOptimiser, HybridHyperbandSigoptOptimiser, HybridHyperbandTpeTransferAllOptimiser, \
    HybridHyperbandTpeNoTransferOptimiser, HybridHyperbandTpeTransferLongestOptimiser, \
    HybridHyperbandTpeTransferSameOptimiser

# Problems
from autotune.core import HyperparameterOptimisationProblem, OptimisationGoals
from autotune.benchmarks import OptFunctionSimulationProblem

OUTPUT_DIR = "D:/datasets/output"

N_RESOURCES = 24
MAX_TIME = None
MAX_ITER = 81
ETA = 3

PROBLEM = "sim-rastrigin"
METHOD = "sim(hb+tpe+transfer+same)"
MIN_OR_MAX = "min"

N_SIMULATIONS = 7000
INIT_NOISE = 10

PLOT_EACH = False

families_of_shapes_egg = (
    ShapeFamily(None, 1.5, 10, 15, False, 0, 1000),  # with aggressive start
    ShapeFamily(None, 0.5, 7, 10, False, 0, 1000),  # with average aggressiveness at start and at the beginning
    ShapeFamily(None, 0.2, 4, 7, True, 0, 1000),  # non aggressive start, aggressive end
)
families_of_shapes_general = (
    ShapeFamily(None, 1.5, 10, 15, False),  # with aggressive start
    ShapeFamily(None, 0.5, 7, 10, False),  # with average aggressiveness at start and at the beginning
    ShapeFamily(None, 0.2, 4, 7, True),  # non aggressive start, aggressive end
    ShapeFamily(None, 1.5, 10, 15, False, 200, 400),  # with aggressive start
    ShapeFamily(None, 0.5, 7, 10, False, 200, 400),  # with average aggressiveness at start and at the beginning
    ShapeFamily(None, 0.2, 4, 7, True, 200, 400),  # non aggressive start, aggressive end

    # ShapeFamily(None, 0, 1, 0, False, 0, 0),  # flat
)

SCHEDULING = "uniform"

SHAPE_FAMILIES = {
    # 'sim-egg': families_of_shapes_egg,
    'sim-wave': families_of_shapes_general,
}.get(PROBLEM, families_of_shapes_general)


def optimisation_func(opt_goals: OptimisationGoals) -> float:
    """fval."""
    return opt_goals.fval


def _get_args() -> Namespace:
    parser = argparse.ArgumentParser(description='Running optimisations')
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


def get_problem(arguments: Namespace) -> HyperparameterOptimisationProblem:
    problem_name = arguments.problem.lower()
    problem_instance = OptFunctionSimulationProblem(
        func_name={
            'sim-branin': 'branin',
            'sim-egg': 'egg',
            'sim-camel': 'camel',
            'sim-wave': 'wave',
            'sim-rastrigin': 'rastrigin',
        }[problem_name])
    problem_instance.log_domain()
    return problem_instance


def get_optimiser() -> Optimiser:
    method = args.method.lower()
    min_or_max = min if args.min_or_max == 'min' else max
    scheduler = {
        'uniform': UniformShapeFamilyScheduler,
        'round-robin': RoundRobinShapeFamilyScheduler
    }[SCHEDULING](shape_families=SHAPE_FAMILIES, max_resources=args.max_iter, init_noise=INIT_NOISE)

    # SIMULATIONS
    if method == "sim(random)":
        return RandomOptimiser(
            n_resources=args.n_resources, max_iter=args.max_iter, max_time=args.max_time, min_or_max=min_or_max,
            optimisation_func=optimisation_func, is_simulation=True, scheduler=scheduler, plot_simulation=PLOT_EACH)
    elif method == "sim(hb)":
        return HyperbandOptimiser(
            eta=args.eta, max_iter=args.max_iter, max_time=args.max_time, min_or_max=min_or_max,
            optimisation_func=optimisation_func, is_simulation=True, scheduler=scheduler, plot_simulation=PLOT_EACH)
    elif method == "sim(tpe)":
        return TpeOptimiser(
            n_resources=args.n_resources, max_iter=args.max_iter, max_time=args.max_time, min_or_max=min_or_max,
            optimisation_func=optimisation_func, is_simulation=True, scheduler=scheduler, plot_simulation=PLOT_EACH)
    elif method == "sim(hb+tpe)":
        return HybridHyperbandTpeOptimiser(
            eta=args.eta, max_iter=args.max_iter, max_time=args.max_time, min_or_max=min_or_max,
            optimisation_func=optimisation_func, is_simulation=True, scheduler=scheduler, plot_simulation=PLOT_EACH)
    elif "sim(hb+tpe+transfer" in method:
        hybrid_transfer = {
            "sim(hb+tpe+transfer+none)": HybridHyperbandTpeNoTransferOptimiser,
            "sim(hb+tpe+transfer+all)": HybridHyperbandTpeTransferAllOptimiser,
            "sim(hb+tpe+transfer+longest)": HybridHyperbandTpeTransferLongestOptimiser,
            "sim(hb+tpe+transfer+same)": HybridHyperbandTpeTransferSameOptimiser
        }[method]
        return hybrid_transfer(
            eta=args.eta, max_iter=args.max_iter, max_time=args.max_time, min_or_max=min_or_max,
            optimisation_func=optimisation_func, is_simulation=True, scheduler=scheduler, plot_simulation=PLOT_EACH)
    elif method == "sim(sigopt)":
        return SigOptimiser(
            n_resources=args.n_resources, max_iter=args.max_iter, max_time=args.max_time, min_or_max=min_or_max,
            optimisation_func=optimisation_func, is_simulation=True, scheduler=scheduler, plot_simulation=PLOT_EACH)
    elif method == "sim(hb+sigopt)":
        return HybridHyperbandSigoptOptimiser(
            eta=args.eta, max_iter=args.max_iter, max_time=args.max_time, min_or_max=min_or_max,
            optimisation_func=optimisation_func, is_simulation=True, scheduler=scheduler, plot_simulation=PLOT_EACH)
    else:
        raise ValueError(f"Supplied problem {method} does not exist")


if __name__ == "__main__":
    args = _get_args()
    optimums = []
    for _ in range(N_SIMULATIONS):
        print("********iteration:", _, "avg so far:", 200 + np.mean(optimums))
        problem = get_problem(args)
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

    average_optimum = sum(optimums) / len(optimums)
    print(optimums)
    print(f"""\n\n------------- OPTIMUM STATISTICS OVER {N_SIMULATIONS} SIMULATIONS -------------
    average optimum: {np.mean(optimums)}
    sample std dev:  {np.std(optimums, ddof=1)}
    avg top 25%:     {np.mean(sorted(optimums)[:int(N_SIMULATIONS/4)])}
    top quartile:    {sorted(optimums)[:int(N_SIMULATIONS/4)]}
    """)
    print(200+average_optimum)

    with open(join_path(OUTPUT_DIR, f"hist-{PROBLEM}-{METHOD}-{N_SIMULATIONS}.pkl"), "wb") as f:
        pickle.dump({
            "method": METHOD,

            "n_simulations": N_SIMULATIONS,
            "all_optimums": optimums,

            "avg_optimum": np.mean(optimums),
            "sample_std_dev": np.std(optimums, ddof=1),
            "avg_top_25%": np.mean(sorted(optimums)[:int(N_SIMULATIONS / 4)]),
            "top_quartile": sorted(optimums)[:int(N_SIMULATIONS / 4)],
        }, f)
