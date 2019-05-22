import argparse
from argparse import Namespace
import matplotlib.pyplot as plt

# Optimisers
from core import Optimiser, RoundRobinShapeFamilyScheduler, ShapeFamily
from optimisers import HybridHyperbandTpeOptimiser, HyperbandOptimiser, RandomOptimiser, SigOptimiser, TpeOptimiser, \
    HybridHyperbandSigoptOptimiser

# Problems
from core import HyperparameterOptimisationProblem, OptimisationGoals
from benchmarks import BraninProblem, BraninSimulationProblem

N_RESOURCES = 81
MAX_TIME = None
MAX_ITER = 81
ETA = 3

PROBLEM = "sim-branin"
METHOD = "sim(hb+tpe)"
MIN_OR_MAX = "min"

N_SIMULATIONS = 100
INIT_NOISE = 0.3

PLOT_EACH = True

SHAPE_FAMILIES = (
    ShapeFamily(None, 1.3, 10.0, 0.14, True),  # with aggressive start
    ShapeFamily(None, 0.6, 7.0, 0.1, True),  # with average aggressiveness at start and at the beginning
    ShapeFamily(None, 0.3, 3.0, 0.2, True),  # non aggressive start, aggressive end
)


def optimisation_func(opt_goals: OptimisationGoals) -> float:
    """fval"""
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
    if problem_name == "branin":
        problem_instance = BraninProblem()
    elif problem_name == "sim-branin":
        problem_instance = BraninSimulationProblem()
    else:
        raise ValueError(f"Supplied problem {problem_name} is not a simulation problem")
    problem_instance.print_domain()
    return problem_instance


def get_optimiser() -> Optimiser:
    method = args.method.lower()
    min_or_max = min if args.min_or_max == 'min' else max
    scheduler = RoundRobinShapeFamilyScheduler(shape_families=SHAPE_FAMILIES, max_resources=args.max_iter,
                                               init_noise=INIT_NOISE)
    # SIMULATIONS
    if method == "sim(random)":
        return RandomOptimiser(
            n_resources=args.n_resources, max_iter=args.max_iter, max_time=args.max_time, min_or_max=min_or_max,
            optimisation_func=optimisation_func, is_simulation=True, scheduler=scheduler)
    elif method == "sim(hb)":
        return HyperbandOptimiser(
            eta=args.eta, max_iter=args.max_iter, max_time=args.max_time, min_or_max=min_or_max,
            optimisation_func=optimisation_func, is_simulation=True, scheduler=scheduler)
    elif method == "sim(tpe)":
        return TpeOptimiser(
            n_resources=args.n_resources, max_iter=args.max_iter, max_time=args.max_time, min_or_max=min_or_max,
            optimisation_func=optimisation_func, is_simulation=True, scheduler=scheduler)
    elif method == "sim(hb+tpe)":
        return HybridHyperbandTpeOptimiser(
            eta=args.eta, max_iter=args.max_iter, max_time=args.max_time, min_or_max=min_or_max,
            optimisation_func=optimisation_func, is_simulation=True, scheduler=scheduler)
    elif method == "sim(sigopt)":
        return SigOptimiser(
            n_resources=args.n_resources, max_iter=args.max_iter, max_time=args.max_time, min_or_max=min_or_max,
            optimisation_func=optimisation_func, is_simulation=True, scheduler=scheduler)
    elif method == "sim(hb+sigopt)":
        return HybridHyperbandSigoptOptimiser(
            eta=args.eta, max_iter=args.max_iter, max_time=args.max_time, min_or_max=min_or_max,
            optimisation_func=optimisation_func, is_simulation=True, scheduler=scheduler)
    else:
        raise ValueError(f"Supplied problem {method} does not exist")


if __name__ == "__main__":
    args = _get_args()
    optimums = []
    for _ in range(N_SIMULATIONS):
        problem = get_problem(args)
        optimiser = get_optimiser()
        optimum_evaluation = optimiser.run_optimisation(problem, verbosity=True)
        print(f"Best hyperparams:\n{optimum_evaluation.evaluator.arm}\n"
              f"with:\n"
              f"  - {optimisation_func.__doc__}: {optimisation_func(optimum_evaluation.optimisation_goals)}\n"
              f"Total time:\n  - {optimiser.checkpoints[-1]} seconds")
        optimums.append(optimisation_func(optimum_evaluation.optimisation_goals))
        if PLOT_EACH:
            plt.show()

    average_optimum = sum(optimums) / len(optimums)
    print(optimums)
    print(f"\n\n------------- AVERAGE OPTIMUM OVER {N_SIMULATIONS} SIMULATIONS ------------- \n{average_optimum}")
    print(200+average_optimum)
