import argparse
from argparse import Namespace
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from experiments.run_closest_loss_fn_approximation import get_known_functions, get_real_problem
from benchmarks import KnownFnProblem
from core import Arm, HyperparameterOptimisationProblem

PROBLEM = "mnist"
N_FOLDS = 10

KNOWN_FUNCTIONS_DIR = "../../../loss_functions/"
INPUT_DIR = "D:/datasets/"
OUTPUT_DIR = "D:/datasets/output"


def _get_args() -> Namespace:
    parser = argparse.ArgumentParser(description='Running optimisations')
    parser.add_argument('-p', '--problem', default=PROBLEM, type=str, help='problem (eg. cifar, mnist, svhn)')
    parser.add_argument('-nf', '--n-folds', default=N_FOLDS, type=int, help='number of cross validation folds')

    parser.add_argument('-i', '--input-dir', default=INPUT_DIR, type=str, help='input dir')
    parser.add_argument('-o', '--output-dir', default=OUTPUT_DIR, type=str, help='output dir')

    arguments = parser.parse_args()
    print(f"""\n
    Problem:          {arguments.problem.upper()}
    """)
    return arguments


def get_normalization_factors(min_loss_fn_len: int, all_known_fns: Dict[Arm, List[float]]) -> np.array:
    normalization_factors = []
    for i in range(min_loss_fn_len):
        fn_values = [loss_fn[i] for _, loss_fn in all_known_fns.items()]
        normalization_factors.append(max(fn_values) - min(fn_values))
    return np.array(normalization_factors)


def get_errors(expected_loss_fn: List[float], actual_loss_fn: List[float], min_loss_fn_len: int) -> np.array:
    return np.array([abs(actual_loss_fn[i] - expected_loss_fn[i]) for i in range(min_loss_fn_len)])


def cross_validate(n_folds: int, all_known_fns: Dict[Arm, List[float]],
                   real_problem: HyperparameterOptimisationProblem) -> Tuple[np.array, int]:
    fold_size = len(all_known_fns) // n_folds
    arms = list(all_known_fns.keys())
    min_loss_fn_len = min(len(all_known_fns[arm]) for arm in arms)

    normalization_factors = get_normalization_factors(min_loss_fn_len, all_known_fns)

    res = []
    for i in range(0, len(all_known_fns), fold_size):
        test_arms = arms[i: i + fold_size]           # arms set apart for validation
        db_arms = arms[0: i] + arms[i + fold_size:]  # arms that are considered as part of the database

        db_loss_fns = {arm: all_known_fns[arm] for arm in db_arms}
        problem = KnownFnProblem(known_fs=db_loss_fns, real_problem=real_problem)

        for arm in test_arms:
            opt_goals = problem.get_evaluator(arm=arm).evaluate(n_resources=min_loss_fn_len)

            expected_loss_fn = all_known_fns[arm]
            actual_loss_fn = opt_goals.fvals

            # differences at each point
            raw_errors = get_errors(expected_loss_fn, actual_loss_fn, min_loss_fn_len)
            res.append(np.divide(raw_errors, normalization_factors))

    return np.array(res), min_loss_fn_len


def update_hist(epoch: int, cross_validation_res: np.array, percent: float = 0.1) -> None:
    res = cross_validation_res
    print(f'epoch: {epoch} error level: {100 * percent}%', sum(res[:, epoch] < percent), 'out of', len(res))
    print(f'epoch: {epoch} error level: {100 * percent}% confidence:', sum(res[:, epoch] < percent) / len(res))
    plt.cla()
    plt.xlim(0, 1)
    plt.ylim(0, 250)
    plt.title(f'Normalized error distribution at epoch {epoch+1}')
    plt.xlabel(f'Measured error between approximated and real loss function at epoch {epoch+1}')
    plt.ylabel('Count')
    plt.hist(res[:, epoch], bins=[0.05 * i for i in range(20)])
    print()


if __name__ == "__main__":
    args = _get_args()
    real_problem_ = get_real_problem(args)
    known_fns_ = get_known_functions(args, known_functions_dir=KNOWN_FUNCTIONS_DIR)

    res_, min_loss_fn_len_ = cross_validate(n_folds=args.n_folds, all_known_fns=known_fns_, real_problem=real_problem_)

    fig = plt.figure()
    hist = plt.hist(res_[:, 0], bins=[0.05 * i for i in range(20)])

    animation = animation.FuncAnimation(fig, update_hist, min_loss_fn_len_, fargs=(res_, ))
    animation.save('distributions_of_errors_at_each_x_point.gif', writer='imagemagick', fps=1)
