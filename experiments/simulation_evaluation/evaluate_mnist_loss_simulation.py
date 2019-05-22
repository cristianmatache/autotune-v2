import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
from typing import List, Any, Optional, Tuple
from os.path import join as join_path
import numpy as np

from experiments.run_experiment import OUTPUT_DIR
from benchmarks.branin_simulation_problem import BraninSimulationProblem, BraninSimulationEvaluator
from core import Arm,  RoundRobinShapeFamilyScheduler, ShapeFamily, Evaluator


# This will fetch the latest experiment on the following problem with the following optimization method
PROBLEM = "mnist"
METHOD = "random"
IS_SIMULATION = True

FILE_PATH = join_path(OUTPUT_DIR, f"results-{PROBLEM}-{METHOD}.pkl")

FILES = [join_path(OUTPUT_DIR, f"results-{PROBLEM}-{METHOD}-{n}.pkl") for n in (10, 20, 15, 19)]

groups: List[List[Arm]] = []


def flatten(list_of_lists: List[List[Any]]) -> List[Any]:
    return [val for sublist in list_of_lists for val in sublist]


def get_evaluator_by_arm(arm: Arm, evaluators: List[Evaluator]) -> Evaluator:
    return [e for e in evaluators if e.arm == arm][0]


def get_avg_profile(loss_functions_per_fam: np.ndarray) -> np.ndarray:
    return np.mean(loss_functions_per_fam, axis=0)  # over columns (average at each epoch)


def get_med_profile(loss_functions_per_fam: np.ndarray) -> np.ndarray:
    return np.median(loss_functions_per_fam, axis=0)  # over columns (average at each epoch)


def get_std_profile(loss_functions_per_fam: np.ndarray) -> np.ndarray:
    return np.std(loss_functions_per_fam, axis=0)  # over columns (average at each epoch)


def get_rel_profile(loss_functions_per_fam: np.ndarray) -> List[float]:
    relativity = []
    num_func, max_time = loss_functions_per_fam.shape
    for t in range(1, max_time):
        function_scores = []
        for i in range(num_func):
            less_i_t, less_i_prev_t = [], []
            great_i_t, great_i_prev_t = [], []
            for j in range(num_func):
                if loss_functions_per_fam[j, t] < loss_functions_per_fam[i, t]:
                    less_i_t.append(j)
                else:
                    great_i_t.append(j)

                if loss_functions_per_fam[j, t-1] < loss_functions_per_fam[i, t-1]:
                    less_i_prev_t.append(j)
                else:
                    great_i_prev_t.append(j)

            still_less = set(less_i_t) & set(less_i_prev_t)
            still_great = set(great_i_t) & set(great_i_prev_t)

            function_score = (len(still_less) + len(still_great)) / num_func
            function_scores.append(function_score)
        relativity.append(sum(function_scores) / len(function_scores))

    return relativity


def plot_simulated(n_simulations: int, max_resources: int = 81, n_resources: Optional[int] = None,
                   shape_families: Tuple[ShapeFamily, ...] = (ShapeFamily(None, 0.9, 10, 0.1),),
                   init_noise: float = 0) -> List[List[float]]:
    """ plots the surface of the values of simulated loss functions at n_resources, by default is plot the losses
    at max resources, in which case their values would be 200-branin (if necessary aggressiveness is not disabled)
    :param n_simulations: number of simulations
    :param max_resources: maximum number of resources
    :param n_resources: show the surface of the values of simulated loss functions at n_resources
                        Note that this is relative to max_resources
    :param shape_families: shape families
    :param init_noise:
    """
    simulator = BraninSimulationProblem()
    if n_resources is None:
        n_resources = max_resources
    assert n_resources <= max_resources
    scheduler = RoundRobinShapeFamilyScheduler(shape_families, max_resources, init_noise)

    loss_functions = []
    for i in range(n_simulations):
        evaluator: BraninSimulationEvaluator = simulator.get_evaluator(*scheduler.get_family())
        evaluator.evaluate(n_resources=n_resources)
        loss_functions.append(evaluator.fs)

    return loss_functions


def plot_profiles(loss_functions: List[List[float]]) -> None:
    loss_functions = np.array(loss_functions)

    avg_profile = get_avg_profile(loss_functions)
    med_profile = get_med_profile(loss_functions)
    std_profile = get_std_profile(loss_functions)
    rel_profile = get_rel_profile(loss_functions)

    rack = list(range(len(avg_profile)))
    ax1.plot(rack, avg_profile)
    ax1.plot(rack, med_profile, '--')
    ax1.fill_between(rack, avg_profile + std_profile, avg_profile - std_profile, alpha=0.2)
    ax2.plot(rack[1:], rel_profile, ':')


if __name__ == "__main__":
    gs = gridspec.GridSpec(10, 10)
    ax1 = plt.subplot(gs[:8, :5])
    ax2 = plt.subplot(gs[8:, :5])
    ax3 = plt.subplot(gs[:8, 5:])

    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss avg, med, std profiles")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("rel profile")

    if IS_SIMULATION:

        families_of_shapes = (
            ShapeFamily(None, 1.2, 3, 0.05, True, 450, 500),    # steep_start_early_flat - bottom blue cloud
            ShapeFamily(None, 1.5, 500, 0.05, True, 170, 390),  # steep_start_late_flat - yellow cloud
            ShapeFamily(None, 0.5, 20, 0.05, False, 0, 30),     # low_steep_medium_late - top green cloud
            ShapeFamily(None, 0.72, 5, 0.15, True, 50, 290),    # medium_steep - red cloud
        )

        for fam in families_of_shapes:
            simulated_loss_functions = plot_simulated(n_simulations=15, max_resources=81, n_resources=81,
                                                      shape_families=(fam,), init_noise=0.3)
            plot_profiles(simulated_loss_functions)

        plt.show()

    else:

        all_evaluators = []
        for file in FILES:
            with open(file, "rb") as f:
                optimum_evaluation, eval_history, checkpoints = pickle.load(f)
                all_evaluators.append([evaluator_t for evaluator_t, _ in eval_history])
        all_evaluators = flatten(all_evaluators)

        steep_start_late_flat = [1, 2, 5, 10, 17, 19, 21, 22, 29, 36]
        steep_start_early_flat = [0, 3, 8, 11, 12, 14, 25, 28, 31, 32, 33, 35, 37, 44, 54, 55, 63]
        jumpy_flat = [6, 9, 13, 15, 24, 27, 45, 46, 48, 39, 43, 49, 50, 52, 56, 57, 62]
        low_steep_medium_late = [4, 7, 20, 23, 40, 41, 42, 47, 58, 60]
        medium_steep = [16, 26, 30, 34, 38, 51, 53, 59, 61]

        families_indices = (steep_start_early_flat, steep_start_late_flat, low_steep_medium_late, medium_steep)
        # jumpy_flat)

        for fam in families_indices:
            loss_functions_per_family = [all_evaluators[i].loss_history[1:] for i in fam]
            plot_profiles(loss_functions_per_family)

        for evaluator_t in [all_evaluators[i] for i in
                            steep_start_late_flat+steep_start_early_flat+low_steep_medium_late+medium_steep+jumpy_flat]:
            plt.plot(list(range(len(evaluator_t.loss_history) - 1)), evaluator_t.loss_history[1:])
        plt.show()
