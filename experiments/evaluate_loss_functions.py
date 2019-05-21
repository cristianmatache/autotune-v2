import matplotlib.pyplot as plt
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

FILE_PATH = join_path(OUTPUT_DIR, f"results-{PROBLEM}-{METHOD}.pkl")

FILES = [join_path(OUTPUT_DIR, f"results-{PROBLEM}-{METHOD}-{n}.pkl") for n in (10, 20, 15, 19)]

groups: List[List[Arm]] = []


def flatten(list_of_lists: List[List[Any]]) -> List[Any]:
    return [val for sublist in list_of_lists for val in sublist]


def get_evaluator_by_arm(arm: Arm, evaluators: List[Evaluator]) -> Evaluator:
    return [e for e in evaluators if e.arm == arm][0]


def get_avg_profile(family_of_shapes: List[List[float]]) -> np.ndarray:
    numpy_family_of_shapes = np.array(family_of_shapes)
    return np.mean(numpy_family_of_shapes, axis=0)  # over columns (average at each epoch)


def get_med_profile(family_of_shapes: List[List[float]]) -> np.ndarray:
    numpy_family_of_shapes = np.array(family_of_shapes)
    return np.median(numpy_family_of_shapes, axis=0)  # over columns (average at each epoch)


def get_std_profile(family_of_shapes: List[List[float]]) -> np.ndarray:
    numpy_family_of_shapes = np.array(family_of_shapes)
    return np.std(numpy_family_of_shapes, axis=0)  # over columns (average at each epoch)


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
        # plt.plot(list(range(len(evaluator.fs))), evaluator.fs)
        loss_functions.append(evaluator.fs)

    return loss_functions


if __name__ == "__main__":
    families_of_shapes = (
        ShapeFamily(None, 7, 2, 0.05, 750, 830),  # steep_start_early_flat
        ShapeFamily(None, 7, 5, 0.05, 480, 710),
        ShapeFamily(None, 5, np.inf, 0.05, 0, 75),
        ShapeFamily(None, 6, 3.0, 0.005, 0, 500),
    )

    for fam in families_of_shapes:
        simulated_loss_functions = plot_simulated(n_simulations=20, max_resources=20, n_resources=20,
                                                  shape_families=(fam,), init_noise=0.3)
        # plt.show()
        avg_simulation_profile = get_avg_profile(simulated_loss_functions)
        med_simulation_profile = get_med_profile(simulated_loss_functions)
        std_simulation_profile = get_std_profile(simulated_loss_functions)

        rack = list(range(len(avg_simulation_profile)))
        plt.plot(rack, avg_simulation_profile)
        plt.plot(rack, med_simulation_profile, '--')
        plt.fill_between(rack, avg_simulation_profile+std_simulation_profile,
                         avg_simulation_profile-std_simulation_profile, alpha=0.2)
    plt.show()

    all_evaluators = []
    for file in FILES:
        with open(file, "rb") as f:
            optimum_evaluation, eval_history, checkpoints = pickle.load(f)
            all_evaluators.append([evaluator_t for evaluator_t, _ in eval_history])

    all_evaluators = flatten(all_evaluators)
    # all_evaluators = all_evaluators[]

    steep_start_late_flat = [1, 2, 5, 10, 17, 19, 21, 22, 29, 36]
    steep_start_early_flat = [0, 3, 8, 11, 12, 14, 25, 28, 31, 32, 33, 35, 37, 44, 54, 55, 63]
    jumpy_flat = [6, 9, 13, 15, 24, 27, 45, 46, 48, 39, 43, 49, 50, 52, 56, 57, 62]
    low_steep_medium_late = [4, 7, 20, 23, 40, 41, 42, 47, 58, 60]
    medium_steep = [16, 26, 30, 34, 38, 51, 53, 59, 61]

    families_indices = (steep_start_early_flat, steep_start_late_flat, low_steep_medium_late, medium_steep)
    # jumpy_flat)

    # plt.xlabel("epoch")
    # plt.ylabel("loss error")
    # for fam in families_indices:
    #     loss_functions_per_fam = [all_evaluators[i].loss_history[1:] for i in fam]
    #     avg_profile = get_avg_profile(loss_functions_per_fam)
    #     med_profile = get_med_profile(loss_functions_per_fam)
    #     std_profile = get_std_profile(loss_functions_per_fam)
    #
    #     rack = list(range(len(avg_profile)))
    #     plt.plot(rack, avg_profile)
    #     plt.plot(rack, med_profile, '--')
    #     plt.fill_between(rack, avg_profile+std_profile, avg_profile-std_profile, alpha=0.2)

    # for evaluator_t in [all_evaluators[i] for i in steep_start_late_flat]:
    # # for evaluator_t in all_evaluators:
    #     plt.plot(list(range(len(evaluator_t.loss_history) - 1)), evaluator_t.loss_history[1:])

    plt.show()
