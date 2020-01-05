import matplotlib.pyplot as plt
import pickle
from typing import List
from os.path import join as join_path

from experiments.run_experiment import OUTPUT_DIR
from core import Arm, ShapeFamily
from experiments.simulation_evaluation.profiles import plot_profiles, plot_simulated, get_suplots_axes_layout
from util import flatten


# This will fetch the latest experiment on the following problem with the following optimization method
PROBLEM = "mnist"
METHOD = "random"
IS_SIMULATION = True
IS_OVERALL_PROFILE = False
UNDERLYING_OPT_FUNCTION = 'rastrigin'


FILE_PATH = join_path(OUTPUT_DIR, f"results-{PROBLEM}-{METHOD}.pkl")

FILES = [join_path(OUTPUT_DIR, f"results-{PROBLEM}-{METHOD}-{n}.pkl") for n in (10, 20, 15, 19, 1, 100, 30)] + \
        [join_path(OUTPUT_DIR, f"results-{PROBLEM}-tpe-{n}.pkl") for n in (30, 200)]

groups: List[List[Arm]] = []


if __name__ == "__main__":

    ax1, ax2, ax3, ax4 = get_suplots_axes_layout()

    if IS_SIMULATION:

        families_of_shapes = (
            # ShapeFamily(None, 1.2, 3, 0.05, True, 450, 500),    # steep_start_early_flat - bottom blue cloud
            # ShapeFamily(None, 1.5, 500, 0.05, True, 170, 390),  # steep_start_late_flat - yellow cloud
            # ShapeFamily(None, 0.9, 3, 15, True, 0, 40),     # low_steep_medium_late - top green cloud
            # ShapeFamily(None, 0.72, 5, 0.15, True, 50, 290),    # medium_steep - red cloud

            ShapeFamily(None, 1.5, 10, 15, False),  # with aggressive start
            ShapeFamily(None, 0.5, 7, 10, False),  # with average aggressiveness at start and at the beginning
            ShapeFamily(None, 0.2, 4, 7, True),  # non aggressive start, aggressive end
            ShapeFamily(None, 1.5, 10, 15, False, 200, 400),  # with aggressive start
            ShapeFamily(None, 0.5, 7, 10, False, 200, 400),  # with average aggressiveness at start and at the beginning
            ShapeFamily(None, 0.2, 4, 7, True, 200, 400),  # non aggressive start, aggressive end

            # ShapeFamily(None, 7, 2, 0.05, True, 750, 830),    # steep_start_early_flat - blue
            # ShapeFamily(None, 6, 5, 0, True, 400, 710),    # steep_start_late_flat - yellow
            # ShapeFamily(None, 5, 500, 0.05, True, 0, 75),  # low_steep_medium_late - green
            # ShapeFamily(None, 4, 10, 4, True, 0, 290),  # flat_slight_incline - red
            # ShapeFamily(None, 6, 3, 0, True, 0, 530),   # medium_steep - purple
            # ShapeFamily(None, 20, 500, 50, False, 750, 800),  # jumpy_flat - brown

            # ShapeFamily(None, 0, 1, 0, False, 0, 0),  # flat
        )

        max_res = 81
        init_noise = 10

        if IS_OVERALL_PROFILE:
            simulated_loss_functions = plot_simulated(func_name=UNDERLYING_OPT_FUNCTION,
                                                      n_simulations=230, max_resources=max_res, n_resources=max_res,
                                                      shape_families=families_of_shapes, init_noise=init_noise)
            plot_profiles(simulated_loss_functions, ax1, ax2, ax4, 0.5, 10)
        else:
            interval_len = 1 / (1 + len(families_of_shapes))
            for i, fam in enumerate(families_of_shapes):
                simulated_loss_functions = plot_simulated(func_name=UNDERLYING_OPT_FUNCTION,
                                                          n_simulations=15, max_resources=max_res, n_resources=max_res,
                                                          shape_families=(fam,), init_noise=init_noise)
                plot_profiles(simulated_loss_functions, ax1, ax2, ax4, interval_len * (i + 1), 10)

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
        low_steep_medium_late = [20, 58, 60, 47, 42, 23]
        flat_slight_incline = [41, 4, 7, 40]
        medium_steep = [16, 26, 30, 34, 38, 51, 53, 59, 61]
        not_labelled = list(range(65, 425))

        if IS_OVERALL_PROFILE:
            families_indices = (steep_start_early_flat + steep_start_late_flat + low_steep_medium_late +
                                flat_slight_incline + medium_steep + jumpy_flat,)
        else:
            families_indices = (steep_start_early_flat, steep_start_late_flat, low_steep_medium_late,
                                flat_slight_incline, medium_steep, jumpy_flat)

        interval_len = 1 / (1 + len(families_indices))
        for i, fam in enumerate(families_indices):
            loss_functions_per_family = [all_evaluators[i].loss_history[1:] for i in fam]
            plot_profiles(loss_functions_per_family, ax1, ax2, ax4, interval_len * (i + 1), 10)

        labelled_indices = (steep_start_late_flat + steep_start_early_flat + low_steep_medium_late +
                            flat_slight_incline + medium_steep + jumpy_flat)
        all_indices = labelled_indices + not_labelled

        for evaluator_t in [all_evaluators[i] for i in all_indices]:
            plt.plot(list(range(len(evaluator_t.loss_history) - 1)), evaluator_t.loss_history[1:])

        plt.show()

        bins = [0.074 + i * 0.001 for i in range(100)]
        print(bins)
        plt.hist([e.loss_history[-1] for e in [all_evaluators[i] for i in all_indices]], bins=bins)
        print("best errors:", sorted([e.loss_history[-1] for e in [all_evaluators[i] for i in all_indices]]))

    plt.show()
