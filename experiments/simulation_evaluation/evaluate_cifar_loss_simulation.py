import matplotlib.pyplot as plt
import pickle
from os.path import join as join_path

from experiments.run_experiment import OUTPUT_DIR
from core import Arm, ShapeFamily
from experiments.simulation_evaluation.profiles import plot_profiles, plot_simulated, get_suplots_axes_layout


# This will fetch the latest experiment on the following problem with the following optimization method

IS_SIMULATION = False

FILE_PATH = join_path(OUTPUT_DIR, "true_loss_functions.pkl")
MIN_LEN_THRESHOLD = 300
UNDERLYING_OPT_FUNCTION = 'branin'

if __name__ == "__main__":

    ax1, ax2, ax3, ax4 = get_suplots_axes_layout()

    if IS_SIMULATION:

        families_of_shapes = (
            ShapeFamily(None, 0.5, 2, 200),        # with aggressive start
        )

        interval_len = 1 / (1 + len(families_of_shapes))
        for i, fam in enumerate(families_of_shapes):
            simulated_loss_functions = plot_simulated(func_name=UNDERLYING_OPT_FUNCTION,
                                                      n_simulations=10, max_resources=400, n_resources=400,
                                                      shape_families=(fam,), init_noise=1)
            plot_profiles(simulated_loss_functions, ax1, ax2, ax4, interval_len * (i + 1), 13)

    else:

        with open(FILE_PATH, "rb") as f:
            loss_histories = pickle.load(f)
            print(len(loss_histories))

        min_len = min([len(lh) for lh in loss_histories if len(lh) > MIN_LEN_THRESHOLD])
        print(min_len)
        best_of_hyperband = [lh[1:min_len] for lh in loss_histories if len(lh) > MIN_LEN_THRESHOLD]

        interval_len = 0.5
        plot_profiles(best_of_hyperband, ax1, ax2, ax4, interval_len)
        [plt.plot(list(range(len(lh) - 1)), lh[1:]) for lh in sorted(loss_histories) if len(lh) > MIN_LEN_THRESHOLD]

    plt.show()
