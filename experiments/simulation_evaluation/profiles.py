import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.axes import Axes
from typing import List, Optional, Tuple, Type
import numpy as np

from benchmarks.branin_simulation_problem import BraninSimulationProblem, BraninSimulationEvaluator
from core import Arm,  RoundRobinShapeFamilyScheduler, ShapeFamily, Evaluator, ShapeFamilyScheduler


def get_evaluator_by_arm(arm: Arm, evaluators: List[Evaluator]) -> Evaluator:
    return [e for e in evaluators if e.arm == arm][0]


def get_avg_profile(loss_functions_per_fam: np.ndarray) -> np.ndarray:
    return np.mean(loss_functions_per_fam, axis=0)  # over columns (average at each epoch)


def get_med_profile(loss_functions_per_fam: np.ndarray) -> np.ndarray:
    return np.median(loss_functions_per_fam, axis=0)  # over columns (average at each epoch)


def get_std_profile(loss_functions_per_fam: np.ndarray) -> np.ndarray:
    return np.std(loss_functions_per_fam, axis=0)  # over columns (average at each epoch)


def get_dynamic_order_profile(loss_functions_per_fam: np.ndarray) -> List[float]:
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
                elif loss_functions_per_fam[j, t] > loss_functions_per_fam[i, t]:
                    great_i_t.append(j)

                if loss_functions_per_fam[j, t-1] < loss_functions_per_fam[i, t-1]:
                    less_i_prev_t.append(j)
                elif loss_functions_per_fam[j, t-1] > loss_functions_per_fam[i, t-1]:
                    great_i_prev_t.append(j)

            still_less = set(less_i_t) & set(less_i_prev_t)
            still_great = set(great_i_t) & set(great_i_prev_t)

            function_score = (len(still_less) + len(still_great)) / (num_func - 1)
            function_scores.append(function_score)
        relativity.append(sum(function_scores) / len(function_scores))

    return relativity


def get_ends_order_profile(loss_functions_per_fam: np.ndarray) -> float:
    num_func, max_time = loss_functions_per_fam.shape

    function_scores = []
    for i in range(num_func):
        less_i_t, less_i_prev_t = [], []
        great_i_t, great_i_prev_t = [], []
        for j in range(num_func):
            if loss_functions_per_fam[j, max_time - 1] < loss_functions_per_fam[i, max_time - 1]:
                less_i_t.append(j)
            elif loss_functions_per_fam[j, max_time - 1] > loss_functions_per_fam[i, max_time - 1]:
                great_i_t.append(j)

            if loss_functions_per_fam[j, 0] < loss_functions_per_fam[i, 0]:
                less_i_prev_t.append(j)
            elif loss_functions_per_fam[j, 0] > loss_functions_per_fam[i, 0]:
                great_i_prev_t.append(j)

        still_less = set(less_i_t) & set(less_i_prev_t)
        still_great = set(great_i_t) & set(great_i_prev_t)

        function_score = (len(still_less) + len(still_great)) / (num_func - 1)
        function_scores.append(function_score)
    return sum(function_scores) / len(function_scores)


def plot_simulated(n_simulations: int, max_resources: int = 81, n_resources: Optional[int] = None,
                   shape_families: Tuple[ShapeFamily, ...] = (ShapeFamily(None, 0.9, 10, 0.1),),
                   init_noise: float = 0, scheduler: Type[ShapeFamilyScheduler] = RoundRobinShapeFamilyScheduler) \
        -> List[List[float]]:
    """ plots the surface of the values of simulated loss functions at n_resources, by default is plot the losses
    at max resources, in which case their values would be 200-branin (if necessary aggressiveness is not disabled)
    :param n_simulations: number of simulations
    :param max_resources: maximum number of resources
    :param n_resources: show the surface of the values of simulated loss functions at n_resources
                        Note that this is relative to max_resources
    :param shape_families: shape families
    :param init_noise: variance of initial noise
    :param scheduler: class not instance of a scheduler
    """
    simulator = BraninSimulationProblem()
    if n_resources is None:
        n_resources = max_resources
    assert n_resources <= max_resources
    scheduler = scheduler(shape_families, max_resources, init_noise)

    loss_functions = []
    for i in range(n_simulations):
        evaluator: BraninSimulationEvaluator = simulator.get_evaluator(*scheduler.get_family(), should_plot=True)
        evaluator.evaluate(n_resources=n_resources)
        loss_functions.append(evaluator.fs)

    return loss_functions


def plot_profiles(loss_functions: List[List[float]], ax1: Axes, ax2: Axes, ax4: Axes,
                  x_eor: Optional[float] = None, eor_fontsize: int = 13) -> None:
    loss_functions = np.array(loss_functions)

    avg_profile = get_avg_profile(loss_functions)
    med_profile = get_med_profile(loss_functions)
    std_profile = get_std_profile(loss_functions)
    dor_profile = get_dynamic_order_profile(loss_functions)
    eor_profile = get_ends_order_profile(loss_functions)

    rack = list(range(len(avg_profile)))
    colour = next(ax2._get_lines.prop_cycler)["color"]
    ax1.plot(rack, avg_profile, color=colour)
    ax1.plot(rack, med_profile, '--', color=colour)
    ax1.fill_between(rack, avg_profile + std_profile, avg_profile - std_profile, alpha=0.3, facecolor=colour)
    ax2.plot(rack[1:], dor_profile, color=colour)
    if x_eor is not None:
        ax4.text(x_eor-0.07, 0.2, f"{eor_profile:.2f}", color=colour, fontsize=eor_fontsize)


def get_suplots_axes_layout() -> Tuple[Axes, Axes, Axes, Axes]:
    gs = gridspec.GridSpec(12, 12)
    ax1: Axes = plt.subplot(gs[:8, :6])
    ax2: Axes = plt.subplot(gs[10:, :6])
    ax4: Axes = plt.subplot(gs[11:, 6:], frameon=False)
    ax3: Axes = plt.subplot(gs[:8, 6:])  # ax3 must be the last one
    plt.subplots_adjust(wspace=0.2)

    ax1.title.set_text("Dynamic profiles")
    ax1.set_xlabel("time/epoch/resources")
    ax1.set_ylabel("loss avg, med, std profiles")
    # ax1.set_ylim(-210, 210)
    ax2.set_xlabel("time/epoch/resources")
    ax2.set_ylabel("order profile")
    ax3.title.set_text("Loss functions")
    ax3.yaxis.tick_right()
    ax3.yaxis.set_label_position("right")
    ax3.set_xlabel("time/epoch/resources")
    ax3.set_ylabel("error/loss")

    ax4.set_xlabel("colours match clouds colours")
    ax4.title.set_text("Order at ends - static profile")
    ax4.get_xaxis().set_ticks([])
    ax4.get_yaxis().set_ticks([])

    return ax1, ax2, ax3, ax4
