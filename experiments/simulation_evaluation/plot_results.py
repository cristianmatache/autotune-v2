from sklearn.neighbors import KernelDensity
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import List, Tuple

from experiments.simulation_evaluation.profiles import get_dynamic_order_profile


def plot_smooth(x: List[float], start: float, end: float, bandwidth: float = 0.5) -> np.ndarray:
    x = np.array(x).reshape((len(x), 1))
    kde = KernelDensity(kernel='epanechnikov', bandwidth=bandwidth).fit(x)
    x_plot = np.linspace(start, end, 1000)[:, np.newaxis]
    log_dens = kde.score_samples(x_plot)
    plt.plot(x_plot[:, 0], np.exp(log_dens), '-', label="epanechnikov", linewidth=3)
    return np.exp(log_dens)


def plot_epdf_ofe(data_: Tuple[List[float], ...], labels_: List[str],
                  *, start: float, end: float, bandwidth: float, order_profile: bool = False, font_size: int = 10) \
        -> None:
    if not order_profile:
        for dataset in data_:
            plot_smooth(dataset, start, end, bandwidth=bandwidth)
        plt.xlabel("Minimum final error found by one optimizer run (OFE)", fontsize=font_size)
        plt.ylabel("Estimated PDF", fontsize=font_size)
        plt.legend(labels_, prop={'size': font_size})
        plt.tick_params(axis='both', which='major', labelsize=font_size)
        plt.tick_params(axis='both', which='minor', labelsize=font_size)
        plt.show()
    else:
        gs = gridspec.GridSpec(2, 1, height_ratios=[11, 1])
        ax2: plt.Axes = plt.subplot(gs[1])
        ax1: plt.Axes = plt.subplot(gs[0])
        smoothed = []
        for dataset in data_:
            smoothed.append(plot_smooth(dataset, start, end, bandwidth=bandwidth))
            ax1.set_xlabel("Minimum final error found by one optimizer run (OFE)", fontsize=font_size)
            ax1.set_ylabel("Estimated PDF", fontsize=font_size)
        dor_profile = get_dynamic_order_profile(np.array(smoothed))
        ax2.plot(list(range(len(dor_profile))), dor_profile)
        ax2.set_ylabel("order profile", fontsize=font_size)
        ax2.get_xaxis().set_ticks([])
        ax2.tick_params(axis='both', which='major', labelsize=font_size)

        plt.legend(labels_, prop={'size': font_size})
        plt.tick_params(axis='both', which='major', labelsize=font_size)
        plt.tick_params(axis='both', which='minor', labelsize=font_size)
        plt.show()


def plot_histograms(data: Tuple[List[float], ...], labels: List[str], bins: List[float], 
                    *, font_size: int = 10) -> None:
    plt.hist(data, bins=bins, label=labels, cumulative=False, stacked=False, histtype='bar', density=True)
    plt.legend(prop={'size': font_size})
    plt.xlabel("Minimum final error found by one optimizer run (OFE)", fontsize=font_size)
    plt.ylabel("Count occurrences", fontsize=font_size)
    plt.tick_params(axis='both', which='major', labelsize=font_size)
    plt.tick_params(axis='both', which='minor', labelsize=font_size)
    plt.show()

    plt.hist(data, bins=bins, label=labels, cumulative=False, stacked=False, histtype='step', fill=True, linewidth=2)
    plt.legend(prop={'size': font_size})
    plt.xlabel("Minimum final error found by one optimizer run (OFE)", fontsize=font_size)
    plt.ylabel("Count occurrences", fontsize=font_size)
    plt.tick_params(axis='both', which='major', labelsize=font_size)
    plt.tick_params(axis='both', which='minor', labelsize=font_size)
    plt.show()

    # plt.hist(data, 100, label=labels, cumulative=False, stacked=False, histtype='step', fill=False, linewidth=2)
    # plt.xlabel("Minimum final error found by one optimizer run (OFE)", fontsize=font_size)
    # plt.ylabel("Count occurrences", fontsize=font_size)
    # plt.legend(prop={'size': font_size})
    # plt.tick_params(axis='both', which='major', labelsize=font_size)
    # plt.tick_params(axis='both', which='minor', labelsize=font_size)
    # plt.show()
