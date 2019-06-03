import matplotlib.pyplot as plt
from os.path import join as join_path
import pickle
from typing import List, Any
from sklearn.neighbors import KernelDensity
import numpy as np

from util import flatten

OUTPUT_DIR = "D:/datasets/output"


def unpickle(method: str, n_simulations: int, til: int = 1) -> List[float]:
    res = []
    for i in range(1, til+1):
        with open(join_path(OUTPUT_DIR, f"known-fns-hist-{method}-{n_simulations}-{til}.pkl"), "rb") as f:
            unpickled = pickle.load(f)
            print(unpickled)
            res.append(unpickled["all_norm_optimums"])
    return flatten(res)


def plot_smooth(x: List[float], start: float, end: float, bandwidth: float = 0.5) -> None:
    x = np.array(x).reshape((len(x), 1))
    kde = KernelDensity(kernel='epanechnikov', bandwidth=bandwidth).fit(x)
    x_plot = np.linspace(start, end, 1000)[:, np.newaxis]
    log_dens = kde.score_samples(x_plot)
    plt.plot(x_plot[:, 0], np.exp(log_dens), '-', label="epanechnikov")


if __name__ == "__main__":
    bins = [
        0.07433333333333336, 0.07566666666666666, 0.07699999999999996, 0.07833333333333325, 0.07966666666666655,
        0.08099999999999985, 0.08233333333333315, 0.08366666666666644, 0.08499999999999974, 0.08633333333333304,
        0.08766666666666634, 0.08899999999999963, 0.09033333333333293, 0.09166666666666623, 0.09299999999999953,
        0.09433333333333282, 0.09566666666666612, 0.09699999999999942, 0.09833333333333272, 0.09966666666666602,
        0.10099999999999931, 0.10233333333333261, 0.10366666666666591, 0.1049999999999992, 0.1063333333333325,
        0.1076666666666658, 0.1089999999999991, 0.1103333333333324, 0.11166666666666569, 0.11299999999999899,
        0.11433333333333229, 0.11566666666666559, 0.11699999999999888, 0.11833333333333218, 0.11966666666666548,
        0.12099999999999878, 0.12233333333333207, 0.12366666666666537, 0.12499999999999867, 0.12633333333333197,
        0.12766666666666526
    ]

    print('n_bins', len(bins))

    known_rand = unpickle("sim(random)", 100) + unpickle("sim(tpe)", 1000)
    known_tpe = unpickle("sim(tpe)", 100)
    known_hb_tpe_none = unpickle("sim(hb+tpe+transfer+none)", 100) + unpickle("sim(hb+tpe+transfer+none)", 1000)
    known_hb_tpe_all = unpickle("sim(hb+tpe+transfer+all)", 100) + unpickle("sim(hb+tpe+transfer+none)", 1000)

    data = (
        known_hb_tpe_none,
        known_hb_tpe_all,
        # known_rand,
        known_tpe,
    )
    labels = ['hb-tpe-transfer-none', 'hb-tpe-transfer-all', 'tpe', 'rand']

    plt.hist(data, bins=bins, label=labels, cumulative=False, stacked=False, histtype='bar', density=True)
    plt.legend()
    plt.xlabel("Branin-normalized (i.e. result+200) values")
    plt.ylabel("Count occurrences in simulation results")
    plt.show()

    plt.hist(data, bins=bins, label=labels, cumulative=False, stacked=False, histtype='step', fill=True, linewidth=2)
    plt.legend()
    plt.xlabel("Branin-normalized (i.e. result+200) values")
    plt.ylabel("Count occurrences in simulation results")
    plt.show()

    plt.hist(data, 100, label=labels, cumulative=False, stacked=False, histtype='step', fill=False, linewidth=2)
    plt.xlabel("Branin-normalized (i.e. result+200) values")
    plt.ylabel("Count occurrences in simulation results")
    plt.legend()
    plt.show()

    for dataset in data:
        plot_smooth(dataset, bins[0], bins[int(len(bins)/4)], bandwidth=0.1)
    plt.xlabel("Branin-normalized (i.e. result+200) values")
    plt.ylabel("Smoothed count around smallest values")
    plt.legend(labels)
    plt.show()

    for dataset in data:
        plot_smooth(dataset, bins[0], bins[int(len(bins)/2)], bandwidth=0.2)
    plt.xlabel("Branin-normalized (i.e. result+200) values")
    plt.ylabel("Smoothed count in the 'better' half")
    plt.legend(labels)
    plt.show()

    for dataset in data:
        plot_smooth(dataset, bins[0], bins[-1], bandwidth=0.5)
        plt.xlabel("Branin-normalized (i.e. result+200) values")
        plt.ylabel("Smoothed count everywhere")
    plt.legend(labels)
    plt.show()
