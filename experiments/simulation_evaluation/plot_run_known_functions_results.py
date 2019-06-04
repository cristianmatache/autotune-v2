import matplotlib.pyplot as plt
from os.path import join as join_path
import pickle
from typing import List
from sklearn.neighbors import KernelDensity
import numpy as np

from util import flatten

OUTPUT_DIR = "D:/datasets/output"


def unpickle(method: str, n_simulations: int, til: int = 1) -> List[float]:
    res = []
    for i in range(1, til+1):
        with open(join_path(OUTPUT_DIR, f"known-fns-hist-{method}-{n_simulations}-{til}.pkl"), "rb") as f:
            unpickled = pickle.load(f)
            print({k: v for k, v in unpickled.items() if k != "all_norm_optimums"})
            res.append(unpickled["all_norm_optimums"])
    return flatten(res)


def plot_smooth(x: List[float], start: float, end: float, bandwidth: float = 0.5) -> None:
    x = np.array(x).reshape((len(x), 1))
    kde = KernelDensity(kernel='epanechnikov', bandwidth=bandwidth).fit(x)
    x_plot = np.linspace(start, end, 1000)[:, np.newaxis]
    log_dens = kde.score_samples(x_plot)
    plt.plot(x_plot[:, 0], np.exp(log_dens), '-', label="epanechnikov")


if __name__ == "__main__":
    bins = [0.074 + i * 0.001 for i in range(20)]
    print('n_bins', len(bins))

    # known_rand = unpickle("sim(random)", 100) + unpickle("sim(tpe)", 1000)
    # known_tpe = unpickle("sim(tpe)", 100)
    known_hb = unpickle("sim(hb)", 2000)  # + unpickle("sim(hb)", 5000)
    known_hb_tpe_none = unpickle("sim(hb+tpe+transfer+none)", 2000)   # + unpickle("sim(hb+tpe+transfer+none)", 5000)
    known_hb_tpe_all = unpickle("sim(hb+tpe+transfer+all)", 2000)  # + unpickle("sim(hb+tpe+transfer+all)", 5000)

    data = (
        known_hb,
        known_hb_tpe_none,
        known_hb_tpe_all,
        # known_rand,
        # known_tpe,
    )
    labels = ['hb', 'hb-tpe-transfer-none', 'hb-tpe-transfer-all', 'tpe', 'rand']
    print(len(known_hb), len(known_hb_tpe_none), len(known_hb_tpe_all))

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
        plot_smooth(dataset, bins[0], bins[int(len(bins)/4)], bandwidth=0.003)
    plt.xlabel("Branin-normalized (i.e. result+200) values")
    plt.ylabel("Smoothed count around smallest values")
    plt.legend(labels)
    plt.show()

    for dataset in data:
        plot_smooth(dataset, bins[0], bins[int(len(bins)/2)], bandwidth=0.005)
    plt.xlabel("Branin-normalized (i.e. result+200) values")
    plt.ylabel("Smoothed count in the 'better' half")
    plt.legend(labels)
    plt.show()

    for dataset in data:
        plot_smooth(dataset, bins[0], bins[-1], bandwidth=0.005)
        plt.xlabel("Branin-normalized (i.e. result+200) values")
        plt.ylabel("Smoothed count everywhere")
    plt.legend(labels)
    plt.show()
