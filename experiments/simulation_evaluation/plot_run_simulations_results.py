import matplotlib.pyplot as plt
from os.path import join as join_path
import pickle
from typing import List, Any
from sklearn.neighbors import KernelDensity
import numpy as np


OUTPUT_DIR = "D:/datasets/output"


def flatten(list_of_lists: List[List[Any]]) -> List[Any]:
    return [val for sublist in list_of_lists for val in sublist]


def unpickle(method: str, n_simulations: int, til: int = 1) -> List[float]:
    res = []
    for i in range(1, til+1):
        with open(join_path(OUTPUT_DIR, f"hist-{method}-{n_simulations}-{til}.pkl"), "rb") as f:
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
    start = 0.3
    end = 2
    step = 0.05

    bins = [start + step * i for i in range(1+int((end-start)/step))]
    print('n_bins', len(bins))

    norm_longest = unpickle("sim(hb+tpe+transfer+longest)", 500, 4) + unpickle("sim(hb+tpe+transfer+longest)", 1000, 1)
    norm_none = unpickle("sim(hb+tpe+transfer+none)", 500, 4) + unpickle("sim(hb+tpe+transfer+none)", 1000, 1)
    norm_all = unpickle("sim(hb+tpe+transfer+all)", 2000, 1) + unpickle("sim(hb+tpe+transfer+all)", 1000, 1)
    print(len(norm_longest), len(norm_none), len(norm_all))

    data = (
        norm_longest,
        norm_none,
        norm_all
    )
    labels = ['very bests from prev brackets', 'none', 'all comparable from prev brackets']

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
        plot_smooth(dataset, start, end, bandwidth=0.1)
    plt.xlabel("Branin-normalized (i.e. result+200) values")
    plt.ylabel("Smoothed count around smallest values")
    plt.legend(labels)
    plt.show()

    for dataset in data:
        plot_smooth(dataset, 0, 5, bandwidth=0.2)
    plt.xlabel("Branin-normalized (i.e. result+200) values")
    plt.ylabel("Smoothed count in the 'better' half")
    plt.legend(labels)
    plt.show()

    for dataset in data:
        plot_smooth(dataset, 0, 26, bandwidth=0.5)
        plt.xlabel("Branin-normalized (i.e. result+200) values")
        plt.ylabel("Smoothed count everywhere")
    plt.legend(labels)
    plt.show()
