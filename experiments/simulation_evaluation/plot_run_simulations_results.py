from os.path import join as join_path
import pickle
from typing import List

from util import flatten
from experiments.simulation_evaluation.plot_results import plot_epdf_ofe, plot_histograms

OUTPUT_DIR = "../../../epdf-ofes/simulation-flat-branin/"


def unpickle(method: str, n_simulations: int, til: int = 1) -> List[float]:
    res = []
    for i in range(1, til+1):
        with open(join_path(OUTPUT_DIR, f"hist-{method}-{n_simulations}-{til}.pkl"), "rb") as f:
            unpickled = pickle.load(f)
            print(unpickled)
            # res.append(unpickled["all_norm_optimums"])
            res.append(unpickled["all_optimums"])
    return flatten(res)


FONT_SIZE = 30

if __name__ == "__main__":
    start = 0.3
    end = 2
    step = 0.05

    bins = [start + step * i for i in range(1+int((end-start)/step))]
    print('n_bins', len(bins))

    hb = unpickle("sim(hb)", 7000)
    none = unpickle("sim(hb+tpe+transfer+none)", 7000)
    all_ = unpickle("sim(hb+tpe+transfer+all)", 7000)
    tpe = unpickle("sim(tpe)", 7000)
    surv = unpickle("sim(hb+tpe+transfer+longest)", 7000)

    data = (
        hb,
        none,
        all_,
        tpe,
        surv
    )
    labels = ['Hyperband', 'NONE', 'ALL', 'TPE', 'SURV']

    plot_histograms(data, labels, bins, font_size=FONT_SIZE)

    plot_epdf_ofe(data, labels, start=start, end=end, bandwidth=0.1, order_profile=False, font_size=FONT_SIZE)
    # plot_epdf_ofe(data, labels, start=0, end=5, bandwidth=0.2, order_profile=False, font_size=FONT_SIZE)
    # plot_epdf_ofe(data, labels, start=0, end=26, bandwidth=0.5, order_profile=True, font_size=FONT_SIZE)

