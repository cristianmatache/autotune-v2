import pickle
from os.path import join as join_path
from typing import List

from scipy.stats import ks_2samp

from autotune.experiments.simulation_evaluation.plot_results import plot_epdf_ofe, plot_histograms
from autotune.util.datastructures import flatten

SIMULATION = "rastrigin-families-2"
output_dir = f"../../../epdf-ofes/simulation-{SIMULATION}/"


def unpickle(method: str, n_simulations: int, til: int = 1) -> List[float]:
    res = []
    for i in range(1, til+1):
        with open(join_path(output_dir, f"hist-sim-rastrigin-{method}-{n_simulations}-{til}.pkl"), "rb") as f:
            unpickled = pickle.load(f)
            print(unpickled)
            res.append(unpickled["all_optimums"])
    return flatten(res)


FONT_SIZE = 50

if __name__ == "__main__":
    start = {
        "flat-branin": 0.37,
        "flat-drop-wave": -1,
        "flat-rastrigin": 0,
        "rastrigin-families-1": -200,
        "rastrigin-families-2": -400,
    }[SIMULATION]
    end = {
        "flat-branin": 1.5,
        "flat-drop-wave": -0.3,
        "flat-rastrigin": 18,
        "rastrigin-families-1": -170,
        "rastrigin-families-2": -370,
    }[SIMULATION]
    step = {
        "flat-branin": 0.05,
        "flat-drop-wave": 0.02,
        "flat-rastrigin": 1,
        "rastrigin-families-1": 1,
        "rastrigin-families-2": 1,
    }[SIMULATION]
    bandwidth = {
        "flat-branin": 0.1,
        "flat-drop-wave": 0.02,
        "flat-rastrigin": 3,
        "rastrigin-families-1": 3,
        "rastrigin-families-2": 3,
    }[SIMULATION]

    bins = [start + step * i for i in range(1+int((end-start)/step))]
    print('n_bins', len(bins))
    # bins = None

    hb = unpickle("sim(hb)", 7000)
    none = unpickle("sim(hb+tpe+transfer+none)", 7000)
    all_ = unpickle("sim(hb+tpe+transfer+all)", 7000)
    surv = unpickle("sim(hb+tpe+transfer+longest)", 7000)
    same = unpickle("sim(hb+tpe+transfer+same)", 7000)
    # tpe = unpickle("sim(tpe)", 7000)
    tpe2 = unpickle("sim(tpe2xbudget)", 7000)

    # print("TPE vs TPE 2xBUDGET:", ks_2samp(tpe, tpe2))
    print("TPE 2xBUDGET vs HB", ks_2samp(tpe2, hb))
    print("HB vs HB+TPE+ALL", ks_2samp(hb, all_))
    print("HB+TPE+ALL vs HB+TPE+SAME", ks_2samp(all_, same))
    print("HB+TPE+SAME vs HB+TPE+NONE", ks_2samp(same, none))
    print("HB+TPE+SURV vs HB+TPE+NONE", ks_2samp(none, surv))

    data = (
        hb,
        none,
        all_,
        surv,
        same,
        # tpe,
        tpe2,
    )
    print([len(d) for d in data])
    # labels = ['Hyperband', 'NONE', 'ALL', 'SURV', 'SAME', 'TPE', 'TPE 2xBUDGET']
    labels = ['Hyperband', 'NONE', 'ALL', 'SURV', 'SAME', 'TPE 2xBUDGET']
    # labels = ['Hyperband', 'TPE', 'TPE 2xBUDGET']

    plot_histograms(data, labels, bins, font_size=FONT_SIZE)

    plot_epdf_ofe(data, labels, start=start, end=end, bandwidth=bandwidth, order_profile=False, font_size=FONT_SIZE)
    # plot_epdf_ofe(data, labels, start=0, end=5, bandwidth=0.2, order_profile=False, font_size=FONT_SIZE)
    # plot_epdf_ofe(data, labels, start=0, end=26, bandwidth=0.5, order_profile=True, font_size=FONT_SIZE)
