from os.path import join as join_path
import pickle
from typing import List
from scipy .stats import ks_2samp

from util import flatten
from experiments.simulation_evaluation.plot_results import plot_epdf_ofe, plot_histograms

OUTPUT_DIR = "../../../epdf-ofes/closest-known-loss-fn-mnist/"


def unpickle(method: str, n_simulations: int, til: int = 1) -> List[float]:
    res = []
    for i in range(1, til+1):
        with open(join_path(OUTPUT_DIR, f"known-fns-hist-{method}-{n_simulations}-{til}.pkl"), "rb") as f:
            unpickled = pickle.load(f)
            print({k: v for k, v in unpickled.items() if k != "all_norm_optimums"})
            res.append(unpickled["all_norm_optimums"])
    return flatten(res)


FONT_SIZE = 50


if __name__ == "__main__":
    bins = [0.074 + i * 0.001 for i in range(20)]
    print('n_bins', len(bins))

    known_hb = unpickle("sim(hb)", 2000) + unpickle("sim(hb)", 5000)
    known_hb_tpe_none = unpickle("sim(hb+tpe+transfer+none)", 2000) + unpickle("sim(hb+tpe+transfer+none)", 5000)
    known_hb_tpe_all = unpickle("sim(hb+tpe+transfer+all)", 2000) + unpickle("sim(hb+tpe+transfer+all)", 5000)
    known_hb_tpe_longest = (unpickle("sim(hb+tpe+transfer+longest)", 2000) +
                            unpickle("sim(hb+tpe+transfer+longest)", 5000))
    known_hb_tpe_same = unpickle("sim(hb+tpe+transfer+same)", 7000)

    known_tpe = unpickle("sim(tpe)", 7000)
    known_tpe2 = unpickle("sim(tpe2xbudget)", 7000)
    # known_rand = unpickle("sim(random)", 7000)

    print("TPE vs TPE 2xBUDGET:", ks_2samp(known_tpe, known_tpe2))
    print("TPE 2xBUDGET vs HB", ks_2samp(known_tpe2, known_hb))
    print("HB vs HB+TPE+ALL", ks_2samp(known_hb, known_hb_tpe_all))
    print("HB+TPE+ALL vs HB+TPE+SAME", ks_2samp(known_hb_tpe_all, known_hb_tpe_same))
    print("HB+TPE+SAME vs HB+TPE+NONE", ks_2samp(known_hb_tpe_same, known_hb_tpe_none))
    print("HB+TPE+SURV vs HB+TPE+NONE", ks_2samp(known_hb_tpe_longest, known_hb_tpe_none))

    data = (
        known_hb,
        known_hb_tpe_none,
        known_hb_tpe_all,
        known_hb_tpe_longest,
        known_hb_tpe_same,

        known_tpe,
        known_tpe2,
        # known_rand,
    )
    labels = ['Hyperband', 'NONE', 'ALL', 'SURV', 'SAME', 'TPE', 'TPE 2xBUDGET']
    # labels = ['Hyperband', 'TPE', 'TPE 2xBUDGET']
    print([len(d) for d in data])

    plot_histograms(data, labels, bins, font_size=FONT_SIZE)

    plot_epdf_ofe(data, labels, start=bins[0], end=bins[int(len(bins)/4)], bandwidth=0.003,
                  order_profile=False, font_size=FONT_SIZE)
    plot_epdf_ofe(data, labels, start=bins[0], end=bins[int(len(bins)/2)], bandwidth=0.005,
                  order_profile=False, font_size=FONT_SIZE)
    plot_epdf_ofe(data, labels, start=bins[0], end=bins[-1], bandwidth=0.005,
                  order_profile=False, font_size=FONT_SIZE)

    # known_hybrid = [float(np.mean(tpes)) for tpes in zip(
    #     sorted(known_hb_tpe_none), sorted(known_hb_tpe_all), sorted(known_hb_tpe_longest), sorted(known_hb_tpe_same)
    # )]
    # data_hybrid = (
    #     known_hb,
    #     known_hybrid,
    # )
    #
    # labels_hybdrid = ['Hyperband', 'Avg HB+TPE hybrids']
    #
    # plot_histograms(data_hybrid, labels_hybrid, bins, font_size=FONT_SIZE)
    #
    # plot_epdf_ofe(data_hybrid, labels_hybdrid, start=bins[0], end=bins[-1], bandwidth=0.005, order_profile=True)
