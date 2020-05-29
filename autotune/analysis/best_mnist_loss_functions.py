import pickle
import matplotlib.pyplot as plt
import numpy as np
from os.path import join

from experiments.simulation_evaluation.profiles import plot_profiles, get_suplots_axes_layout
from scipy.stats import ks_2samp


FILE = '../../../opt_lfns_mnist.pkl'
FILES_J = {
    'hb': '../../../mnist-long_loss_functions_from_arm_hb.pkl',
    'none': '../../../mnist-long_loss_functions_from_arm_none.pkl',
    'surv': '../../../mnist-long_loss_functions_from_arm_surv.pkl'
}


def pad(lfns):
    max_len = max([len(lf) for lf in lfns])
    for lf in lfns:
        lf += [lf[-1] for _ in range(max_len - len(lf))]
    return lfns


if __name__ == '__main__':
    ax1, ax2, ax3, ax4 = get_suplots_axes_layout()

    with open(FILE, 'rb') as f:
        res = pickle.load(f)

    for k, v in FILES_J.items():
        with open(FILES_J[k], 'rb') as f:
            file_to_lfs = pickle.load(f)
            res[k] += [lf[1:] for _, lf in file_to_lfs.items() if len(lf) >= 30]

    results = {k: [lf[-1] for lf in v] for k, v in res.items()}
    print({len(lf) for k, v in res.items() for lf in v})

    [print(k, sorted(v)) for k, v in results.items()]
    [print(k, np.mean(v)) for k, v in results.items()]
    print(ks_2samp(res['hb'], res['none']), 'hb', 'none')
    print(ks_2samp(res['none'], res['surv']), 'none', 'surv')
    print(ks_2samp(res['hb'], res['surv']), 'hb', 'surv')

    interval_len = 1 / (1 + len(res))
    for i, (k, lfs) in enumerate(res.items()):
        lfs = [lf[1:] for lf in lfs]
        plot_profiles(pad(lfs), ax1, ax2, ax4, x_eor=interval_len * (i + 1), eor_fontsize=50)

    [plt.plot(lf[1:]) for k, v in res.items() for lf in v]
    plt.show()
