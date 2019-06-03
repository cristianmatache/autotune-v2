import numpy as np
import scipy.stats as stats
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from typing import List


def _plot_gamma_process_distribs(n: int, k: int) -> None:
    """ Overlaps all Gamma distributions of the Gamma process on which the simulation is based.
    :param n: number of distributions
    :param k: mode
    """
    x = np.linspace(0, 20, 200)

    def plot_gamma_distrib(time: int) -> None:
        sqrt_beta_component = np.sqrt(k ** 2 + 4 * (n - time))
        beta_t = (k + sqrt_beta_component) / (2 * (n - time))  # beta increases in terms of time so variance decreases
        alpha_t = k * beta_t + 1  # mode is always k (threshold for 0 aggressiveness)

        shape = alpha_t
        scale = 1 / beta_t

        y = stats.gamma.pdf(x, a=shape, scale=scale)
        plt.plot(x, y, "y-", label=r'$\alpha=29, \beta=3$')

    for t in range(n):
        plot_gamma_distrib(t)

    plt.xlabel("level of aggressiveness")
    plt.ylabel("pdf")
    plt.show()


def get_aggressiveness_from_gamma_distrib(time: int, n: int, k: int) -> float:
    sqrt_beta_component = np.sqrt(k**2 + 4 * (n - time))
    beta_t = (k + sqrt_beta_component) / (2*(n-time))  # beta is increasing in terms of time so variance is decreasing
    alpha_t = k * beta_t + 1  # mode is always k (threshold for 0 aggressiveness)

    shape = alpha_t
    scale = 1 / beta_t

    aggresiveness = np.random.gamma(shape, scale)
    # x = np.linspace(0, 20, 200)
    # y = stats.gamma.pdf(x, a=shape, scale=scale)
    # plt.plot(x, y, "y-", label=r'$\alpha=29, \beta=3$')
    return aggresiveness


class SimulationEvaluator:

    """
    """

    def __init__(self, ml_aggressiveness: float, necessary_aggressiveness: float, up_spikiness: float,
                 max_resources: int, is_smooth: bool = True):
        """
        :param ml_aggressiveness:
        :param necessary_aggressiveness:
        :param up_spikiness:
        :param max_resources:
        :param is_smooth:
        """
        self.max_resources = max_resources

        self.non_smooth_fs: List[float] = []
        self.is_smooth = is_smooth

        # Curve shape parameters DEPEND ON self.max_resources - feel free to add a schedule to them if you want
        # - ml aggressiveness = the higher h1 the more it bites from function debt - especially at the beginning
        self.ml_aggressiveness = ml_aggressiveness
        # - necessary aggressiveness = the higher h2 the later necessary aggressiveness starts - flattens tail later
        self.necessary_aggressiveness = necessary_aggressiveness
        # - up spikiniess = the lower h3 the smoother the function, the higher h3 the more upwards spikes
        #                   spikes are proportional with time left = bigger spikes at the beginning, smaller in the end
        self.up_spikiness = up_spikiness

    @property
    def fs(self) -> List[float]:
        if not self.is_smooth:
            return self.non_smooth_fs
        else:
            window = int(0.17 * self.max_resources + 6)
            window += 1 if window % 2 == 0 else 0
            return list(savgol_filter(self.non_smooth_fs, window, 3))  # throws FutureWarning: Using a non-tuple ...

    def simulate(self, time: int, n: int, f_n: float) -> None:
        """ X is time/any other resources, Y is f(X) - simulated function at time X. Note that this function can
        simulate any type of resource not only time but for simplicity in naming we use time for resource. This method,
        will simulate Ys up to the given time. If the function was already evaluated at some times, it will continue
        from there.
        :param time: generate new Xs to simulate Ys, where lastX < newXs <= time
        :param n: the last point that the simulated function will be evaluated at
        :param f_n: target, f(n), last value of f, usually = branin - 200
        """
        k = 2  # mode of all gamma distributions - corresponds to 0 aggressiveness

        prev_time = len(self.non_smooth_fs)
        for t in range(prev_time, time):
            agg_level = get_aggressiveness_from_gamma_distrib(t, n + 1, k)
            f_time = self.non_smooth_fs[-1]
            if agg_level == k:  # be neutral
                f_next_time = f_time
            elif agg_level > k:  # be aggressive - go down with different aggressivenesses
                absolute_aggressiveness = agg_level - k
                function_debt = f_n - f_time
                ml_aggressed = f_time + absolute_aggressiveness * self.ml_aggressiveness * function_debt / 100
                time_aggressed = (f_n - ml_aggressed) * ((t / (n - 1)) ** self.necessary_aggressiveness)
                f_next_time = ml_aggressed + time_aggressed
            else:  # aggressiveness < k - go up
                # time_left = n - t
                # f_next_time = f_time + self.up_spikiness * 1 / (1 + agg_level)
                time_left = n - t
                up_aggressed = f_time + self.up_spikiness * 1 / (1 + agg_level)
                time_aggressed = (f_n - up_aggressed) * ((t / (n - 1)) ** (1.1 * self.necessary_aggressiveness))
                f_next_time = up_aggressed + time_aggressed
                if time_left == 1:
                    f_next_time = f_n
            self.non_smooth_fs.append(f_next_time)
