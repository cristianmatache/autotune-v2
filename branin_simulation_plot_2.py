from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from typing import Union
import scipy.stats as stats


from core import Domain, Param

PLOT_SURFACE = True
OUTPUT_DIR = "D:/datasets/output"
HYPERPARAMS_DOMAIN = Domain(
    x=Param('x', -5, 10, distrib='uniform', scale='linear'),
    y=Param('y', 1, 15, distrib='uniform', scale='linear'))


def plot_aggressiveness_gammas(n: int, k: int) -> None:
    x = np.linspace(0, 20, 200)

    def plot_gamma_distrib(time: int) -> None:
        sqrt_beta_component = np.sqrt(k ** 2 + 4 * (n - time))
        beta_t = (k + sqrt_beta_component) / (
                    2 * (n - time))  # beta is increasing in terms of time so variance is decreasing
        alpha_t = k * beta_t + 1  # mode is always k (threshold for 0 aggressiveness)

        shape = alpha_t
        scale = 1 / beta_t

        y = stats.gamma.pdf(x, a=shape, scale=scale)
        plt.plot(x, y, "y-", label=r'$\alpha=29, \beta=3$')

    # for t in range(100, n):
    #     plot_gamma_distrib(t)
    plot_gamma_distrib(1)
    plot_gamma_distrib(30)
    plot_gamma_distrib(50)
    plt.show()


def branin(x1: Union[int, np.ndarray], x2: Union[int, np.ndarray]) -> Union[int, np.ndarray]:
    a = 1
    b = 5.1 / (4 * np.pi ** 2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)

    f = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s
    return f


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


def branin_simulate_ml(x1: Union[int, np.ndarray], x2: Union[int, np.ndarray], time: int = 0,
                       n: int = 81) -> Union[int, np.ndarray]:
    k = 2  # mode of gamma distribution - corresponds to 0 aggressiveness
    h1 = 0.5  # h1, h2 HYPERPARAMETERS TO OPTIMISE
    h2 = 15  # necessary aggressiveness or 5
    h3 = 0.1  # up spikiniess

    if time == 0:
        return branin(x1, x2)

    f_n = branin(x1, x2) - 200
    fs = [branin(x1, x2)]
    print(f"Starting from: {branin(x1, x2)}  and aiming to finish at: {f_n}")

    for t in range(time):
        agg = get_aggressiveness_from_gamma_distrib(t, n + 1, k)
        if agg == k:  # be neutral
            f_next_time = fs[-1]
        elif agg > k:  # be aggressive - go down with different aggressivenesses
            absolute_aggressiveness = agg - k
            function_debt = f_n - fs[-1]
            ml_aggressed = fs[-1] + absolute_aggressiveness * h1 * function_debt / 100
            time_aggressed = (f_n - ml_aggressed) * ((t / (n - 1)) ** h2)
            f_next_time = ml_aggressed + time_aggressed
        else:  # aggressiveness < k - go up
            time_left = n - t
            f_next_time = fs[-1] + h3 * time_left
            if time_left == 1:
                f_next_time = f_n
        fs.append(f_next_time)

    # if time == n:
    #     assert fs[-1] == f_n

    # plt.plot(list(range(time+1)), fs)
    return fs[-1]


def plot_branin_surface(n_resources: int, n_iterations: int) -> None:
    xs = []
    ys = []
    for i in range(n_iterations):
        xs.append(HYPERPARAMS_DOMAIN["x"].get_param_range(1, stochastic=True)[0])
        ys.append(HYPERPARAMS_DOMAIN["y"].get_param_range(1, stochastic=True)[0])

    if n_resources == 0:
        xs = np.array(xs, dtype="float64")
        ys = np.array(ys, dtype="float64")
        zs = branin(xs, ys) - 200
    else:
        zs = []
        for x in xs:
            for y in ys:
                z = branin_simulate_ml(x, y, time=n_resources, n=n_resources)
                assert z == branin(x, y) - 200
                zs.append(z)

    fig = plt.figure()
    ax = fig.gca(projection=Axes3D.name)
    surf = ax.plot_trisurf(xs, ys, zs, cmap="coolwarm", antialiased=True)
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


def plot_simulations(n_simulations: int) -> None:
    def evaluate_once() -> None:
        x = HYPERPARAMS_DOMAIN["x"].get_param_range(1, stochastic=True)[0]
        y = HYPERPARAMS_DOMAIN["y"].get_param_range(1, stochastic=True)[0]
        branin_simulate_ml(x, y, 81, 81)

    [evaluate_once() for _ in range(n_simulations)]
    plt.show()


if __name__ == "__main__":
    if PLOT_SURFACE:
        plot_branin_surface(0, 100)
    else:
        plot_simulations(5)
    # plot_aggressiveness_gammas(n=81, k=2)
