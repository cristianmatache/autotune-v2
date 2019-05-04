from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from typing import Union, Tuple, List
from core import Domain, Param


OUTPUT_DIR = "D:/datasets/output"
HYPERPARAMS_DOMAIN = Domain(
    x=Param('x', -5, 10, distrib='uniform', scale='linear'),
    y=Param('y', 1, 15, distrib='uniform', scale='linear'))


def branin(x1: Union[int, np.ndarray], x2: Union[int, np.ndarray],
           n_resources: int = 0, prev_n_resources: int = 0, max_resources: int = 81,
           aggressivenesses: Tuple[int, ...] = ()) -> Tuple[Union[int, np.ndarray], Tuple[int, ...]]:
    a = 1
    b = 5.1 / (4 * np.pi ** 2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)

    f = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s  # actual branin
    print(aggressivenesses)

    if n_resources >= 0:
        aggressivenesses = list(aggressivenesses)
        # re-do the steps that have already been calculated
        for n_res in range(prev_n_resources):
            f += aggressivenesses[n_res]

        distributions = _get_aggressivenesses_distributions(max_resources)

        # continue to generate for un-seen number of resources
        for n_res in range(prev_n_resources, n_resources):
            aggressiveness = _get_random_aggressiveness(distributions[n_res])
            f += aggressiveness
            aggressivenesses.append(aggressiveness)

    return f, tuple(aggressivenesses)


def plot_branin_surface(n_resources: int = 0) -> None:
    x = []
    y = []
    for i in range(100):
        x.append(HYPERPARAMS_DOMAIN["x"].get_param_range(1, stochastic=True)[0])
        y.append(HYPERPARAMS_DOMAIN["y"].get_param_range(1, stochastic=True)[0])
    x = np.array(x, dtype="float64")
    y = np.array(y, dtype="float64")
    z, _ = branin(x, y)

    fig = plt.figure()
    ax = fig.gca(projection=Axes3D.name)
    surf = ax.plot_trisurf(x, y, z, cmap="coolwarm", antialiased=True)
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


def plot_one_branin_simulation(max_n_resources: int = 81) -> None:
    x = HYPERPARAMS_DOMAIN["x"].get_param_range(1, stochastic=True)[0]
    y = HYPERPARAMS_DOMAIN["y"].get_param_range(1, stochastic=True)[0]

    f_vals = []
    resources = list(range(max_n_resources))
    signs = (1,)
    for n_res in resources:
        f_val, signs = branin(x, y, n_res, n_res-1 if n_res > 1 else 0, max_n_resources, signs)
        f_vals.append(f_val)

    plt.plot(resources, f_vals)
    plt.show()


def _get_aggressivenesses_distributions(n: int) -> List[List[int]]:
    distribution = np.array([5, 20, 50, 20, 5, 0, 0, 0, 0, 0]) / 100  # final distribution

    distributions = [distribution]
    redistribution = 1 / (10 * n)
    for _ in range(n-1):
        # print(sum(distribution), list(distribution * 100))
        distribution = distribution * (1 - 1 / n)
        distribution += redistribution
        distributions.append(list(distribution))
    # print(sum(distribution), list(distribution * 100))
    return list(reversed(distributions))


def _get_random_aggressiveness(distribution: List[int]) -> int:
    aggressiveness_index = np.random.choice(np.arange(len(distribution)), p=distribution)
    return [20, 10, 0, -5, -10, -15, -20, -25, -30, -35][aggressiveness_index]


def plot_one_run_through(max_resources: int = 0) -> None:
    distributions = _get_aggressivenesses_distributions(max_resources)

    f = 0

    f_vals = []
    resources = list(range(max_resources))
    print(f"len of distribs {len(distributions)}")
    for n_res in resources:
        f += _get_random_aggressiveness(distributions[n_res])
        f_vals.append(f)

    plt.plot(resources, f_vals)
    plt.show()


if __name__ == "__main__":
    # plot_branin_surface()
    plot_one_branin_simulation()

