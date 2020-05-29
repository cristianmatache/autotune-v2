import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern


BOUNDS = np.array([[-1.0, 2.0]])
NOISE = 0.2


def plot_approximation(gpr, x, y, x_sample, y_sample, x_next=None, show_legend=False):
    mu, std = gpr.predict(x, return_std=True)
    plt.fill_between(x.ravel(),
                     mu.ravel() + 1.96 * std,
                     mu.ravel() - 1.96 * std,
                     alpha=0.1)
    plt.plot(x, y, 'y--', lw=1, label='Noise-free objective')
    plt.plot(x, mu, 'b-', lw=1, label='Surrogate function')
    plt.plot(x_sample, y_sample, 'kx', mew=3, label='Noisy samples')
    if x_next:
        plt.axvline(x=x_next, ls='--', c='k', lw=1)
    if show_legend:
        plt.legend()


def plot_acquisition(x, y, x_next, show_legend=False):
    plt.plot(x, y, 'r-', lw=1, label='Acquisition function')
    plt.axvline(x=x_next, ls='--', c='k', lw=1, label='Next sampling location')
    if show_legend:
        plt.legend()


def plot_convergence(x_sample, y_sample, n_init=2):
    plt.figure(figsize=(12, 3))

    x = x_sample[n_init:].ravel()
    y = y_sample[n_init:].ravel()
    r = range(1, len(x) + 1)

    x_neighbor_dist = [np.abs(a - b) for a, b in zip(x, x[1:])]
    y_max_watermark = np.maximum.accumulate(y)

    plt.subplot(1, 2, 1)
    plt.plot(r[1:], x_neighbor_dist, 'bo-')
    plt.xlabel('Iteration')
    plt.ylabel('Distance')
    plt.title('Distance between consecutive x\'s')

    plt.subplot(1, 2, 2)
    plt.plot(r, y_max_watermark, 'ro-')
    plt.xlabel('Iteration')
    plt.ylabel('Best Y')
    plt.title('Value of best selected sample')


def f(x, noise=NOISE):
    return -np.sin(3 * x) - x ** 2 + 0.7 * x + noise * np.random.randn(*x.shape)


def expected_improvement(x, x_sample, y_sample, gpr, xi=0.01):
    """Computes EI at points X based on existing samples X_sample and Y_sample using a Gaussian process surrogate model.
    :param x: Points at which EI shall be computed (m x d).
    :param x_sample: Sample locations (n x d)
    :param y_sample: Sample values (n x 1)
    :param gpr: A GaussianProcessRegressor fitted to samples
    :param xi: Exploitation-exploration trade-off parameter
    :return: Expected improvements at points X
    """
    mu, sigma = gpr.predict(x, return_std=True)
    mu_sample = gpr.predict(x_sample)

    sigma = sigma.reshape(-1, x_sample.shape[1])

    # Needed for noise-based model,
    # otherwise use np.max(Y_sample).
    # See also section 2.4 in [...]
    mu_sample_opt = np.max(mu_sample)

    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        z = imp / sigma
        ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
        ei[sigma == 0.0] = 0.0

    return ei


def propose_location(acquisition, x_sample, y_sample, gpr, bounds, n_restarts=25):
    """Proposes the next sampling point by optimizing the acquisition function.
    :param acquisition: Acquisition function
    :param x_sample: Sample locations (n x d)
    :param y_sample: Sample values (n x 1)
    :param gpr: A GaussianProcessRegressor fitted to samples
    :param bounds:
    :param n_restarts:
    :return:
    """
    dim = x_sample.shape[1]
    min_val = 1
    min_x = None

    def min_obj(x):
        # Minimization objective is the negative acquisition function
        return -acquisition(x.reshape(-1, dim), x_sample, y_sample, gpr)

    # Find the best optimum by starting from n_restart different random points.
    for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, dim)):
        res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')
        if res.fun < min_val:
            min_val = res.fun[0]
            min_x = res.x

    return min_x.reshape(-1, 1)


if __name__ == "__main__":
    X_init = np.array([[-0.9], [1.1]])
    Y_init = f(X_init)
    print(X_init)

    # Dense grid of points within bounds
    X = np.arange(BOUNDS[:, 0], BOUNDS[:, 1], 0.01).reshape(-1, 1)

    # Noise-free objective function values at X
    Y = f(X, 0)

    m52 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
    gpr = GaussianProcessRegressor(kernel=m52, alpha=NOISE ** 2)

    # Initialize samples
    X_sample = X_init
    Y_sample = Y_init

    # Number of iterations
    n_iter = 10

    plt.figure(figsize=(12, n_iter * 3))
    plt.subplots_adjust(hspace=0.4)

    for i in range(n_iter):
        # Update Gaussian process with existing samples
        gpr.fit(X_sample, Y_sample)

        # Obtain next sampling point from the acquisition function (expected_improvement)
        X_next = propose_location(expected_improvement, X_sample, Y_sample, gpr, BOUNDS)

        # Obtain next noisy sample from the objective function
        Y_next = f(X_next, NOISE)

        # Plot samples, surrogate function, noise-free objective and next sampling location
        plt.subplot(n_iter, 2, 2 * i + 1)
        plot_approximation(gpr, X, Y, X_sample, Y_sample, X_next, show_legend=i == 0)
        plt.title(f'Iteration {i + 1}')

        plt.subplot(n_iter, 2, 2 * i + 2)
        plot_acquisition(X, expected_improvement(X, X_sample, Y_sample, gpr), X_next, show_legend=i == 0)

        # Add sample to previous samples
        X_sample = np.vstack((X_sample, X_next))
        Y_sample = np.vstack((Y_sample, Y_next))

    plt.show()
