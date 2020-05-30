from math import log, ceil
import numpy as np
import mpmath


def generate_arms(n, r=1, is_hybrid=False):
    if is_hybrid:
        print(f"{'=' * 90}\n>> Generated {n} arms and evaluated with TPE for {r} resources\n")
    else:
        print(f"{'=' * 90}\n>> Generated {n} arms")
    return [1 for _ in range(n)]


def evaluate_arm(arm, r_i):
    return arm + r_i


def hyperband(max_iter=None, eta=3):
    # --- Initialize iterations and running time
    logeta = lambda x: log(x) / log(eta)
    s_max = int(logeta(max_iter))  # number of unique executions of Successive Halving (minus one)
    B = (s_max + 1) * max_iter  # total number of iterations (without reuse) per execution of Successive Halving (n,r)
    print(s_max, B)

    #### Begin Finite Horizon Hyperband outlerloop. Repeat indefinetely.
    for s in reversed(range(s_max + 1)):
        n = int(ceil(int(B / max_iter / (s + 1)) * eta ** s))  # initial number of configurations
        r = max_iter * eta ** (-s)  # initial number of iterations to run configurations for

        #### Begin Finite Horizon Successive Halving with (n,r)
        arms = generate_arms(n, r)

        for i in range(s + 1):
            # Run each of the n_i configs for r_i iterations and keep best n_i/eta
            n_i = n * eta ** (-i)
            r_i = r * eta ** i
            val_losses = [evaluate_arm(arm, r_i) for arm in arms]
            print(f"** Evaluated {len(arms)} arms (n_i is {n_i}), each with {r_i:.2f} resources")

            arms = [arms[i] for i in np.argsort(val_losses)[0:int(n_i / eta)]]  # halving
            print(f"   after halving {len(arms)} arms are left")

        #### End Finite Horizon Successive Halving with (n,r)


def blippar_hybrid(max_iter=None, eta=3):
    # --- Initialize iterations and running time
    logeta = lambda x: log(x) / log(eta)
    s_max = int(logeta(max_iter))  # number of unique executions of Successive Halving (minus one)
    B = (s_max + 1) * max_iter  # total number of iterations (without reuse) per execution of Successive Halving (n,r)
    print(s_max, B)

    #### Begin Finite Horizon Hyperband outlerloop. Repeat indefinetely.
    for s in reversed(range(s_max + 1)):
        n = int(ceil(int(B / max_iter / (s + 1)) * eta ** s))  # initial number of configurations
        r = max_iter * eta ** (-s)  # initial number of iterations to run configurations for

        #### Begin Finite Horizon Successive Halving with (n,r)
        for i in range(s + 1):
            # Run each of the n_i configs for r_i iterations and keep best n_i/eta
            n_i = n * eta ** (-i)
            r_i = r * eta ** i
            if i == 0:
                arms = generate_arms(n_i, r_i, True)
            else:
                val_losses = [evaluate_arm(arm, r_i) for arm in arms]
                print(f"** Evaluated {len(arms)} arms (n_i is {n_i}), each with {r_i:.2f} resources")
                # arms = [arms[i] for i in np.argsort(val_losses)[0:int(n_i / eta)]]  # halving
                arms = [arms[i] for i in np.argsort(val_losses)[0:int(len(arms) / eta)]]  # halving
                print(f"   after halving {len(arms)} arms are left")
        #### End Finite Horizon Successive Halving with (n,r)


def get_random_hyperparameter_configuration(): return 1


def run_then_return_val_loss(num_iters, hyperparameters): return 1


mpmath.mp.dps = 64


def jamieson_hb():
    max_iter = 1121  # maximum iterations/epochs per configuration
    eta = 3  # defines downsampling rate (default=3)
    logeta = lambda x: mpmath.log(x) / mpmath.log(eta)
    s_max = int(logeta(max_iter))  # number of unique executions of Successive Halving (minus one)
    B = (s_max + 1) * max_iter  # total number of iterations (without reuse) per execution of Succesive Halving (n,r)

    #### Begin Finite Horizon Hyperband outlerloop. Repeat indefinetely.
    for s in reversed(range(s_max + 1)):
        n = int(ceil(int(B / max_iter / (s + 1)) * eta ** s))  # initial number of configurations
        r = max_iter * eta ** (-s)  # initial number of iterations to run configurations for

        #### Begin Finite Horizon Successive Halving with (n,r)
        T = [get_random_hyperparameter_configuration() for i in range(n)]
        print(f"{'=' * 90}\n>> Generated {n} arms and evaluated with TPE for {r} resources\n")
        for i in range(s + 1):
            # Run each of the n_i configs for r_i iterations and keep best n_i/eta
            n_i = n * eta ** (-i)
            r_i = r * eta ** (i)
            val_losses = [run_then_return_val_loss(num_iters=r_i, hyperparameters=t) for t in T]
            print(f"** Evaluated {len(val_losses)} arms (n_i is {n_i}), each with {r_i:.2f} resources")
            T = [T[i] for i in np.argsort(val_losses)[0:int(n_i / eta)]]
        #### End Finite Horizon Successive Halving with (n,r)


if __name__ == "__main__":
    # hyperband(81)
    # print("\n\n\n\n\n\n\n")
    # blippar_hybrid(81)
    jamieson_hb()
