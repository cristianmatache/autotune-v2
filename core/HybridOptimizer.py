import time
from math import ceil
import mpmath
from core.RandomOptimiser import RandomOptimiser
from core.TpeOptimiser import TpeOptimiser

mpmath.mp.dps = 64


def print_evaluations(evaluations):
    for arm, val_loss, test_loss in evaluations:
        print("-" * 50)
        print("arm:", arm)
        print("val_loss", val_loss)
        print("test_loss", test_loss)
        print("\n\n")


class HybridOptimiser(RandomOptimiser):
    def __init__(self):
        super(HybridOptimiser, self).__init__()
        self.name = "Hyperband"

    def run_optimization(self, problem, n_units=None, max_iter=None, eta=3, verbosity=False):
        # problem provides generate_random_arm and eval_arm(x)

        print("\n---- Running hyperband optimisation ----")
        print("Max iterations = {}".format(max_iter))
        print("Halving rate eta = {}".format(eta))
        print("----------------------------------------")

        # --- Initialize iterations and running time
        self.time_zero = time.time()
        self.cum_time = 0
        self.num_iterations = 0
        self.checkpoints = []

        logeta = lambda x: mpmath.log(x)/mpmath.log(eta)
        s_max = int(logeta(max_iter))  # number of unique executions of Successive Halving (minus one)
        if s_max >= 2:
            s_min = 2  # skip the rest of the brackets after s_min
        else:
            s_min = 0
        B = (s_max+1)*max_iter  # total number of iterations (without reuse) per execution of Successive Halving (n,r)

        #### Begin Finite Horizon Hyperband outlerloop. Repeat indefinetely.
        for s in reversed(range(s_min, s_max+1)):
            n = int(ceil(int(B/max_iter/(s+1))*eta**s))  # initial number of configurations
            r = max_iter*eta**(-s)  # initial number of iterations to run configurations for

            #### Begin Finite Horizon Successive Halving with (n,r)
            for i in range(s+1):
                # Run each of the n_i configs for r_i iterations and keep best n_i/eta
                n_i = n*eta**(-i)
                r_i = r*eta**i

                if i == 0:
                    tpe_optimizer = TpeOptimiser()
                    tpe_optimizer.run_optimization(problem, max_iter=n_i, n_resources=r_i)
                    evaluations = list(zip(tpe_optimizer.arms, tpe_optimizer.val_loss, tpe_optimizer.Y))
                    print(f"\n{'=' * 73}\n>> Generated {n} evaluators and evaluated with TPE for {r_i} resources\n"
                          f"--- Starting halving ---")
                else:
                    evaluations = [(arm, *problem.eval_arm(arm, r_i)) for arm, _, _ in evaluations]
                    print(f"** Evaluated {len(evaluations)} arms (n_i is {n_i}), each with {r_i:.2f} resources")

                # print_evaluations(evaluations)

                # Update history
                best_arm, min_val, Y_new = sorted(evaluations, key=lambda evaluation: evaluation[1])[0]
                self.arms.append(best_arm)
                self.val_loss.append(min_val)
                self.Y.append(Y_new)

                # --- Update current evaluation time and function evaluations
                self.cum_time = time.time() - self.time_zero
                self.checkpoints.append(self.cum_time)

                if verbosity:
                    print("time elapsed: {:.2f}s, f_current: {:.5f}, f_best: {:.5f}".format(
                        self.cum_time, Y_new, min(self.Y)))

                evaluations = sorted(evaluations, key=lambda evaluation: evaluation[1])[:int(n_i/eta)]
            #### End Finite Horizon Successive Halving with (n,r)

        self._compute_results()
