import matplotlib.pyplot as plt
import pickle
from os.path import join as join_path
from experiments.run_experiment import OUTPUT_DIR
from core import OptimisationGoals

# This will fetch the latest experiment on the following problem with the following optimization method
PROBLEM = "mnist"
METHOD = "random"

FILE_PATH = join_path(OUTPUT_DIR, f"results-{PROBLEM}-{METHOD}.pkl")


def optimisation_func(opt_goals: OptimisationGoals) -> float:
    """validation_error"""
    return opt_goals.validation_error


def optimisation_func_branin(opt_goals: OptimisationGoals) -> float:
    """fval"""
    return opt_goals.fval


if PROBLEM == "branin":
    optimisation_func = optimisation_func_branin


if __name__ == "__main__":
    with open(FILE_PATH, "rb") as f:
        optimum_evaluation, eval_history, checkpoints = pickle.load(f)
        [print(eh.evaluator.loss_history) for eh in eval_history]
        [plt.plot(list(range(len(evaluator_t.loss_history) - 1)), evaluator_t.loss_history[1:])
         for evaluator_t, _ in eval_history]
        plt.show()
