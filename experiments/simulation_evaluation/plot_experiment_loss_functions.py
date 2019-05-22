import matplotlib.pyplot as plt
import pickle
from os.path import join as join_path
from experiments.run_experiment import OUTPUT_DIR

# This will fetch the latest experiment on the following problem with the following optimization method
PROBLEM = "mnist"
METHOD = "random"

OUTPUT_DIR = "~/Desktop/reclone-2/param-tuning/loss_functions/mnist"

FILE_PATH = join_path(OUTPUT_DIR, f"results-{PROBLEM}-{METHOD}.pkl")


if __name__ == "__main__":
    with open(FILE_PATH, "rb") as f:
        optimum_evaluation, eval_history, checkpoints = pickle.load(f)
        [print(eh.evaluator.loss_history) for eh in eval_history]
        [plt.plot(list(range(len(evaluator_t.loss_history) - 1)), evaluator_t.loss_history[1:])
         for evaluator_t, _ in eval_history]
        plt.show()
