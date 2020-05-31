import pickle
import argparse
from os.path import join as join_path

from autotune.experiments.run_experiment import get_problem
from autotune.core import Arm

INPUT_DIR = "D:/datasets/"
OUTPUT_DIR = "D:/datasets/output"

N_RESOURCES = 3
PROBLEM = "cifar"


ARM_DICT = {
    "batch_size": 32,
    "gamma": 0.1,
    "learning_rate": 0.0005171139689348882,
    "lr_step": 1,
    "momentum": 0.9,
    "n_units_1": 256.0,
    "n_units_2": 256.0,
    "n_units_3": 131.0,
    "weight_decay": 0.004
}


def _get_args():
    parser = argparse.ArgumentParser(description='Evaluate a problem on a given arm')
    parser.add_argument('-i', '--input-dir', default=INPUT_DIR, type=str, help='input dir')
    parser.add_argument('-o', '--output-dir', default=OUTPUT_DIR, type=str, help='output dir')
    parser.add_argument('-p', '--problem', default=PROBLEM, type=str, help='problem (eg. cifar, mnist, svhn)')
    parser.add_argument('-res', '--n-resources', default=N_RESOURCES, type=int, help='n_resources', required=False)
    arguments = parser.parse_args()
    print(f"""\n> ARGS random run:
    Input directory:  {arguments.input_dir}
    Output directory: {arguments.output_dir}

    Problem:          {arguments.problem.upper()}
    """)
    return arguments


if __name__ == "__main__":
    args = _get_args()
    problem = get_problem(args)

    if sorted(problem.domain.hyperparams_names()) != sorted(ARM_DICT.keys()):
        raise ValueError(f"ARM_DICT keys (hyperparameter names) do not match keys in {type(problem).__name__} domain")

    evaluator = problem.get_evaluator(Arm(**ARM_DICT))
    opt_goals = evaluator.evaluate(n_resources=args.n_resources)

    output_file_path = join_path(args.output_dir, "results-single-run.pkl")
    with open(output_file_path, 'wb') as f:
        pickle.dump([Arm.__dict__], f)
