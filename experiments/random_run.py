import pickle
import argparse
from os.path import join as join_path

from optimisers.random_optimizer import RandomOptimiser
from benchmarks.mnist_problem import MnistProblem
from benchmarks.cifar_problem import CifarProblem
from benchmarks.svhn_problem import SvhnProblem

DEF_INPUT_DIR = "D:/datasets/"
DEF_OUTPUT_DIR = "D:/datasets/output"
DEF_N_RESOURCES = 3
MAX_ITER = 2
PROBLEM = "cifar"


def _get_args():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-i', '--input_dir', default=DEF_INPUT_DIR, type=str, help='input dir')
    parser.add_argument('-o', '--output_dir', default=DEF_OUTPUT_DIR, type=str, help='output dir')
    parser.add_argument('-res', '--n_resources', default=DEF_N_RESOURCES, type=int, help='n_resources')
    arguments = parser.parse_args()
    print(f"""\n> ARGS random run:
    Input directory:  {arguments.input_dir}
    Output directory: {arguments.output_dir}
    # resources:      {arguments.n_resources}""")
    return arguments


if __name__ == "__main__":
    args = _get_args()

    # Define problem instance
    problem = {
        "cifar": CifarProblem,
        "mnist": MnistProblem,
        "svhn": SvhnProblem
    }[PROBLEM.lower()](args.input_dir, args.output_dir)
    problem.print_domain()

    # Define maximum units of resource assigned to each optimisation iteration
    random_opt = RandomOptimiser(args.n_resources, max_iter=MAX_ITER)
    optimum = random_opt.run_optimization(problem, n_resources=args.n_resources, verbosity=True)
    print(f"Best hyperparams:\n{optimum['arm']}\nwith\n  - test error: {optimum['test_error']}\n"
          f"  - validation error: {optimum['validation_error']}")

    output_file_path = join_path(args.output_dir, "results.pkl")
    with open(output_file_path, 'wb') as f:
        pickle.dump([random_opt], f)
