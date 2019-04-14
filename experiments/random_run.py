import pickle
import argparse
from os.path import join as join_path

from core.optimisers.random_optimizer import RandomOptimiser
from benchmarks.mnist_problem import MnistProblem
from benchmarks.cifar_problem_2 import CifarProblem


DEF_INPUT_DIR = "D:/datasets/"
DEF_OUTPUT_DIR = "D:/datasets/output"
DEF_N_RESOURCES = 3
MAX_ITER = 2


def _get_args():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-i', '--input_dir', default=DEF_INPUT_DIR, type=str, help='input dir')
    parser.add_argument('-o', '--output_dir', default=DEF_OUTPUT_DIR, type=str, help='output dir')
    parser.add_argument('-res', '--n_resources', default=DEF_N_RESOURCES, type=int, help='n_resources')
    args = parser.parse_args()
    print(f"""\n> ARGS random run:
    Input directory:  {args.input_dir}
    Output directory: {args.output_dir}
    # resources:      {args.n_resources}""")
    return args


if __name__ == "__main__":
    args = _get_args()

    # Define problem instance
    problem = CifarProblem(args.input_dir, args.output_dir)
    problem.print_domain()

    # Define maximum units of resource assigned to each optimisation iteration
    random_opt = RandomOptimiser(args.n_resources, max_iter=MAX_ITER)
    optimum = random_opt.run_optimization(problem, n_resources=args.n_resources, verbosity=True)
    print(f"Best hyperparams:\n{optimum['arm']}\nwith\n  - test error: {optimum['test_error']}\n"
          f"  - validation error: {optimum['validation_error']}")

    output_file_path = join_path(args.output_dir, "results.pkl")
    with open(output_file_path, 'wb') as f:
        pickle.dump([random_opt], f)
