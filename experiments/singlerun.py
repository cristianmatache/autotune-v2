import argparse

from ..benchmarks.model_builders import CifarProblem2

# input_dir = '/Users/signapoop/Desktop/datasets/'
# output_dir = '/Users/signapoop/Desktop/autotune/autotune/experiments/checkpoint/'

# input_dir = '/datasets/'
# output_dir = '/checkpoint/'

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('-i', '--input_dir', type=str, help='input dir')
parser.add_argument('-o', '--output_dir', type=str, help='output dir')
args = parser.parse_args()

print(args.input_dir)
print(args.output_dir)

problem = CifarProblem2(args.input_dir, args.output_dir)
problem.print_domain()

arm = problem.generate_random_arm()
arm['n_resources'] = 3
problem.eval_arm(arm)


