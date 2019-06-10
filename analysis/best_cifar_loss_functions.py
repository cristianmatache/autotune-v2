from typing import Any, List
import pickle
import matplotlib.pyplot as plt

BEST_LOSS_FUNCTIONS_PATH = '../../loss_functions/cifar/best_loss_functions-1.pkl'


def unpickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def distinct(l: List[List[Any]]) -> List[List[Any]]:
    return [list(x) for x in set(tuple(x) for x in l)]


if __name__ == '__main__':
    best_error_to_arm = {}

    loss_functions = []
    res = unpickle(BEST_LOSS_FUNCTIONS_PATH)
    for _, evals in res.items():
        for e in evals:
            loss_func, arm = e
            loss_functions.append(loss_func[1:])
            if loss_func:
                best_error_to_arm[loss_func[-1]] = arm

    loss_functions = distinct([lf for lf in loss_functions if len(lf) > 300])
    for lf in loss_functions:
        print(lf)
        plt.plot(list(range(len(lf))), lf)
    print(len(loss_functions))
    print(sorted([len(lf) for lf in loss_functions]))

    min_errors = sorted(best_error_to_arm.keys())[:3]
    [print(me, "\n", best_error_to_arm[me], "\n")for me in min_errors]
    plt.show()

    plt.hist(best_error_to_arm.keys())
    plt.show()

