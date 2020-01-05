from typing import Callable, Tuple
from colorama import Fore, Style

from core import OptimisationGoals, Evaluator

EVALUATE_METHOD_TYPE = Callable[[Evaluator, int], OptimisationGoals]


def print_evaluation(verbose: bool = False, goals_to_print: Tuple[str] = ()) \
        -> Callable[[EVALUATE_METHOD_TYPE], EVALUATE_METHOD_TYPE]:
    """ If verbose:
    Before evaluation - prints the arm that will be evaluated
    After  evaluation - prints the OptimisationGoals (the results) for example, that includes validation_error in
                        several cases
    :param verbose: whether to print
    :param goals_to_print: optimisation goals to print (printing all attributes -names and values- of OptimisationGoals
                           can be too verbose). Set to () if you want to see all optimisation goals/
    :return: decorated Evaluator.evaluate method
    """
    def decorator(evaluate_method: EVALUATE_METHOD_TYPE) -> EVALUATE_METHOD_TYPE:
        def wrapper(self: Evaluator, n_resources: int) -> OptimisationGoals:
            if verbose:
                print(f"\n\n\n{Fore.CYAN}{'-' * 20} Evaluating model on arm {'-' * 20}\n{self.arm}{Style.RESET_ALL}")
            opt_goal = evaluate_method(self, n_resources)
            if verbose:
                print("\n" + opt_goal.goals_to_str(goals_to_print))
                if hasattr(opt_goal, "val_correct") and hasattr(opt_goal, "val_total"):
                    _print_accuracy("Validation", opt_goal.val_correct, opt_goal.val_total)
                if hasattr(opt_goal, "test_correct") and hasattr(opt_goal, "test_total"):
                    _print_accuracy("Test", opt_goal.test_correct, opt_goal.test_total)
            return opt_goal
        return wrapper
    return decorator


def _print_accuracy(train_val_or_test: str, correct: int, total: int) -> None:
    accuracy = 100. * correct / total
    padding = ' ' * (len("validation") - len(train_val_or_test))
    print(f"{train_val_or_test} accuracy:{padding} {accuracy:.3f}% ({correct}/{total})")
