from typing import Callable, Any, Tuple
from colorama import Fore, Style

from core.optimization_goals import OptimizationGoals
from util.progress_bar import progress_bar


def print_evaluation(verbose: bool = False, goals_to_print: Tuple[str] = ()) -> Callable:
    def decorator(evaluate_method: Callable) -> Callable:
        def wrapper(self, *args: Any, **kwargs: Any) -> OptimizationGoals:
            print(f"\n\n\n{Fore.CYAN}{'-' * 20} Evaluating model on arm {'-' * 20}\n{self.arm}{Style.RESET_ALL}")
            opt_goal = evaluate_method(self, *args, **kwargs)
            if verbose:
                print("\n" + opt_goal.goals_to_str(goals_to_print))
                if hasattr(opt_goal, "val_correct") and hasattr(opt_goal, "val_total"):
                    print_accuracy("Validation", opt_goal.val_correct, opt_goal.val_total)
                if hasattr(opt_goal, "test_correct") and hasattr(opt_goal, "test_total"):
                    print_accuracy("Test", opt_goal.test_correct, opt_goal.test_total)
            return opt_goal
        return wrapper
    return decorator


def print_accuracy(train_val_or_test: str, correct: int, total: int) -> None:
    accuracy = 100. * correct / total
    padding = ' ' * (len("validation") - len(train_val_or_test))
    print(f"{train_val_or_test} accuracy:{padding} {accuracy:.3f}% ({correct}/{total})")


def _display_progress_bar(batch_idx: int, len_data_loader: int, correct: int, total: int, total_loss: int,
                          disp_interval: int, train_val_or_test: str = None) -> None:
    if train_val_or_test not in ["Train", "Validation", "Test", None]:
        raise ValueError('train_val_or_test must be "Train", "Validation" or "Test"')

    if batch_idx % disp_interval == 0 or batch_idx == len_data_loader:
        progress_bar(batch_idx, len_data_loader, f'Loss: %.3f | {train_val_or_test} Acc: %.3f%% (%d/%d)'
                     % (total_loss / batch_idx, 100. * correct / total, correct, total))