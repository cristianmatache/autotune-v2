from typing import Callable
from colorama import Fore, Style


def print_evaluation(evaluate_method: Callable) -> Callable:
    def wrapper(self, *args, **kwargs):
        print(f"\n\n\n{Fore.CYAN}{'-' * 20} Evaluating model on arm {'-' * 20}\n{self.arm}{Style.RESET_ALL}")
        return evaluate_method(self, *args, **kwargs)
    return wrapper
