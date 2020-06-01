from typing import List, TypeVar

T = TypeVar('T')


def flatten(list_of_lists: List[List[T]]) -> List[T]:
    return [val for sublist in list_of_lists for val in sublist]
