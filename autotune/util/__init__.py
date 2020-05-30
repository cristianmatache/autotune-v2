from typing import List, Any


def flatten(list_of_lists: List[List[Any]]) -> List[Any]:
    return [val for sublist in list_of_lists for val in sublist]
