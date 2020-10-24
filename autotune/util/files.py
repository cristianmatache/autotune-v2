from os import makedirs
from pathlib import Path
from typing import TypeVar, Union

TPath = TypeVar('TPath', str, Path)
PathType = Union[str, Path]


def ensure_dir(path: TPath) -> TPath:
    """If the directory at given path doesn't exist, it will create it.

    :param path: path to directory
    :return: path to directory
    """
    if not Path(path).exists():
        makedirs(path)
    return path
