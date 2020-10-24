# pylint: disable=pointless-statement,multiple-statements,function-redefined
from typing import Any, Callable, Iterable, TypeVar, Union, overload

from typing_extensions import Protocol

T = TypeVar('T')
VT = TypeVar('VT')


class MinOrMax(Protocol):
    __name__: str

    @overload
    def __call__(self, __arg1: T, __arg2: T, *_args: T, key: Callable[[T], Any] = ...) -> T: ...

    @overload
    def __call__(self, iterable: Iterable[T], *, key: Callable[[T], Any] = ...) -> T: ...

    @overload
    def __call__(self, __iterable: Iterable[T], *, key: Callable[[T], Any] = ..., default: VT) -> Union[T, VT]: ...
