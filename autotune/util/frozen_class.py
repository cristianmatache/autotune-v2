from typing import Any, Type, TypeVar

T = TypeVar('T')


def frozen_class(cls: Type[T]) -> Type[T]:
    """ Allows setting attributes in the decorated class only in __init__
    :param cls: class that needs to be frozen
    :return: frozen class
    """
    # original_setattr = cls.__setattr__
    def __setattr__(self, name: str, val: Any) -> None:
        raise AttributeError(f"{type(self).__name__} is frozen, so you are not allowed to set attributes. "
                             f"You attempted to set {name} to {val}")
    cls.__setattr__ = __setattr__  # type: ignore
    return cls