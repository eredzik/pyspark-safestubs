from _typeshed import Incomplete
from pyspark._globals import _NoValueType
from typing import Any, Callable, Iterator

__all__ = ['get_option', 'set_option', 'reset_option', 'options', 'option_context']

class Option:
    key: Incomplete
    doc: Incomplete
    default: Incomplete
    types: Incomplete
    check_func: Incomplete
    def __init__(self, *, key: str, doc: str, default: Any, types: tuple[type, ...] | type = ..., check_func: tuple[Callable[[Any], bool], str] = ...) -> None: ...
    def validate(self, v: Any) -> None: ...

class OptionError(AttributeError, KeyError): ...

def get_option(key: str, default: Any | _NoValueType = ...) -> Any: ...
def set_option(key: str, value: Any) -> None: ...
def reset_option(key: str) -> None: ...
def option_context(*args: Any) -> Iterator[None]: ...

class DictWrapper:
    def __init__(self, d: dict[str, Option], prefix: str = '') -> None: ...
    def __setattr__(self, key: str, val: Any) -> None: ...
    def __getattr__(self, key: str) -> DictWrapper | Any: ...
    def __dir__(self) -> list[str]: ...

options: Incomplete
