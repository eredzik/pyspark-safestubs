import threading
from pyspark import SparkContext
from typing import Any, BinaryIO, Callable, Generic, Iterator, TypeVar, overload

__all__ = ['Broadcast']

T = TypeVar('T')

class Broadcast(Generic[T]):
    @overload
    def __init__(self, sc: SparkContext, value: T, pickle_registry: BroadcastPickleRegistry) -> None: ...
    @overload
    def __init__(self, *, path: str) -> None: ...
    @overload
    def __init__(self, *, sock_file: str) -> None: ...
    def dump(self, value: T, f: BinaryIO) -> None: ...
    def load_from_path(self, path: str) -> T: ...
    def load(self, file: BinaryIO) -> T: ...
    @property
    def value(self) -> T: ...
    def unpersist(self, blocking: bool = False) -> None: ...
    def destroy(self, blocking: bool = False) -> None: ...
    def __reduce__(self) -> tuple[Callable[[int], 'Broadcast[T]'], tuple[int]]: ...

class BroadcastPickleRegistry(threading.local):
    def __init__(self) -> None: ...
    def __iter__(self) -> Iterator[Broadcast[Any]]: ...
    def add(self, bcast: Broadcast[Any]) -> None: ...
    def clear(self) -> None: ...