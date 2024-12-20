import threading
from py4j.java_gateway import JavaObject as JavaObject
from pyspark.errors import PySparkRuntimeError as PySparkRuntimeError
from types import TracebackType
from typing import Any, Callable, Iterator, TextIO

def print_exec(stream: TextIO) -> None: ...

class VersionUtils:
    @staticmethod
    def majorMinorVersion(sparkVersion: str) -> tuple[int, int]: ...

def fail_on_stopiteration(f: Callable) -> Callable: ...
def walk_tb(tb: TracebackType | None) -> Iterator[TracebackType]: ...
def try_simplify_traceback(tb: TracebackType) -> TracebackType | None: ...
def inheritable_thread_target(f: Callable) -> Callable: ...

class InheritableThread(threading.Thread):
    def __init__(self, target: Callable, *args: Any, **kwargs: Any) -> None: ...
    def start(self) -> None: ...
