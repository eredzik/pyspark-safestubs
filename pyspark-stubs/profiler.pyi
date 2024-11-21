import pstats
from _typeshed import Incomplete
from memory_profiler import CodeMap, LineProfiler
from pyspark.accumulators import AccumulatorParam as AccumulatorParam
from pyspark.context import SparkContext as SparkContext
from pyspark.errors import PySparkRuntimeError as PySparkRuntimeError
from typing import Any, Callable

has_memory_profiler: bool
MemoryTuple = tuple[float, float, int]
LineProfile = tuple[int, MemoryTuple | None]
CodeMapDict = dict[str, list[LineProfile]]

class ProfilerCollector:
    profiler_cls: Incomplete
    udf_profiler_cls: Incomplete
    memory_profiler_cls: Incomplete
    profile_dump_path: Incomplete
    profilers: Incomplete
    def __init__(self, profiler_cls: type['Profiler'], udf_profiler_cls: type['Profiler'], memory_profiler_cls: type['Profiler'], dump_path: str | None = None) -> None: ...
    def new_profiler(self, ctx: SparkContext) -> Profiler: ...
    def new_udf_profiler(self, ctx: SparkContext) -> Profiler: ...
    def new_memory_profiler(self, ctx: SparkContext) -> Profiler: ...
    def add_profiler(self, id: int, profiler: Profiler) -> None: ...
    def dump_profiles(self, path: str) -> None: ...
    def show_profiles(self) -> None: ...

class Profiler:
    def __init__(self, ctx: SparkContext) -> None: ...
    def profile(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any: ...
    def stats(self) -> pstats.Stats | dict: ...
    def show(self, id: int) -> None: ...
    def dump(self, id: int, path: str) -> None: ...

class CodeMapForUDF(CodeMap):
    def add(self, code: Any, toplevel_code: Any | None = None, *, sub_lines: list | None = None, start_line: int | None = None) -> None: ...

class UDFLineProfiler(LineProfiler):
    code_map: Incomplete
    enable_count: int
    max_mem: Incomplete
    prevlines: Incomplete
    backend: Incomplete
    prev_lineno: Incomplete
    def __init__(self, **kw: Any) -> None: ...
    def __call__(self, func: Callable[..., Any] | None = None, precision: int = 1, *, sub_lines: list | None = None, start_line: int | None = None) -> Callable[..., Any]: ...
    def add_function(self, func: Callable[..., Any], *, sub_lines: list | None = None, start_line: int | None = None) -> None: ...

class PStatsParam(AccumulatorParam[pstats.Stats | None]):
    @staticmethod
    def zero(value: pstats.Stats | None) -> None: ...
    @staticmethod
    def addInPlace(value1: pstats.Stats | None, value2: pstats.Stats | None) -> pstats.Stats | None: ...

class MemUsageParam(AccumulatorParam[CodeMapDict | None]):
    @staticmethod
    def zero(value: CodeMapDict | None) -> None: ...
    @staticmethod
    def addInPlace(value1: CodeMapDict | None, value2: CodeMapDict | None) -> CodeMapDict | None: ...

class BasicProfiler(Profiler):
    def __init__(self, ctx: SparkContext) -> None: ...
    def profile(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any: ...
    def stats(self) -> pstats.Stats: ...
    def show(self, id: int) -> None: ...
    def dump(self, id: int, path: str) -> None: ...

class UDFBasicProfiler(BasicProfiler):
    def show(self, id: int) -> None: ...
    def dump(self, id: int, path: str) -> None: ...

class MemoryProfiler(Profiler):
    def __init__(self, ctx: SparkContext) -> None: ...
    def profile(self, sub_lines: list | None, start_line: int | None, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any: ...
    def stats(self) -> CodeMapDict: ...
    def show(self, id: int) -> None: ...
    def dump(self, id: int, path: str) -> None: ...
