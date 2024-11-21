from _typeshed import Incomplete
from py4j.java_collections import JavaArray as JavaArray
from py4j.java_gateway import JavaClass as JavaClass, JavaGateway as JavaGateway, JavaObject as JavaObject
from pyspark import SparkContext as SparkContext
from pyspark.errors import AnalysisException as AnalysisException, IllegalArgumentException as IllegalArgumentException, ParseException as ParseException, PySparkNotImplementedError as PySparkNotImplementedError, PythonException as PythonException, QueryExecutionException as QueryExecutionException, SparkUpgradeException as SparkUpgradeException, StreamingQueryException as StreamingQueryException, UnknownException as UnknownException
from pyspark.errors.exceptions.captured import CapturedException as CapturedException
from pyspark.pandas._typing import IndexOpsLike as IndexOpsLike, SeriesOrIndex as SeriesOrIndex
from pyspark.sql.column import Column as Column
from pyspark.sql.dataframe import DataFrame as DataFrame
from pyspark.sql.session import SparkSession as SparkSession
from pyspark.sql.window import Window as Window
from typing import Any, Callable, Sequence, TypeVar

has_numpy: bool
FuncT = TypeVar('FuncT', bound=Callable[..., Any])

def toJArray(gateway: JavaGateway, jtype: JavaClass, arr: Sequence[Any]) -> JavaArray: ...
def require_test_compiled() -> None: ...

class ForeachBatchFunction:
    func: Incomplete
    session: Incomplete
    def __init__(self, session: SparkSession, func: Callable[[DataFrame, int], None]) -> None: ...
    error: Incomplete
    def call(self, jdf: JavaObject, batch_id: int) -> None: ...
    class Java:
        implements: Incomplete

def to_str(value: Any) -> str | None: ...
def is_timestamp_ntz_preferred() -> bool: ...
def is_remote() -> bool: ...
def try_remote_functions(f: FuncT) -> FuncT: ...
def try_remote_avro_functions(f: FuncT) -> FuncT: ...
def try_remote_protobuf_functions(f: FuncT) -> FuncT: ...
def try_remote_window(f: FuncT) -> FuncT: ...
def try_remote_windowspec(f: FuncT) -> FuncT: ...
def get_active_spark_context() -> SparkContext: ...
def try_remote_observation(f: FuncT) -> FuncT: ...
def try_remote_session_classmethod(f: FuncT) -> FuncT: ...
def pyspark_column_op(func_name: str, left: IndexOpsLike, right: Any, fillna: Any = None) -> SeriesOrIndex | None: ...
def get_column_class() -> type['Column']: ...
def get_dataframe_class() -> type['DataFrame']: ...
def get_window_class() -> type['Window']: ...
