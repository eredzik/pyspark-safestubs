from pyspark.pandas._typing import T as T
from pyspark.pandas.frame import DataFrame as DataFrame
from pyspark.pandas.indexes import Index as Index
from pyspark.pandas.series import Series as Series
from typing import Callable, Generic

class CachedAccessor(Generic[T]):
    def __init__(self, name: str, accessor: type[T]) -> None: ...
    def __get__(self, obj: DataFrame | Series | Index | None, cls: type[T]) -> T | type[T]: ...

def register_dataframe_accessor(name: str) -> Callable[[type[T]], type[T]]: ...
def register_series_accessor(name: str) -> Callable[[type[T]], type[T]]: ...
def register_index_accessor(name: str) -> Callable[[type[T]], type[T]]: ...
