from _typeshed import Incomplete
from py4j.java_gateway import JavaObject
from pyspark.sql._typing import LiteralType
from pyspark.sql.column import Column
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.pandas.group_ops import PandasGroupedOpsMixin
from typing import overload

__all__ = ['GroupedData']

class GroupedData(PandasGroupedOpsMixin):
    session: Incomplete
    def __init__(self, jgd: JavaObject, df: DataFrame) -> None: ...
    @overload
    def agg(self, *exprs: Column) -> DataFrame: ...
    @overload
    def agg(self, /, __exprs: dict[str, str]) -> DataFrame: ...
    def count(self) -> DataFrame: ...
    def mean(self, *cols: str) -> DataFrame: ...
    def avg(self, *cols: str) -> DataFrame: ...
    def max(self, *cols: str) -> DataFrame: ...
    def min(self, *cols: str) -> DataFrame: ...
    def sum(self, *cols: str) -> DataFrame: ...
    def pivot(self, pivot_col: str, values: list['LiteralType'] | None = None) -> GroupedData: ...
