from pyspark.pandas._typing import Dtype as Dtype, IndexOpsLike as IndexOpsLike, SeriesOrIndex as SeriesOrIndex
from pyspark.pandas.base import IndexOpsMixin as IndexOpsMixin, column_op as column_op
from pyspark.pandas.data_type_ops.base import DataTypeOps as DataTypeOps
from pyspark.pandas.typedef import pandas_on_spark_type as pandas_on_spark_type
from pyspark.sql import Column as Column
from pyspark.sql.types import ArrayType as ArrayType, BooleanType as BooleanType, NumericType as NumericType, StringType as StringType
from typing import Any

class ArrayOps(DataTypeOps):
    @property
    def pretty_name(self) -> str: ...
    def add(self, left: IndexOpsLike, right: Any) -> SeriesOrIndex: ...
    def lt(self, left: IndexOpsLike, right: Any) -> SeriesOrIndex: ...
    def le(self, left: IndexOpsLike, right: Any) -> SeriesOrIndex: ...
    def ge(self, left: IndexOpsLike, right: Any) -> SeriesOrIndex: ...
    def gt(self, left: IndexOpsLike, right: Any) -> SeriesOrIndex: ...
    def astype(self, index_ops: IndexOpsLike, dtype: str | type | Dtype) -> IndexOpsLike: ...

class MapOps(DataTypeOps):
    @property
    def pretty_name(self) -> str: ...

class StructOps(DataTypeOps):
    @property
    def pretty_name(self) -> str: ...
    def lt(self, left: IndexOpsLike, right: Any) -> SeriesOrIndex: ...
    def le(self, left: IndexOpsLike, right: Any) -> SeriesOrIndex: ...
    def ge(self, left: IndexOpsLike, right: Any) -> SeriesOrIndex: ...
    def gt(self, left: IndexOpsLike, right: Any) -> SeriesOrIndex: ...
