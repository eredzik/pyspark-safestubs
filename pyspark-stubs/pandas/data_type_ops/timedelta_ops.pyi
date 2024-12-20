import pandas as pd
from pyspark.pandas._typing import Dtype as Dtype, IndexOpsLike as IndexOpsLike, SeriesOrIndex as SeriesOrIndex
from pyspark.pandas.base import IndexOpsMixin as IndexOpsMixin
from pyspark.pandas.data_type_ops.base import DataTypeOps as DataTypeOps
from pyspark.pandas.typedef import pandas_on_spark_type as pandas_on_spark_type
from pyspark.sql.types import BooleanType as BooleanType, DayTimeIntervalType as DayTimeIntervalType, StringType as StringType
from pyspark.sql.utils import pyspark_column_op as pyspark_column_op
from typing import Any

class TimedeltaOps(DataTypeOps):
    @property
    def pretty_name(self) -> str: ...
    def astype(self, index_ops: IndexOpsLike, dtype: str | type | Dtype) -> IndexOpsLike: ...
    def prepare(self, col: pd.Series) -> pd.Series: ...
    def sub(self, left: IndexOpsLike, right: Any) -> SeriesOrIndex: ...
    def rsub(self, left: IndexOpsLike, right: Any) -> SeriesOrIndex: ...
    def lt(self, left: IndexOpsLike, right: Any) -> SeriesOrIndex: ...
    def le(self, left: IndexOpsLike, right: Any) -> SeriesOrIndex: ...
    def ge(self, left: IndexOpsLike, right: Any) -> SeriesOrIndex: ...
    def gt(self, left: IndexOpsLike, right: Any) -> SeriesOrIndex: ...
