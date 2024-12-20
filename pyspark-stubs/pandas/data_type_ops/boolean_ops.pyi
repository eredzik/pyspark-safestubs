import pandas as pd
from pyspark.errors import PySparkValueError as PySparkValueError
from pyspark.pandas._typing import Dtype as Dtype, IndexOpsLike as IndexOpsLike, SeriesOrIndex as SeriesOrIndex
from pyspark.pandas.base import IndexOpsMixin as IndexOpsMixin, column_op as column_op
from pyspark.pandas.data_type_ops.base import DataTypeOps as DataTypeOps, is_valid_operand_for_numeric_arithmetic as is_valid_operand_for_numeric_arithmetic, transform_boolean_operand_to_numeric as transform_boolean_operand_to_numeric
from pyspark.pandas.typedef.typehints import as_spark_type as as_spark_type, extension_dtypes as extension_dtypes, pandas_on_spark_type as pandas_on_spark_type
from pyspark.sql.types import BooleanType as BooleanType, StringType as StringType
from pyspark.sql.utils import get_column_class as get_column_class
from typing import Any

class BooleanOps(DataTypeOps):
    @property
    def pretty_name(self) -> str: ...
    def add(self, left: IndexOpsLike, right: Any) -> SeriesOrIndex: ...
    def sub(self, left: IndexOpsLike, right: Any) -> SeriesOrIndex: ...
    def mul(self, left: IndexOpsLike, right: Any) -> SeriesOrIndex: ...
    def truediv(self, left: IndexOpsLike, right: Any) -> SeriesOrIndex: ...
    def floordiv(self, left: IndexOpsLike, right: Any) -> SeriesOrIndex: ...
    def mod(self, left: IndexOpsLike, right: Any) -> SeriesOrIndex: ...
    def pow(self, left: IndexOpsLike, right: Any) -> SeriesOrIndex: ...
    def radd(self, left: IndexOpsLike, right: Any) -> SeriesOrIndex: ...
    def rsub(self, left: IndexOpsLike, right: Any) -> SeriesOrIndex: ...
    def rmul(self, left: IndexOpsLike, right: Any) -> SeriesOrIndex: ...
    def rtruediv(self, left: IndexOpsLike, right: Any) -> SeriesOrIndex: ...
    def rfloordiv(self, left: IndexOpsLike, right: Any) -> SeriesOrIndex: ...
    def rpow(self, left: IndexOpsLike, right: Any) -> SeriesOrIndex: ...
    def rmod(self, left: IndexOpsLike, right: Any) -> SeriesOrIndex: ...
    def __and__(self, left: IndexOpsLike, right: Any) -> SeriesOrIndex: ...
    def xor(self, left: IndexOpsLike, right: Any) -> SeriesOrIndex: ...
    def __or__(self, left: IndexOpsLike, right: Any) -> SeriesOrIndex: ...
    def astype(self, index_ops: IndexOpsLike, dtype: str | type | Dtype) -> IndexOpsLike: ...
    def neg(self, operand: IndexOpsLike) -> IndexOpsLike: ...
    def abs(self, operand: IndexOpsLike) -> IndexOpsLike: ...
    def lt(self, left: IndexOpsLike, right: Any) -> SeriesOrIndex: ...
    def le(self, left: IndexOpsLike, right: Any) -> SeriesOrIndex: ...
    def ge(self, left: IndexOpsLike, right: Any) -> SeriesOrIndex: ...
    def gt(self, left: IndexOpsLike, right: Any) -> SeriesOrIndex: ...
    def invert(self, operand: IndexOpsLike) -> IndexOpsLike: ...

class BooleanExtensionOps(BooleanOps):
    @property
    def pretty_name(self) -> str: ...
    def __and__(self, left: IndexOpsLike, right: Any) -> SeriesOrIndex: ...
    def __or__(self, left: IndexOpsLike, right: Any) -> SeriesOrIndex: ...
    def xor(self, left: IndexOpsLike, right: Any) -> SeriesOrIndex: ...
    def restore(self, col: pd.Series) -> pd.Series: ...
    def neg(self, operand: IndexOpsLike) -> IndexOpsLike: ...
    def invert(self, operand: IndexOpsLike) -> IndexOpsLike: ...
    def abs(self, operand: IndexOpsLike) -> IndexOpsLike: ...
