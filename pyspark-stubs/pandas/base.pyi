import pandas as pd
from abc import ABCMeta, abstractmethod
from pyspark.pandas._typing import Axis as Axis, Dtype as Dtype, IndexOpsLike as IndexOpsLike, Label as Label, SeriesOrIndex as SeriesOrIndex
from pyspark.pandas.config import get_option as get_option, option_context as option_context
from pyspark.pandas.data_type_ops.base import DataTypeOps as DataTypeOps
from pyspark.pandas.frame import DataFrame as DataFrame
from pyspark.pandas.internal import InternalField as InternalField, InternalFrame as InternalFrame, NATURAL_ORDER_COLUMN_NAME as NATURAL_ORDER_COLUMN_NAME, SPARK_DEFAULT_INDEX_NAME as SPARK_DEFAULT_INDEX_NAME
from pyspark.pandas.series import Series as Series
from pyspark.pandas.spark.accessors import SparkIndexOpsMethods as SparkIndexOpsMethods
from pyspark.pandas.typedef import extension_dtypes as extension_dtypes
from pyspark.pandas.utils import ERROR_MESSAGE_CANNOT_COMBINE as ERROR_MESSAGE_CANNOT_COMBINE, combine_frames as combine_frames, same_anchor as same_anchor, scol_for as scol_for, validate_axis as validate_axis
from pyspark.sql import Column as Column, Window as Window
from pyspark.sql._typing import ColumnOrName as ColumnOrName
from pyspark.sql.types import BooleanType as BooleanType, LongType as LongType, NumericType as NumericType
from typing import Any, Callable, Sequence

def should_alignment_for_column_op(self, other: SeriesOrIndex) -> bool: ...
def align_diff_index_ops(func: Callable[..., Column], this_index_ops: SeriesOrIndex, *args: Any) -> SeriesOrIndex: ...
def booleanize_null(scol: Column, f: Callable[..., Column]) -> Column: ...
def column_op(f: Callable[..., Column]) -> Callable[..., SeriesOrIndex]: ...
def numpy_column_op(f: Callable[..., Column]) -> Callable[..., SeriesOrIndex]: ...

class IndexOpsMixin(metaclass=ABCMeta):
    @property
    @abstractmethod
    def spark(self) -> SparkIndexOpsMethods[IndexOpsLike]: ...
    @abstractmethod
    def copy(self) -> IndexOpsLike: ...
    def __neg__(self) -> IndexOpsLike: ...
    def __add__(self, other: Any) -> SeriesOrIndex: ...
    def __sub__(self, other: Any) -> SeriesOrIndex: ...
    def __mul__(self, other: Any) -> SeriesOrIndex: ...
    def __truediv__(self, other: Any) -> SeriesOrIndex: ...
    def __mod__(self, other: Any) -> SeriesOrIndex: ...
    def __radd__(self, other: Any) -> SeriesOrIndex: ...
    def __rsub__(self, other: Any) -> SeriesOrIndex: ...
    def __rmul__(self, other: Any) -> SeriesOrIndex: ...
    def __rtruediv__(self, other: Any) -> SeriesOrIndex: ...
    def __floordiv__(self, other: Any) -> SeriesOrIndex: ...
    def __rfloordiv__(self, other: Any) -> SeriesOrIndex: ...
    def __rmod__(self, other: Any) -> SeriesOrIndex: ...
    def __pow__(self, other: Any) -> SeriesOrIndex: ...
    def __rpow__(self, other: Any) -> SeriesOrIndex: ...
    def __abs__(self) -> IndexOpsLike: ...
    def __eq__(self, other: Any) -> SeriesOrIndex: ...
    def __ne__(self, other: Any) -> SeriesOrIndex: ...
    def __lt__(self, other: Any) -> SeriesOrIndex: ...
    def __le__(self, other: Any) -> SeriesOrIndex: ...
    def __ge__(self, other: Any) -> SeriesOrIndex: ...
    def __gt__(self, other: Any) -> SeriesOrIndex: ...
    def __invert__(self) -> IndexOpsLike: ...
    def __and__(self, other: Any) -> SeriesOrIndex: ...
    def __or__(self, other: Any) -> SeriesOrIndex: ...
    def __rand__(self, other: Any) -> SeriesOrIndex: ...
    def __ror__(self, other: Any) -> SeriesOrIndex: ...
    def __xor__(self, other: Any) -> SeriesOrIndex: ...
    def __rxor__(self, other: Any) -> SeriesOrIndex: ...
    def __len__(self) -> int: ...
    def __array_ufunc__(self, ufunc: Callable, method: str, *inputs: Any, **kwargs: Any) -> SeriesOrIndex: ...
    @property
    def dtype(self) -> Dtype: ...
    @property
    def empty(self) -> bool: ...
    @property
    def hasnans(self) -> bool: ...
    @property
    def is_monotonic(self) -> bool: ...
    @property
    def is_monotonic_increasing(self) -> bool: ...
    @property
    def is_monotonic_decreasing(self) -> bool: ...
    @property
    def ndim(self) -> int: ...
    def astype(self, dtype: str | type | Dtype) -> IndexOpsLike: ...
    def isin(self, values: Sequence[Any]) -> IndexOpsLike: ...
    def isnull(self) -> IndexOpsLike: ...
    isna = isnull
    def notnull(self) -> IndexOpsLike: ...
    notna = notnull
    def all(self, axis: Axis = 0, skipna: bool = True) -> bool: ...
    def any(self, axis: Axis = 0) -> bool: ...
    def shift(self, periods: int = 1, fill_value: Any | None = None) -> IndexOpsLike: ...
    def value_counts(self, normalize: bool = False, sort: bool = True, ascending: bool = False, bins: None = None, dropna: bool = True) -> Series: ...
    def nunique(self, dropna: bool = True, approx: bool = False, rsd: float = 0.05) -> int: ...
    def take(self, indices: Sequence[int]) -> IndexOpsLike: ...
    def factorize(self, sort: bool = True, na_sentinel: int | None = -1) -> tuple[IndexOpsLike, pd.Index]: ...