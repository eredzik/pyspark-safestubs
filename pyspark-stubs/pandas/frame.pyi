import datetime
import numpy as np
import pandas as pd
from _typeshed import Incomplete
from collections.abc import Mapping
from pandas.io.formats.style import Styler as Styler
from pandas.tseries.frequencies import DateOffset as DateOffset
from pyspark import StorageLevel as StorageLevel
from pyspark.pandas._typing import Axis as Axis, DataFrameOrSeries as DataFrameOrSeries, Dtype as Dtype, Label as Label, Name as Name, Scalar as Scalar, T as T
from pyspark.pandas.accessors import PandasOnSparkFrameMethods as PandasOnSparkFrameMethods
from pyspark.pandas.config import get_option as get_option, option_context as option_context
from pyspark.pandas.correlation import CORRELATION_CORR_OUTPUT_COLUMN as CORRELATION_CORR_OUTPUT_COLUMN, CORRELATION_COUNT_OUTPUT_COLUMN as CORRELATION_COUNT_OUTPUT_COLUMN, CORRELATION_VALUE_1_COLUMN as CORRELATION_VALUE_1_COLUMN, CORRELATION_VALUE_2_COLUMN as CORRELATION_VALUE_2_COLUMN, compute as compute
from pyspark.pandas.generic import Frame as Frame
from pyspark.pandas.groupby import DataFrameGroupBy as DataFrameGroupBy
from pyspark.pandas.indexes import Index as Index
from pyspark.pandas.internal import HIDDEN_COLUMNS as HIDDEN_COLUMNS, InternalField as InternalField, InternalFrame as InternalFrame, NATURAL_ORDER_COLUMN_NAME as NATURAL_ORDER_COLUMN_NAME, SPARK_DEFAULT_INDEX_NAME as SPARK_DEFAULT_INDEX_NAME, SPARK_DEFAULT_SERIES_NAME as SPARK_DEFAULT_SERIES_NAME, SPARK_INDEX_NAME_FORMAT as SPARK_INDEX_NAME_FORMAT, SPARK_INDEX_NAME_PATTERN as SPARK_INDEX_NAME_PATTERN
from pyspark.pandas.missing.frame import MissingPandasLikeDataFrame as MissingPandasLikeDataFrame
from pyspark.pandas.plot import PandasOnSparkPlotAccessor as PandasOnSparkPlotAccessor
from pyspark.pandas.resample import DataFrameResampler as DataFrameResampler
from pyspark.pandas.series import Series as Series
from pyspark.pandas.spark.accessors import CachedSparkFrameMethods as CachedSparkFrameMethods, SparkFrameMethods as SparkFrameMethods
from pyspark.pandas.typedef.typehints import DataFrameType as DataFrameType, ScalarType as ScalarType, SeriesType as SeriesType, as_spark_type as as_spark_type, create_tuple_for_frame_type as create_tuple_for_frame_type, infer_return_type as infer_return_type, pandas_on_spark_type as pandas_on_spark_type, spark_type_to_pandas_dtype as spark_type_to_pandas_dtype
from pyspark.pandas.utils import align_diff_frames as align_diff_frames, column_labels_level as column_labels_level, combine_frames as combine_frames, default_session as default_session, is_name_like_tuple as is_name_like_tuple, is_name_like_value as is_name_like_value, is_testing as is_testing, log_advice as log_advice, name_like_string as name_like_string, same_anchor as same_anchor, scol_for as scol_for, validate_arguments_and_invoke_function as validate_arguments_and_invoke_function, validate_axis as validate_axis, validate_bool_kwarg as validate_bool_kwarg, validate_how as validate_how, validate_mode as validate_mode, verify_temp_column_name as verify_temp_column_name
from pyspark.sql import DataFrame as PySparkDataFrame
from pyspark.sql._typing import OptionalPrimitiveType as OptionalPrimitiveType
from pyspark.sql.functions import pandas_udf as pandas_udf
from pyspark.sql.types import ArrayType as ArrayType, BooleanType as BooleanType, DataType as DataType, DecimalType as DecimalType, DoubleType as DoubleType, NumericType as NumericType, Row as Row, StringType as StringType, StructField as StructField, StructType as StructType, TimestampNTZType as TimestampNTZType, TimestampType as TimestampType
from pyspark.sql.utils import get_column_class as get_column_class, get_dataframe_class as get_dataframe_class
from pyspark.sql.window import Window as Window
from types import TracebackType
from typing import Any, Callable, Generic, IO, Iterable, Iterator, Sequence

REPR_PATTERN: Incomplete
REPR_HTML_PATTERN: Incomplete

class DataFrame(Frame, Generic[T]):
    def __init__(self, data: Incomplete | None = None, index: Incomplete | None = None, columns: Incomplete | None = None, dtype: Incomplete | None = None, copy: bool = False) -> None: ...
    @property
    def ndim(self) -> int: ...
    @property
    def axes(self) -> list: ...
    def __add__(self, other: Any) -> DataFrame: ...
    def __radd__(self, other: Any) -> DataFrame: ...
    def __truediv__(self, other: Any) -> DataFrame: ...
    def __rtruediv__(self, other: Any) -> DataFrame: ...
    def __mul__(self, other: Any) -> DataFrame: ...
    def __rmul__(self, other: Any) -> DataFrame: ...
    def __sub__(self, other: Any) -> DataFrame: ...
    def __rsub__(self, other: Any) -> DataFrame: ...
    def __pow__(self, other: Any) -> DataFrame: ...
    def __rpow__(self, other: Any) -> DataFrame: ...
    def __mod__(self, other: Any) -> DataFrame: ...
    def __rmod__(self, other: Any) -> DataFrame: ...
    def __floordiv__(self, other: Any) -> DataFrame: ...
    def __rfloordiv__(self, other: Any) -> DataFrame: ...
    def __abs__(self) -> DataFrame: ...
    def __neg__(self) -> DataFrame: ...
    def add(self, other: Any) -> DataFrame: ...
    plot: Incomplete
    spark: Incomplete
    pandas_on_spark: Incomplete
    koalas: Incomplete
    def hist(self, bins: int = 10, **kwds): ...
    def boxplot(self, **kwds): ...
    def kde(self, bw_method: Incomplete | None = None, ind: Incomplete | None = None, **kwds): ...
    def radd(self, other: Any) -> DataFrame: ...
    def div(self, other: Any) -> DataFrame: ...
    divide = div
    def rdiv(self, other: Any) -> DataFrame: ...
    def truediv(self, other: Any) -> DataFrame: ...
    def rtruediv(self, other: Any) -> DataFrame: ...
    def mul(self, other: Any) -> DataFrame: ...
    multiply = mul
    def rmul(self, other: Any) -> DataFrame: ...
    def sub(self, other: Any) -> DataFrame: ...
    subtract = sub
    def rsub(self, other: Any) -> DataFrame: ...
    def mod(self, other: Any) -> DataFrame: ...
    def rmod(self, other: Any) -> DataFrame: ...
    def pow(self, other: Any) -> DataFrame: ...
    def rpow(self, other: Any) -> DataFrame: ...
    def floordiv(self, other: Any) -> DataFrame: ...
    def rfloordiv(self, other: Any) -> DataFrame: ...
    def __eq__(self, other: Any) -> DataFrame: ...
    def __ne__(self, other: Any) -> DataFrame: ...
    def __lt__(self, other: Any) -> DataFrame: ...
    def __le__(self, other: Any) -> DataFrame: ...
    def __ge__(self, other: Any) -> DataFrame: ...
    def __gt__(self, other: Any) -> DataFrame: ...
    def eq(self, other: Any) -> DataFrame: ...
    equals = eq
    def gt(self, other: Any) -> DataFrame: ...
    def ge(self, other: Any) -> DataFrame: ...
    def lt(self, other: Any) -> DataFrame: ...
    def le(self, other: Any) -> DataFrame: ...
    def ne(self, other: Any) -> DataFrame: ...
    def applymap(self, func: Callable[[Any], Any]) -> DataFrame: ...
    def aggregate(self, func: list[str] | dict[Name, list[str]]) -> DataFrame: ...
    agg = aggregate
    def corr(self, method: str = 'pearson', min_periods: int | None = None) -> DataFrame: ...
    def corrwith(self, other: DataFrameOrSeries, axis: Axis = 0, drop: bool = False, method: str = 'pearson') -> Series: ...
    def items(self) -> Iterator[tuple[Name, 'Series']]: ...
    def iterrows(self) -> Iterator[tuple[Name, pd.Series]]: ...
    def itertuples(self, index: bool = True, name: str | None = 'PandasOnSpark') -> Iterator[tuple]: ...
    def iteritems(self) -> Iterator[tuple[Name, 'Series']]: ...
    def to_clipboard(self, excel: bool = True, sep: str | None = None, **kwargs: Any) -> None: ...
    def to_html(self, buf: IO[str] | None = None, columns: Sequence[Name] | None = None, col_space: str | int | dict[Name, str | int] | None = None, header: bool = True, index: bool = True, na_rep: str = 'NaN', formatters: list[Callable[[Any], str]] | dict[Name, Callable[[Any], str]] | None = None, float_format: Callable[[float], str] | None = None, sparsify: bool | None = None, index_names: bool = True, justify: str | None = None, max_rows: int | None = None, max_cols: int | None = None, show_dimensions: bool = False, decimal: str = '.', bold_rows: bool = True, classes: str | list | tuple | None = None, escape: bool = True, notebook: bool = False, border: int | None = None, table_id: str | None = None, render_links: bool = False) -> str | None: ...
    def to_string(self, buf: IO[str] | None = None, columns: Sequence[Name] | None = None, col_space: str | int | dict[Name, str | int] | None = None, header: bool = True, index: bool = True, na_rep: str = 'NaN', formatters: list[Callable[[Any], str]] | dict[Name, Callable[[Any], str]] | None = None, float_format: Callable[[float], str] | None = None, sparsify: bool | None = None, index_names: bool = True, justify: str | None = None, max_rows: int | None = None, max_cols: int | None = None, show_dimensions: bool = False, decimal: str = '.', line_width: int | None = None) -> str | None: ...
    def to_dict(self, orient: str = 'dict', into: type = ...) -> list | Mapping: ...
    def to_latex(self, buf: IO[str] | None = None, columns: list[Name] | None = None, col_space: int | None = None, header: bool = True, index: bool = True, na_rep: str = 'NaN', formatters: list[Callable[[Any], str]] | dict[Name, Callable[[Any], str]] | None = None, float_format: Callable[[float], str] | None = None, sparsify: bool | None = None, index_names: bool = True, bold_rows: bool = False, column_format: str | None = None, longtable: bool | None = None, escape: bool | None = None, encoding: str | None = None, decimal: str = '.', multicolumn: bool | None = None, multicolumn_format: str | None = None, multirow: bool | None = None) -> str | None: ...
    def transpose(self) -> DataFrame: ...
    T: Incomplete
    def apply(self, func: Callable, axis: Axis = 0, args: Sequence[Any] = (), **kwds: Any) -> Series | DataFrame | Index: ...
    def transform(self, func: Callable[..., 'Series'], axis: Axis = 0, *args: Any, **kwargs: Any) -> DataFrame: ...
    def pop(self, item: Name) -> DataFrame: ...
    def xs(self, key: Name, axis: Axis = 0, level: int | None = None) -> DataFrameOrSeries: ...
    def between_time(self, start_time: datetime.time | str, end_time: datetime.time | str, include_start: bool = True, include_end: bool = True, axis: Axis = 0) -> DataFrame: ...
    def at_time(self, time: datetime.time | str, asof: bool = False, axis: Axis = 0) -> DataFrame: ...
    def where(self, cond: DataFrameOrSeries, other: DataFrameOrSeries | Any = ..., axis: Axis = None) -> DataFrame: ...
    def mask(self, cond: DataFrameOrSeries, other: DataFrameOrSeries | Any = ...) -> DataFrame: ...
    @property
    def index(self) -> Index: ...
    @property
    def empty(self) -> bool: ...
    @property
    def style(self) -> Styler: ...
    def set_index(self, keys: Name | list[Name], drop: bool = True, append: bool = False, inplace: bool = False) -> DataFrame | None: ...
    def reset_index(self, level: int | Name | Sequence[int | Name] | None = None, drop: bool = False, inplace: bool = False, col_level: int = 0, col_fill: str = '') -> DataFrame | None: ...
    def isnull(self) -> DataFrame: ...
    isna = isnull
    def notnull(self) -> DataFrame: ...
    notna = notnull
    def insert(self, loc: int, column: Name, value: Scalar | Series | Iterable, allow_duplicates: bool = False) -> None: ...
    def shift(self, periods: int = 1, fill_value: Any | None = None) -> DataFrame: ...
    def diff(self, periods: int = 1, axis: Axis = 0) -> DataFrame: ...
    def nunique(self, axis: Axis = 0, dropna: bool = True, approx: bool = False, rsd: float = 0.05) -> Series: ...
    def round(self, decimals: int | dict[Name, int] | Series = 0) -> DataFrame: ...
    def duplicated(self, subset: Name | list[Name] | None = None, keep: bool | str = 'first') -> Series: ...
    def dot(self, other: Series) -> Series: ...
    def __matmul__(self, other: Series) -> Series: ...
    def to_table(self, name: str, format: str | None = None, mode: str = 'w', partition_cols: str | list[str] | None = None, index_col: str | list[str] | None = None, **options: Any) -> None: ...
    def to_delta(self, path: str, mode: str = 'w', partition_cols: str | list[str] | None = None, index_col: str | list[str] | None = None, **options: OptionalPrimitiveType) -> None: ...
    def to_parquet(self, path: str, mode: str = 'w', partition_cols: str | list[str] | None = None, compression: str | None = None, index_col: str | list[str] | None = None, **options: Any) -> None: ...
    def to_orc(self, path: str, mode: str = 'w', partition_cols: str | list[str] | None = None, index_col: str | list[str] | None = None, **options: OptionalPrimitiveType) -> None: ...
    def to_spark_io(self, path: str | None = None, format: str | None = None, mode: str = 'overwrite', partition_cols: str | list[str] | None = None, index_col: str | list[str] | None = None, **options: OptionalPrimitiveType) -> None: ...
    def to_spark(self, index_col: str | list[str] | None = None) -> PySparkDataFrame: ...
    def to_pandas(self) -> pd.DataFrame: ...
    def assign(self, **kwargs: Any) -> DataFrame: ...
    @staticmethod
    def from_records(data: np.ndarray | list[tuple] | dict | pd.DataFrame, index: str | list | np.ndarray = None, exclude: list = None, columns: list = None, coerce_float: bool = False, nrows: int = None) -> DataFrame: ...
    def to_records(self, index: bool = True, column_dtypes: str | Dtype | dict[Name, str | Dtype] | None = None, index_dtypes: str | Dtype | dict[Name, str | Dtype] | None = None) -> np.recarray: ...
    def copy(self, deep: bool = True) -> DataFrame: ...
    def dropna(self, axis: Axis = 0, how: str = 'any', thresh: int | None = None, subset: Name | list[Name] | None = None, inplace: bool = False) -> DataFrame | None: ...
    def fillna(self, value: Any | dict[Name, Any] | None = None, method: str | None = None, axis: Axis | None = None, inplace: bool = False, limit: int | None = None) -> DataFrame | None: ...
    def interpolate(self, method: str = 'linear', limit: int | None = None, limit_direction: str | None = None, limit_area: str | None = None) -> DataFrame: ...
    def replace(self, to_replace: Any | list | tuple | dict | None = None, value: Any | None = None, inplace: bool = False, limit: int | None = None, regex: bool = False, method: str = 'pad') -> DataFrame | None: ...
    def clip(self, lower: float | int = None, upper: float | int = None) -> DataFrame: ...
    def head(self, n: int = 5) -> DataFrame: ...
    def last(self, offset: str | DateOffset) -> DataFrame: ...
    def first(self, offset: str | DateOffset) -> DataFrame: ...
    def pivot_table(self, values: Name | list[Name] | None = None, index: list[Name] | None = None, columns: Name | None = None, aggfunc: str | dict[Name, str] = 'mean', fill_value: Any | None = None) -> DataFrame: ...
    def pivot(self, index: Name | None = None, columns: Name | None = None, values: Name | None = None) -> DataFrame: ...
    @property
    def columns(self) -> pd.Index: ...
    @columns.setter
    def columns(self, columns: pd.Index | list[Name]) -> None: ...
    @property
    def dtypes(self) -> pd.Series: ...
    def select_dtypes(self, include: str | list[str] | None = None, exclude: str | list[str] | None = None) -> DataFrame: ...
    def droplevel(self, level: int | Name | list[int | Name], axis: Axis = 0) -> DataFrame: ...
    def drop(self, labels: Name | list[Name] | None = None, axis: Axis | None = 0, index: Name | list[Name] = None, columns: Name | list[Name] = None) -> DataFrame: ...
    def sort_values(self, by: Name | list[Name], ascending: bool | list[bool] = True, inplace: bool = False, na_position: str = 'last', ignore_index: bool = False) -> DataFrame | None: ...
    def sort_index(self, axis: Axis = 0, level: int | list[int] | None = None, ascending: bool = True, inplace: bool = False, kind: str = None, na_position: str = 'last', ignore_index: bool = False) -> DataFrame | None: ...
    def swaplevel(self, i: int | Name = -2, j: int | Name = -1, axis: Axis = 0) -> DataFrame: ...
    def swapaxes(self, i: Axis, j: Axis, copy: bool = True) -> DataFrame: ...
    def nlargest(self, n: int, columns: Name | list[Name], keep: str = 'first') -> DataFrame: ...
    def nsmallest(self, n: int, columns: Name | list[Name], keep: str = 'first') -> DataFrame: ...
    def isin(self, values: list | dict) -> DataFrame: ...
    @property
    def shape(self) -> tuple[int, int]: ...
    def merge(self, right: DataFrame, how: str = 'inner', on: Name | list[Name] | None = None, left_on: Name | list[Name] | None = None, right_on: Name | list[Name] | None = None, left_index: bool = False, right_index: bool = False, suffixes: tuple[str, str] = ('_x', '_y')) -> DataFrame: ...
    def join(self, right: DataFrame, on: Name | list[Name] | None = None, how: str = 'left', lsuffix: str = '', rsuffix: str = '') -> DataFrame: ...
    def combine_first(self, other: DataFrame) -> DataFrame: ...
    def append(self, other: DataFrame, ignore_index: bool = False, verify_integrity: bool = False, sort: bool = False) -> DataFrame: ...
    def update(self, other: DataFrame, join: str = 'left', overwrite: bool = True) -> None: ...
    def cov(self, min_periods: int | None = None, ddof: int = 1) -> DataFrame: ...
    def sample(self, n: int | None = None, frac: float | None = None, replace: bool = False, random_state: int | None = None, ignore_index: bool = False) -> DataFrame: ...
    def astype(self, dtype: str | Dtype | dict[Name, str | Dtype]) -> DataFrame: ...
    def add_prefix(self, prefix: str) -> DataFrame: ...
    def add_suffix(self, suffix: str) -> DataFrame: ...
    def describe(self, percentiles: list[float] | None = None) -> DataFrame: ...
    def drop_duplicates(self, subset: Name | list[Name] | None = None, keep: bool | str = 'first', inplace: bool = False, ignore_index: bool = False) -> DataFrame | None: ...
    def reindex(self, labels: Sequence[Any] | None = None, index: Index | Sequence[Any] | None = None, columns: pd.Index | Sequence[Any] | None = None, axis: Axis | None = None, copy: bool | None = True, fill_value: Any | None = None) -> DataFrame: ...
    def reindex_like(self, other: DataFrame, copy: bool = True) -> DataFrame: ...
    def melt(self, id_vars: Name | list[Name] | None = None, value_vars: Name | list[Name] | None = None, var_name: str | list[str] | None = None, value_name: str = 'value') -> DataFrame: ...
    def stack(self) -> DataFrameOrSeries: ...
    def unstack(self) -> DataFrameOrSeries: ...
    def all(self, axis: Axis = 0, bool_only: bool | None = None, skipna: bool = True) -> Series: ...
    def any(self, axis: Axis = 0, bool_only: bool | None = None) -> Series: ...
    def rank(self, method: str = 'average', ascending: bool = True, numeric_only: bool | None = None) -> DataFrame: ...
    def filter(self, items: Sequence[Any] | None = None, like: str | None = None, regex: str | None = None, axis: Axis | None = None) -> DataFrame: ...
    def rename(self, mapper: dict | Callable[[Any], Any] | None = None, index: dict | Callable[[Any], Any] | None = None, columns: dict | Callable[[Any], Any] | None = None, axis: Axis = 'index', inplace: bool = False, level: int | None = None, errors: str = 'ignore') -> DataFrame | None: ...
    def rename_axis(self, mapper: Any | Sequence[Any] | dict[Name, Any] | Callable[[Name], Any] = None, index: Any | Sequence[Any] | dict[Name, Any] | Callable[[Name], Any] = None, columns: Any | Sequence[Any] | dict[Name, Any] | Callable[[Name], Any] = None, axis: Axis | None = 0, inplace: bool | None = False) -> DataFrame | None: ...
    def keys(self) -> pd.Index: ...
    def pct_change(self, periods: int = 1) -> DataFrame: ...
    def idxmax(self, axis: Axis = 0) -> Series: ...
    def idxmin(self, axis: Axis = 0) -> Series: ...
    count: Incomplete
    def info(self, verbose: bool | None = None, buf: IO[str] | None = None, max_cols: int | None = None) -> None: ...
    def quantile(self, q: float | Iterable[float] = 0.5, axis: Axis = 0, numeric_only: bool = True, accuracy: int = 10000) -> DataFrameOrSeries: ...
    def query(self, expr: str, inplace: bool = False) -> DataFrame | None: ...
    def take(self, indices: list[int], axis: Axis = 0, **kwargs: Any) -> DataFrame: ...
    def eval(self, expr: str, inplace: bool = False) -> DataFrameOrSeries | None: ...
    def explode(self, column: Name, ignore_index: bool = False) -> DataFrame: ...
    def mad(self, axis: Axis = 0) -> Series: ...
    def mode(self, axis: Axis = 0, numeric_only: bool = False, dropna: bool = True) -> DataFrame: ...
    def tail(self, n: int = 5) -> DataFrame: ...
    def align(self, other: DataFrameOrSeries, join: str = 'outer', axis: Axis | None = None, copy: bool = True) -> tuple['DataFrame', DataFrameOrSeries]: ...
    @staticmethod
    def from_dict(data: dict[Name, Sequence[Any]], orient: str = 'columns', dtype: str | Dtype = None, columns: list[Name] | None = None) -> DataFrame: ...
    def groupby(self, by: Name | Series | list[Name | Series], axis: Axis = 0, as_index: bool = True, dropna: bool = True) -> DataFrameGroupBy: ...
    def resample(self, rule: str, closed: str | None = None, label: str | None = None, on: Series | None = None) -> DataFrameResampler: ...
    def __getitem__(self, key: Any) -> Any: ...
    def __setitem__(self, key: Any, value: Any) -> None: ...
    def __getattr__(self, key: str) -> Any: ...
    def __setattr__(self, key: str, value: Any) -> None: ...
    def __len__(self) -> int: ...
    def __dir__(self) -> Iterable[str]: ...
    def __iter__(self) -> Iterator[Name]: ...
    def __array_ufunc__(self, ufunc: Callable, method: str, *inputs: Any, **kwargs: Any) -> DataFrame: ...
    def __class_getitem__(cls, params: Any) -> object: ...

class CachedDataFrame(DataFrame):
    def __init__(self, internal: InternalFrame, storage_level: StorageLevel | None = None) -> None: ...
    def __enter__(self) -> CachedDataFrame: ...
    def __exit__(self, exception_type: type[BaseException] | None, exception_value: BaseException | None, traceback: TracebackType | None) -> bool | None: ...
    spark: Incomplete
