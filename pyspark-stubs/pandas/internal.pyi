import pandas as pd
from _typeshed import Incomplete
from pandas.api.types import CategoricalDtype as CategoricalDtype
from pyspark._globals import _NoValueType
from pyspark.pandas._typing import Label as Label
from pyspark.pandas.data_type_ops.base import DataTypeOps as DataTypeOps
from pyspark.pandas.series import Series as Series
from pyspark.pandas.spark.utils import as_nullable_spark_type as as_nullable_spark_type, force_decimal_precision_scale as force_decimal_precision_scale
from pyspark.pandas.typedef import Dtype as Dtype, as_spark_type as as_spark_type, extension_dtypes as extension_dtypes, infer_pd_series_spark_type as infer_pd_series_spark_type, spark_type_to_pandas_dtype as spark_type_to_pandas_dtype
from pyspark.pandas.utils import column_labels_level as column_labels_level, default_session as default_session, is_name_like_tuple as is_name_like_tuple, is_testing as is_testing, lazy_property as lazy_property, name_like_string as name_like_string, scol_for as scol_for, spark_column_equals as spark_column_equals
from pyspark.sql import Column as PySparkColumn, DataFrame as PySparkDataFrame, Window as Window
from pyspark.sql.types import BooleanType as BooleanType, DataType as DataType, LongType as LongType, StringType as StringType, StructField as StructField, StructType as StructType
from pyspark.sql.utils import get_column_class as get_column_class, get_dataframe_class as get_dataframe_class, is_remote as is_remote, is_timestamp_ntz_preferred as is_timestamp_ntz_preferred
from typing import Any, Sequence

SPARK_INDEX_NAME_FORMAT: Incomplete
SPARK_DEFAULT_INDEX_NAME: Incomplete
SPARK_INDEX_NAME_PATTERN: Incomplete
NATURAL_ORDER_COLUMN_NAME: str
HIDDEN_COLUMNS: Incomplete
DEFAULT_SERIES_NAME: int
SPARK_DEFAULT_SERIES_NAME: Incomplete

class InternalField:
    def __init__(self, dtype: Dtype, struct_field: StructField | None = None) -> None: ...
    @staticmethod
    def from_struct_field(struct_field: StructField, *, use_extension_dtypes: bool = False) -> InternalField: ...
    @property
    def dtype(self) -> Dtype: ...
    @property
    def struct_field(self) -> StructField | None: ...
    @property
    def name(self) -> str: ...
    @property
    def spark_type(self) -> DataType: ...
    @property
    def nullable(self) -> bool: ...
    @property
    def metadata(self) -> dict[str, Any]: ...
    @property
    def is_extension_dtype(self) -> bool: ...
    def normalize_spark_type(self) -> InternalField: ...
    def copy(self, *, name: str | _NoValueType = ..., dtype: Dtype | _NoValueType = ..., spark_type: DataType | _NoValueType = ..., nullable: bool | _NoValueType = ..., metadata: dict[str, Any] | None | _NoValueType = ...) -> InternalField: ...
    def __eq__(self, other: Any) -> bool: ...

class InternalFrame:
    def __init__(self, spark_frame: PySparkDataFrame, index_spark_columns: list[PySparkColumn] | None, index_names: list[Label | None] | None = None, index_fields: list[InternalField] | None = None, column_labels: list[Label] | None = None, data_spark_columns: list[PySparkColumn] | None = None, data_fields: list[InternalField] | None = None, column_label_names: list[Label | None] | None = None) -> None: ...
    @staticmethod
    def attach_default_index(sdf: PySparkDataFrame, default_index_type: str | None = None) -> PySparkDataFrame: ...
    @staticmethod
    def attach_sequence_column(sdf: PySparkDataFrame, column_name: str) -> PySparkDataFrame: ...
    @staticmethod
    def attach_distributed_column(sdf: PySparkDataFrame, column_name: str) -> PySparkDataFrame: ...
    @staticmethod
    def attach_distributed_sequence_column(sdf: PySparkDataFrame, column_name: str) -> PySparkDataFrame: ...
    def spark_column_for(self, label: Label) -> PySparkColumn: ...
    def spark_column_name_for(self, label_or_scol: Label | PySparkColumn) -> str: ...
    def spark_type_for(self, label_or_scol: Label | PySparkColumn) -> DataType: ...
    def spark_column_nullable_for(self, label_or_scol: Label | PySparkColumn) -> bool: ...
    def field_for(self, label: Label) -> InternalField: ...
    @property
    def spark_frame(self) -> PySparkDataFrame: ...
    def data_spark_column_names(self) -> list[str]: ...
    @property
    def data_spark_columns(self) -> list[PySparkColumn]: ...
    @property
    def index_spark_column_names(self) -> list[str]: ...
    @property
    def index_spark_columns(self) -> list[PySparkColumn]: ...
    def spark_column_names(self) -> list[str]: ...
    def spark_columns(self) -> list[PySparkColumn]: ...
    @property
    def index_names(self) -> list[Label | None]: ...
    def index_level(self) -> int: ...
    @property
    def column_labels(self) -> list[Label]: ...
    def column_labels_level(self) -> int: ...
    @property
    def column_label_names(self) -> list[Label | None]: ...
    @property
    def index_fields(self) -> list[InternalField]: ...
    @property
    def data_fields(self) -> list[InternalField]: ...
    def to_internal_spark_frame(self) -> PySparkDataFrame: ...
    def to_pandas_frame(self) -> pd.DataFrame: ...
    def arguments_for_restore_index(self) -> dict: ...
    @staticmethod
    def restore_index(pdf: pd.DataFrame, *, index_columns: list[str], index_names: list[Label], data_columns: list[str], column_labels: list[Label], column_label_names: list[Label], fields: list[InternalField] = None) -> pd.DataFrame: ...
    def resolved_copy(self) -> InternalFrame: ...
    def with_new_sdf(self, spark_frame: PySparkDataFrame, *, index_fields: list[InternalField] | None = None, data_columns: list[str] | None = None, data_fields: list[InternalField] | None = None) -> InternalFrame: ...
    def with_new_columns(self, scols_or_pssers: Sequence[PySparkColumn | Series], *, column_labels: list[Label] | None = None, data_fields: list[InternalField] | None = None, column_label_names: list[Label | None] | None | _NoValueType = ..., keep_order: bool = True) -> InternalFrame: ...
    def with_filter(self, pred: PySparkColumn | Series) -> InternalFrame: ...
    def with_new_spark_column(self, column_label: Label, scol: PySparkColumn, *, field: InternalField | None = None, keep_order: bool = True) -> InternalFrame: ...
    def select_column(self, column_label: Label) -> InternalFrame: ...
    def copy(self, *, spark_frame: PySparkDataFrame | _NoValueType = ..., index_spark_columns: list[PySparkColumn] | _NoValueType = ..., index_names: list[Label | None] | None | _NoValueType = ..., index_fields: list[InternalField] | None | _NoValueType = ..., column_labels: list[Label] | None | _NoValueType = ..., data_spark_columns: list[PySparkColumn] | None | _NoValueType = ..., data_fields: list[InternalField] | None | _NoValueType = ..., column_label_names: list[Label | None] | None | _NoValueType = ...) -> InternalFrame: ...
    @staticmethod
    def from_pandas(pdf: pd.DataFrame) -> InternalFrame: ...
    @staticmethod
    def prepare_pandas_frame(pdf: pd.DataFrame, *, retain_index: bool = True, prefer_timestamp_ntz: bool = False) -> tuple[pd.DataFrame, list[str], list[InternalField], list[str], list[InternalField]]: ...