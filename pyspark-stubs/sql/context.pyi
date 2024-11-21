from _typeshed import Incomplete
from py4j.java_gateway import JavaObject
from pyspark._globals import _NoValueType
from pyspark.context import SparkContext
from pyspark.rdd import RDD
from pyspark.sql._typing import AtomicValue, RowLike, UserDefinedFunctionLike
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.pandas._typing import DataFrameLike as PandasDataFrameLike
from pyspark.sql.readwriter import DataFrameReader
from pyspark.sql.session import SparkSession
from pyspark.sql.streaming import DataStreamReader, StreamingQueryManager
from pyspark.sql.types import AtomicType, DataType, StructType
from pyspark.sql.udf import UDFRegistration
from pyspark.sql.udtf import UDTFRegistration
from typing import Any, Callable, Iterable, overload

__all__ = ['SQLContext', 'HiveContext']

class SQLContext:
    sparkSession: Incomplete
    def __init__(self, sparkContext: SparkContext, sparkSession: SparkSession | None = None, jsqlContext: JavaObject | None = None) -> None: ...
    @classmethod
    def getOrCreate(cls, sc: SparkContext) -> SQLContext: ...
    def newSession(self) -> SQLContext: ...
    def setConf(self, key: str, value: bool | int | str) -> None: ...
    def getConf(self, key: str, defaultValue: str | None | _NoValueType = ...) -> str | None: ...
    @property
    def udf(self) -> UDFRegistration: ...
    @property
    def udtf(self) -> UDTFRegistration: ...
    def range(self, start: int, end: int | None = None, step: int = 1, numPartitions: int | None = None) -> DataFrame: ...
    def registerFunction(self, name: str, f: Callable[..., Any], returnType: DataType | None = None) -> UserDefinedFunctionLike: ...
    def registerJavaFunction(self, name: str, javaClassName: str, returnType: DataType | None = None) -> None: ...
    @overload
    def createDataFrame(self, data: RDD[RowLike] | Iterable['RowLike'], schema: list[str] | tuple[str, ...] = ..., samplingRatio: float | None = ...) -> DataFrame: ...
    @overload
    def createDataFrame(self, data: RDD[RowLike] | Iterable['RowLike'], schema: StructType | str, *, verifySchema: bool = ...) -> DataFrame: ...
    @overload
    def createDataFrame(self, data: RDD[AtomicValue] | Iterable['AtomicValue'], schema: AtomicType | str, verifySchema: bool = ...) -> DataFrame: ...
    @overload
    def createDataFrame(self, data: PandasDataFrameLike, samplingRatio: float | None = ...) -> DataFrame: ...
    @overload
    def createDataFrame(self, data: PandasDataFrameLike, schema: StructType | str, verifySchema: bool = ...) -> DataFrame: ...
    def registerDataFrameAsTable(self, df: DataFrame, tableName: str) -> None: ...
    def dropTempTable(self, tableName: str) -> None: ...
    def createExternalTable(self, tableName: str, path: str | None = None, source: str | None = None, schema: StructType | None = None, **options: str) -> DataFrame: ...
    def sql(self, sqlQuery: str) -> DataFrame: ...
    def table(self, tableName: str) -> DataFrame: ...
    def tables(self, dbName: str | None = None) -> DataFrame: ...
    def tableNames(self, dbName: str | None = None) -> list[str]: ...
    def cacheTable(self, tableName: str) -> None: ...
    def uncacheTable(self, tableName: str) -> None: ...
    def clearCache(self) -> None: ...
    @property
    def read(self) -> DataFrameReader: ...
    @property
    def readStream(self) -> DataStreamReader: ...
    @property
    def streams(self) -> StreamingQueryManager: ...

class HiveContext(SQLContext):
    def __init__(self, sparkContext: SparkContext, sparkSession: SparkSession | None = None, jhiveContext: JavaObject | None = None) -> None: ...
    def refreshTable(self, tableName: str) -> None: ...