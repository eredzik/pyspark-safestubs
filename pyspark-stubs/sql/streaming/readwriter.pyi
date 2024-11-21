from pyspark.sql._typing import OptionalPrimitiveType, SupportsProcess
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.readwriter import OptionUtils
from pyspark.sql.session import SparkSession
from pyspark.sql.streaming.query import StreamingQuery
from pyspark.sql.types import Row, StructType
from typing import Callable, overload

__all__ = ['DataStreamReader', 'DataStreamWriter']

class DataStreamReader(OptionUtils):
    def __init__(self, spark: SparkSession) -> None: ...
    def format(self, source: str) -> DataStreamReader: ...
    def schema(self, schema: StructType | str) -> DataStreamReader: ...
    def option(self, key: str, value: OptionalPrimitiveType) -> DataStreamReader: ...
    def options(self, **options: OptionalPrimitiveType) -> DataStreamReader: ...
    def load(self, path: str | None = None, format: str | None = None, schema: StructType | str | None = None, **options: OptionalPrimitiveType) -> DataFrame: ...
    def json(self, path: str, schema: StructType | str | None = None, primitivesAsString: bool | str | None = None, prefersDecimal: bool | str | None = None, allowComments: bool | str | None = None, allowUnquotedFieldNames: bool | str | None = None, allowSingleQuotes: bool | str | None = None, allowNumericLeadingZero: bool | str | None = None, allowBackslashEscapingAnyCharacter: bool | str | None = None, mode: str | None = None, columnNameOfCorruptRecord: str | None = None, dateFormat: str | None = None, timestampFormat: str | None = None, multiLine: bool | str | None = None, allowUnquotedControlChars: bool | str | None = None, lineSep: str | None = None, locale: str | None = None, dropFieldIfAllNull: bool | str | None = None, encoding: str | None = None, pathGlobFilter: bool | str | None = None, recursiveFileLookup: bool | str | None = None, allowNonNumericNumbers: bool | str | None = None) -> DataFrame: ...
    def orc(self, path: str, mergeSchema: bool | None = None, pathGlobFilter: bool | str | None = None, recursiveFileLookup: bool | str | None = None) -> DataFrame: ...
    def parquet(self, path: str, mergeSchema: bool | None = None, pathGlobFilter: bool | str | None = None, recursiveFileLookup: bool | str | None = None, datetimeRebaseMode: bool | str | None = None, int96RebaseMode: bool | str | None = None) -> DataFrame: ...
    def text(self, path: str, wholetext: bool = False, lineSep: str | None = None, pathGlobFilter: bool | str | None = None, recursiveFileLookup: bool | str | None = None) -> DataFrame: ...
    def csv(self, path: str, schema: StructType | str | None = None, sep: str | None = None, encoding: str | None = None, quote: str | None = None, escape: str | None = None, comment: str | None = None, header: bool | str | None = None, inferSchema: bool | str | None = None, ignoreLeadingWhiteSpace: bool | str | None = None, ignoreTrailingWhiteSpace: bool | str | None = None, nullValue: str | None = None, nanValue: str | None = None, positiveInf: str | None = None, negativeInf: str | None = None, dateFormat: str | None = None, timestampFormat: str | None = None, maxColumns: int | str | None = None, maxCharsPerColumn: int | str | None = None, maxMalformedLogPerPartition: int | str | None = None, mode: str | None = None, columnNameOfCorruptRecord: str | None = None, multiLine: bool | str | None = None, charToEscapeQuoteEscaping: bool | str | None = None, enforceSchema: bool | str | None = None, emptyValue: str | None = None, locale: str | None = None, lineSep: str | None = None, pathGlobFilter: bool | str | None = None, recursiveFileLookup: bool | str | None = None, unescapedQuoteHandling: str | None = None) -> DataFrame: ...
    def table(self, tableName: str) -> DataFrame: ...

class DataStreamWriter:
    def __init__(self, df: DataFrame) -> None: ...
    def outputMode(self, outputMode: str) -> DataStreamWriter: ...
    def format(self, source: str) -> DataStreamWriter: ...
    def option(self, key: str, value: OptionalPrimitiveType) -> DataStreamWriter: ...
    def options(self, **options: OptionalPrimitiveType) -> DataStreamWriter: ...
    @overload
    def partitionBy(self, *cols: str) -> DataStreamWriter: ...
    @overload
    def partitionBy(self, /, __cols: list[str]) -> DataStreamWriter: ...
    def queryName(self, queryName: str) -> DataStreamWriter: ...
    @overload
    def trigger(self, *, processingTime: str) -> DataStreamWriter: ...
    @overload
    def trigger(self, *, once: bool) -> DataStreamWriter: ...
    @overload
    def trigger(self, *, continuous: str) -> DataStreamWriter: ...
    @overload
    def trigger(self, *, availableNow: bool) -> DataStreamWriter: ...
    @overload
    def foreach(self, f: Callable[[Row], None]) -> DataStreamWriter: ...
    @overload
    def foreach(self, f: SupportsProcess) -> DataStreamWriter: ...
    def foreachBatch(self, func: Callable[[DataFrame, int], None]) -> DataStreamWriter: ...
    def start(self, path: str | None = None, format: str | None = None, outputMode: str | None = None, partitionBy: str | list[str] | None = None, queryName: str | None = None, **options: OptionalPrimitiveType) -> StreamingQuery: ...
    def toTable(self, tableName: str, format: str | None = None, outputMode: str | None = None, partitionBy: str | list[str] | None = None, queryName: str | None = None, **options: OptionalPrimitiveType) -> StreamingQuery: ...