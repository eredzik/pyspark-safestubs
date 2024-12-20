import pyarrow as pa
from pyspark.errors import PySparkTypeError as PySparkTypeError, UnsupportedOperationException as UnsupportedOperationException
from pyspark.sql.types import ArrayType as ArrayType, BinaryType as BinaryType, BooleanType as BooleanType, ByteType as ByteType, DataType as DataType, DateType as DateType, DayTimeIntervalType as DayTimeIntervalType, DecimalType as DecimalType, DoubleType as DoubleType, FloatType as FloatType, IntegerType as IntegerType, IntegralType as IntegralType, LongType as LongType, MapType as MapType, NullType as NullType, Row as Row, ShortType as ShortType, StringType as StringType, StructField as StructField, StructType as StructType, TimestampNTZType as TimestampNTZType, TimestampType as TimestampType, UserDefinedType as UserDefinedType, cast as cast

def to_arrow_type(dt: DataType) -> pa.DataType: ...
def to_arrow_schema(schema: StructType) -> pa.Schema: ...
def from_arrow_type(at: pa.DataType, prefer_timestamp_ntz: bool = False) -> DataType: ...
def from_arrow_schema(arrow_schema: pa.Schema, prefer_timestamp_ntz: bool = False) -> StructType: ...
