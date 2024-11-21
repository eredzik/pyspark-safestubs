from pyspark.sql._typing import ColumnOrName as ColumnOrName
from pyspark.sql.column import Column as Column
from pyspark.sql.utils import get_active_spark_context as get_active_spark_context, try_remote_avro_functions as try_remote_avro_functions

def from_avro(data: ColumnOrName, jsonFormatSchema: str, options: dict[str, str] | None = None) -> Column: ...
def to_avro(data: ColumnOrName, jsonFormatSchema: str = '') -> Column: ...
