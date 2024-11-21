from pyspark.sql._typing import ColumnOrName as ColumnOrName
from pyspark.sql.column import Column as Column
from pyspark.sql.utils import get_active_spark_context as get_active_spark_context, try_remote_protobuf_functions as try_remote_protobuf_functions

def from_protobuf(data: ColumnOrName, messageName: str, descFilePath: str | None = None, options: dict[str, str] | None = None, binaryDescriptorSet: bytes | None = None) -> Column: ...
def to_protobuf(data: ColumnOrName, messageName: str, descFilePath: str | None = None, options: dict[str, str] | None = None, binaryDescriptorSet: bytes | None = None) -> Column: ...
