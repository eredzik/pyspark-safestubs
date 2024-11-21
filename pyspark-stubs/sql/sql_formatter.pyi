import string
from pyspark.sql import DataFrame as DataFrame, SparkSession as SparkSession
from pyspark.sql.functions import lit as lit
from typing import Any, Mapping, Sequence

class SQLStringFormatter(string.Formatter):
    def __init__(self, session: SparkSession) -> None: ...
    def get_field(self, field_name: str, args: Sequence[Any], kwargs: Mapping[str, Any]) -> Any: ...
    def clear(self) -> None: ...