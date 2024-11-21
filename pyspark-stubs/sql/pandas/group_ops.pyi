from pyspark.rdd import PythonEvalType as PythonEvalType
from pyspark.sql.column import Column as Column
from pyspark.sql.dataframe import DataFrame as DataFrame
from pyspark.sql.group import GroupedData as GroupedData
from pyspark.sql.pandas._typing import GroupedMapPandasUserDefinedFunction as GroupedMapPandasUserDefinedFunction, PandasCogroupedMapFunction as PandasCogroupedMapFunction, PandasGroupedMapFunction as PandasGroupedMapFunction, PandasGroupedMapFunctionWithState as PandasGroupedMapFunctionWithState
from pyspark.sql.streaming.state import GroupStateTimeout as GroupStateTimeout
from pyspark.sql.types import StructType as StructType

class PandasGroupedOpsMixin:
    def apply(self, udf: GroupedMapPandasUserDefinedFunction) -> DataFrame: ...
    def applyInPandas(self, func: PandasGroupedMapFunction, schema: StructType | str) -> DataFrame: ...
    def applyInPandasWithState(self, func: PandasGroupedMapFunctionWithState, outputStructType: StructType | str, stateStructType: StructType | str, outputMode: str, timeoutConf: str) -> DataFrame: ...
    def cogroup(self, other: GroupedData) -> PandasCogroupedOps: ...

class PandasCogroupedOps:
    def __init__(self, gd1: GroupedData, gd2: GroupedData) -> None: ...
    def applyInPandas(self, func: PandasCogroupedMapFunction, schema: StructType | str) -> DataFrame: ...
