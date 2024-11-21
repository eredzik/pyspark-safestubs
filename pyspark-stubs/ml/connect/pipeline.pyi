from _typeshed import Incomplete
from pyspark import keyword_only as keyword_only, since as since
from pyspark.ml._typing import ParamMap as ParamMap
from pyspark.ml.common import inherit_doc as inherit_doc
from pyspark.ml.connect.base import Estimator as Estimator, Model as Model, Transformer as Transformer
from pyspark.ml.connect.io_utils import MetaAlgorithmReadWrite as MetaAlgorithmReadWrite, ParamsReadWrite as ParamsReadWrite
from pyspark.ml.param import Param as Param, Params as Params
from pyspark.sql.dataframe import DataFrame as DataFrame

class _PipelineReadWrite(MetaAlgorithmReadWrite): ...

class Pipeline(Estimator['PipelineModel'], _PipelineReadWrite):
    stages: Param[list[Params]]
    def __init__(self, *, stages: list[Params] | None = None) -> None: ...
    def setStages(self, value: list[Params]) -> Pipeline: ...
    def getStages(self) -> list[Params]: ...
    def setParams(self, *, stages: list[Params] | None = None) -> Pipeline: ...
    def copy(self, extra: ParamMap | None = None) -> Pipeline: ...

class PipelineModel(Model, _PipelineReadWrite):
    stages: Incomplete
    def __init__(self, stages: list[Params] | None = None) -> None: ...
    def copy(self, extra: ParamMap | None = None) -> PipelineModel: ...