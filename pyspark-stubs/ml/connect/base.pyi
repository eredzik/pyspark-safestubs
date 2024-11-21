import pandas as pd
from abc import ABCMeta, abstractmethod
from pyspark import since as since
from pyspark.ml._typing import ParamMap as ParamMap
from pyspark.ml.common import inherit_doc as inherit_doc
from pyspark.ml.connect.util import transform_dataframe_column as transform_dataframe_column
from pyspark.ml.param import Params as Params
from pyspark.ml.param.shared import HasFeaturesCol as HasFeaturesCol, HasLabelCol as HasLabelCol, HasPredictionCol as HasPredictionCol
from pyspark.sql.dataframe import DataFrame as DataFrame
from typing import Generic, TypeVar

M = TypeVar('M', bound='Transformer')

class Estimator(Params, Generic[M], metaclass=ABCMeta):
    def fit(self, dataset: DataFrame | pd.DataFrame, params: ParamMap | None = None) -> M | list[M]: ...

class Transformer(Params, metaclass=ABCMeta):
    def transform(self, dataset: DataFrame | pd.DataFrame, params: ParamMap | None = None) -> DataFrame | pd.DataFrame: ...

class Evaluator(Params, metaclass=ABCMeta):
    def evaluate(self, dataset: DataFrame, params: ParamMap | None = None) -> float: ...
    def isLargerBetter(self) -> bool: ...

class Model(Transformer, metaclass=ABCMeta): ...
class _PredictorParams(HasLabelCol, HasFeaturesCol, HasPredictionCol): ...

class Predictor(Estimator[M], _PredictorParams, metaclass=ABCMeta):
    def setLabelCol(self, value: str) -> Predictor: ...
    def setFeaturesCol(self, value: str) -> Predictor: ...
    def setPredictionCol(self, value: str) -> Predictor: ...

class PredictionModel(Model, _PredictorParams, metaclass=ABCMeta):
    def setFeaturesCol(self, value: str) -> PredictionModel: ...
    def setPredictionCol(self, value: str) -> PredictionModel: ...
    @property
    @abstractmethod
    def numFeatures(self) -> int: ...
