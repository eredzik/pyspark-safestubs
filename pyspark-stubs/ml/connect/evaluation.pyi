from pyspark import keyword_only as keyword_only
from pyspark.ml.connect.base import Evaluator as Evaluator
from pyspark.ml.connect.io_utils import ParamsReadWrite as ParamsReadWrite
from pyspark.ml.connect.util import aggregate_dataframe as aggregate_dataframe
from pyspark.ml.param import Param as Param, Params as Params, TypeConverters as TypeConverters
from pyspark.ml.param.shared import HasLabelCol as HasLabelCol, HasPredictionCol as HasPredictionCol, HasProbabilityCol as HasProbabilityCol
from pyspark.sql import DataFrame as DataFrame

class _TorchMetricEvaluator(Evaluator):
    metricName: Param[str]
    def getMetricName(self) -> str: ...

class RegressionEvaluator(_TorchMetricEvaluator, HasLabelCol, HasPredictionCol, ParamsReadWrite):
    def __init__(self, *, metricName: str = 'rmse', labelCol: str = 'label', predictionCol: str = 'prediction') -> None: ...
    def isLargerBetter(self) -> bool: ...

class BinaryClassificationEvaluator(_TorchMetricEvaluator, HasLabelCol, HasProbabilityCol, ParamsReadWrite):
    def __init__(self, *, metricName: str = 'areaUnderROC', labelCol: str = 'label', probabilityCol: str = 'probability') -> None: ...
    def isLargerBetter(self) -> bool: ...

class MulticlassClassificationEvaluator(_TorchMetricEvaluator, HasLabelCol, HasPredictionCol, ParamsReadWrite):
    def __init__(self, metricName: str = 'accuracy', labelCol: str = 'label', predictionCol: str = 'prediction') -> None: ...
    def isLargerBetter(self) -> bool: ...
