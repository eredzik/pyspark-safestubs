from abc import ABCMeta
from pyspark.ml._typing import BinaryClassificationEvaluatorMetricType, ClusteringEvaluatorDistanceMeasureType, ClusteringEvaluatorMetricType, MulticlassClassificationEvaluatorMetricType, MultilabelClassificationEvaluatorMetricType, ParamMap, RankingEvaluatorMetricType, RegressionEvaluatorMetricType
from pyspark.ml.param import Param, Params
from pyspark.ml.param.shared import HasFeaturesCol, HasLabelCol, HasPredictionCol, HasProbabilityCol, HasRawPredictionCol, HasWeightCol
from pyspark.ml.util import JavaMLReadable, JavaMLWritable
from pyspark.ml.wrapper import JavaParams
from pyspark.sql.dataframe import DataFrame

__all__ = ['Evaluator', 'BinaryClassificationEvaluator', 'RegressionEvaluator', 'MulticlassClassificationEvaluator', 'MultilabelClassificationEvaluator', 'ClusteringEvaluator', 'RankingEvaluator']

class Evaluator(Params, metaclass=ABCMeta):
    def evaluate(self, dataset: DataFrame, params: ParamMap | None = None) -> float: ...
    def isLargerBetter(self) -> bool: ...

class JavaEvaluator(JavaParams, Evaluator, metaclass=ABCMeta):
    def isLargerBetter(self) -> bool: ...

class BinaryClassificationEvaluator(JavaEvaluator, HasLabelCol, HasRawPredictionCol, HasWeightCol, JavaMLReadable['BinaryClassificationEvaluator'], JavaMLWritable):
    metricName: Param['BinaryClassificationEvaluatorMetricType']
    numBins: Param[int]
    def __init__(self, *, rawPredictionCol: str = 'rawPrediction', labelCol: str = 'label', metricName: BinaryClassificationEvaluatorMetricType = 'areaUnderROC', weightCol: str | None = None, numBins: int = 1000) -> None: ...
    def setMetricName(self, value: BinaryClassificationEvaluatorMetricType) -> BinaryClassificationEvaluator: ...
    def getMetricName(self) -> str: ...
    def setNumBins(self, value: int) -> BinaryClassificationEvaluator: ...
    def getNumBins(self) -> int: ...
    def setLabelCol(self, value: str) -> BinaryClassificationEvaluator: ...
    def setRawPredictionCol(self, value: str) -> BinaryClassificationEvaluator: ...
    def setWeightCol(self, value: str) -> BinaryClassificationEvaluator: ...
    def setParams(self, *, rawPredictionCol: str = 'rawPrediction', labelCol: str = 'label', metricName: BinaryClassificationEvaluatorMetricType = 'areaUnderROC', weightCol: str | None = None, numBins: int = 1000) -> BinaryClassificationEvaluator: ...

class RegressionEvaluator(JavaEvaluator, HasLabelCol, HasPredictionCol, HasWeightCol, JavaMLReadable['RegressionEvaluator'], JavaMLWritable):
    metricName: Param['RegressionEvaluatorMetricType']
    throughOrigin: Param[bool]
    def __init__(self, *, predictionCol: str = 'prediction', labelCol: str = 'label', metricName: RegressionEvaluatorMetricType = 'rmse', weightCol: str | None = None, throughOrigin: bool = False) -> None: ...
    def setMetricName(self, value: RegressionEvaluatorMetricType) -> RegressionEvaluator: ...
    def getMetricName(self) -> RegressionEvaluatorMetricType: ...
    def setThroughOrigin(self, value: bool) -> RegressionEvaluator: ...
    def getThroughOrigin(self) -> bool: ...
    def setLabelCol(self, value: str) -> RegressionEvaluator: ...
    def setPredictionCol(self, value: str) -> RegressionEvaluator: ...
    def setWeightCol(self, value: str) -> RegressionEvaluator: ...
    def setParams(self, *, predictionCol: str = 'prediction', labelCol: str = 'label', metricName: RegressionEvaluatorMetricType = 'rmse', weightCol: str | None = None, throughOrigin: bool = False) -> RegressionEvaluator: ...

class MulticlassClassificationEvaluator(JavaEvaluator, HasLabelCol, HasPredictionCol, HasWeightCol, HasProbabilityCol, JavaMLReadable['MulticlassClassificationEvaluator'], JavaMLWritable):
    metricName: Param['MulticlassClassificationEvaluatorMetricType']
    metricLabel: Param[float]
    beta: Param[float]
    eps: Param[float]
    def __init__(self, *, predictionCol: str = 'prediction', labelCol: str = 'label', metricName: MulticlassClassificationEvaluatorMetricType = 'f1', weightCol: str | None = None, metricLabel: float = 0.0, beta: float = 1.0, probabilityCol: str = 'probability', eps: float = 1e-15) -> None: ...
    def setMetricName(self, value: MulticlassClassificationEvaluatorMetricType) -> MulticlassClassificationEvaluator: ...
    def getMetricName(self) -> MulticlassClassificationEvaluatorMetricType: ...
    def setMetricLabel(self, value: float) -> MulticlassClassificationEvaluator: ...
    def getMetricLabel(self) -> float: ...
    def setBeta(self, value: float) -> MulticlassClassificationEvaluator: ...
    def getBeta(self) -> float: ...
    def setEps(self, value: float) -> MulticlassClassificationEvaluator: ...
    def getEps(self) -> float: ...
    def setLabelCol(self, value: str) -> MulticlassClassificationEvaluator: ...
    def setPredictionCol(self, value: str) -> MulticlassClassificationEvaluator: ...
    def setProbabilityCol(self, value: str) -> MulticlassClassificationEvaluator: ...
    def setWeightCol(self, value: str) -> MulticlassClassificationEvaluator: ...
    def setParams(self, *, predictionCol: str = 'prediction', labelCol: str = 'label', metricName: MulticlassClassificationEvaluatorMetricType = 'f1', weightCol: str | None = None, metricLabel: float = 0.0, beta: float = 1.0, probabilityCol: str = 'probability', eps: float = 1e-15) -> MulticlassClassificationEvaluator: ...

class MultilabelClassificationEvaluator(JavaEvaluator, HasLabelCol, HasPredictionCol, JavaMLReadable['MultilabelClassificationEvaluator'], JavaMLWritable):
    metricName: Param['MultilabelClassificationEvaluatorMetricType']
    metricLabel: Param[float]
    def __init__(self, *, predictionCol: str = 'prediction', labelCol: str = 'label', metricName: MultilabelClassificationEvaluatorMetricType = 'f1Measure', metricLabel: float = 0.0) -> None: ...
    def setMetricName(self, value: MultilabelClassificationEvaluatorMetricType) -> MultilabelClassificationEvaluator: ...
    def getMetricName(self) -> MultilabelClassificationEvaluatorMetricType: ...
    def setMetricLabel(self, value: float) -> MultilabelClassificationEvaluator: ...
    def getMetricLabel(self) -> float: ...
    def setLabelCol(self, value: str) -> MultilabelClassificationEvaluator: ...
    def setPredictionCol(self, value: str) -> MultilabelClassificationEvaluator: ...
    def setParams(self, *, predictionCol: str = 'prediction', labelCol: str = 'label', metricName: MultilabelClassificationEvaluatorMetricType = 'f1Measure', metricLabel: float = 0.0) -> MultilabelClassificationEvaluator: ...

class ClusteringEvaluator(JavaEvaluator, HasPredictionCol, HasFeaturesCol, HasWeightCol, JavaMLReadable['ClusteringEvaluator'], JavaMLWritable):
    metricName: Param['ClusteringEvaluatorMetricType']
    distanceMeasure: Param['ClusteringEvaluatorDistanceMeasureType']
    def __init__(self, *, predictionCol: str = 'prediction', featuresCol: str = 'features', metricName: ClusteringEvaluatorMetricType = 'silhouette', distanceMeasure: str = 'squaredEuclidean', weightCol: str | None = None) -> None: ...
    def setParams(self, *, predictionCol: str = 'prediction', featuresCol: str = 'features', metricName: ClusteringEvaluatorMetricType = 'silhouette', distanceMeasure: str = 'squaredEuclidean', weightCol: str | None = None) -> ClusteringEvaluator: ...
    def setMetricName(self, value: ClusteringEvaluatorMetricType) -> ClusteringEvaluator: ...
    def getMetricName(self) -> ClusteringEvaluatorMetricType: ...
    def setDistanceMeasure(self, value: ClusteringEvaluatorDistanceMeasureType) -> ClusteringEvaluator: ...
    def getDistanceMeasure(self) -> ClusteringEvaluatorDistanceMeasureType: ...
    def setFeaturesCol(self, value: str) -> ClusteringEvaluator: ...
    def setPredictionCol(self, value: str) -> ClusteringEvaluator: ...
    def setWeightCol(self, value: str) -> ClusteringEvaluator: ...

class RankingEvaluator(JavaEvaluator, HasLabelCol, HasPredictionCol, JavaMLReadable['RankingEvaluator'], JavaMLWritable):
    metricName: Param['RankingEvaluatorMetricType']
    k: Param[int]
    def __init__(self, *, predictionCol: str = 'prediction', labelCol: str = 'label', metricName: RankingEvaluatorMetricType = 'meanAveragePrecision', k: int = 10) -> None: ...
    def setMetricName(self, value: RankingEvaluatorMetricType) -> RankingEvaluator: ...
    def getMetricName(self) -> RankingEvaluatorMetricType: ...
    def setK(self, value: int) -> RankingEvaluator: ...
    def getK(self) -> int: ...
    def setLabelCol(self, value: str) -> RankingEvaluator: ...
    def setPredictionCol(self, value: str) -> RankingEvaluator: ...
    def setParams(self, *, predictionCol: str = 'prediction', labelCol: str = 'label', metricName: RankingEvaluatorMetricType = 'meanAveragePrecision', k: int = 10) -> RankingEvaluator: ...
