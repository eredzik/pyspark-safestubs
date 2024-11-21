from _typeshed import Incomplete
from abc import ABCMeta
from pyspark.ml import PredictionModel, Predictor
from pyspark.ml.base import Transformer, _PredictorParams
from pyspark.ml.linalg import Matrix, Vector
from pyspark.ml.param.shared import HasAggregationDepth, HasElasticNetParam, HasFeaturesCol, HasFitIntercept, HasLabelCol, HasLoss, HasMaxBlockSizeInMB, HasMaxIter, HasPredictionCol, HasRegParam, HasSeed, HasSolver, HasStandardization, HasStepSize, HasTol, HasVarianceCol, HasWeightCol, Param
from pyspark.ml.tree import _DecisionTreeModel, _DecisionTreeParams, _GBTParams, _RandomForestParams, _TreeEnsembleModel, _TreeRegressorParams
from pyspark.ml.util import GeneralJavaMLWritable, HasTrainingSummary, JavaMLReadable, JavaMLWritable
from pyspark.ml.wrapper import JavaEstimator, JavaModel, JavaPredictionModel, JavaPredictor, JavaTransformer, JavaWrapper
from pyspark.sql import DataFrame
from typing import Any, Generic, TypeVar

__all__ = ['AFTSurvivalRegression', 'AFTSurvivalRegressionModel', 'DecisionTreeRegressor', 'DecisionTreeRegressionModel', 'GBTRegressor', 'GBTRegressionModel', 'GeneralizedLinearRegression', 'GeneralizedLinearRegressionModel', 'GeneralizedLinearRegressionSummary', 'GeneralizedLinearRegressionTrainingSummary', 'IsotonicRegression', 'IsotonicRegressionModel', 'LinearRegression', 'LinearRegressionModel', 'LinearRegressionSummary', 'LinearRegressionTrainingSummary', 'RandomForestRegressor', 'RandomForestRegressionModel', 'FMRegressor', 'FMRegressionModel']

T = TypeVar('T')
M = TypeVar('M', bound=Transformer)
JM = TypeVar('JM', bound=JavaTransformer)

class Regressor(Predictor[M], _PredictorParams, Generic[M], metaclass=ABCMeta): ...
class RegressionModel(PredictionModel[T], _PredictorParams, metaclass=ABCMeta): ...
class _JavaRegressor(Regressor, JavaPredictor[JM], Generic[JM], metaclass=ABCMeta): ...
class _JavaRegressionModel(RegressionModel, JavaPredictionModel[T], metaclass=ABCMeta): ...

class _LinearRegressionParams(_PredictorParams, HasRegParam, HasElasticNetParam, HasMaxIter, HasTol, HasFitIntercept, HasStandardization, HasWeightCol, HasSolver, HasAggregationDepth, HasLoss, HasMaxBlockSizeInMB):
    solver: Param[str]
    loss: Param[str]
    epsilon: Param[float]
    def __init__(self, *args: Any) -> None: ...
    def getEpsilon(self) -> float: ...

class LinearRegression(_JavaRegressor['LinearRegressionModel'], _LinearRegressionParams, JavaMLWritable, JavaMLReadable['LinearRegression']):
    def __init__(self, *, featuresCol: str = 'features', labelCol: str = 'label', predictionCol: str = 'prediction', maxIter: int = 100, regParam: float = 0.0, elasticNetParam: float = 0.0, tol: float = 1e-06, fitIntercept: bool = True, standardization: bool = True, solver: str = 'auto', weightCol: str | None = None, aggregationDepth: int = 2, loss: str = 'squaredError', epsilon: float = 1.35, maxBlockSizeInMB: float = 0.0) -> None: ...
    def setParams(self, *, featuresCol: str = 'features', labelCol: str = 'label', predictionCol: str = 'prediction', maxIter: int = 100, regParam: float = 0.0, elasticNetParam: float = 0.0, tol: float = 1e-06, fitIntercept: bool = True, standardization: bool = True, solver: str = 'auto', weightCol: str | None = None, aggregationDepth: int = 2, loss: str = 'squaredError', epsilon: float = 1.35, maxBlockSizeInMB: float = 0.0) -> LinearRegression: ...
    def setEpsilon(self, value: float) -> LinearRegression: ...
    def setMaxIter(self, value: int) -> LinearRegression: ...
    def setRegParam(self, value: float) -> LinearRegression: ...
    def setTol(self, value: float) -> LinearRegression: ...
    def setElasticNetParam(self, value: float) -> LinearRegression: ...
    def setFitIntercept(self, value: bool) -> LinearRegression: ...
    def setStandardization(self, value: bool) -> LinearRegression: ...
    def setWeightCol(self, value: str) -> LinearRegression: ...
    def setSolver(self, value: str) -> LinearRegression: ...
    def setAggregationDepth(self, value: int) -> LinearRegression: ...
    def setLoss(self, value: str) -> LinearRegression: ...
    def setMaxBlockSizeInMB(self, value: float) -> LinearRegression: ...

class LinearRegressionModel(_JavaRegressionModel, _LinearRegressionParams, GeneralJavaMLWritable, JavaMLReadable['LinearRegressionModel'], HasTrainingSummary['LinearRegressionSummary']):
    @property
    def coefficients(self) -> Vector: ...
    @property
    def intercept(self) -> float: ...
    @property
    def scale(self) -> float: ...
    @property
    def summary(self) -> LinearRegressionTrainingSummary: ...
    def evaluate(self, dataset: DataFrame) -> LinearRegressionSummary: ...

class LinearRegressionSummary(JavaWrapper):
    @property
    def predictions(self) -> DataFrame: ...
    @property
    def predictionCol(self) -> str: ...
    @property
    def labelCol(self) -> str: ...
    @property
    def featuresCol(self) -> str: ...
    @property
    def explainedVariance(self) -> float: ...
    @property
    def meanAbsoluteError(self) -> float: ...
    @property
    def meanSquaredError(self) -> float: ...
    @property
    def rootMeanSquaredError(self) -> float: ...
    @property
    def r2(self) -> float: ...
    @property
    def r2adj(self) -> float: ...
    @property
    def residuals(self) -> DataFrame: ...
    @property
    def numInstances(self) -> int: ...
    @property
    def degreesOfFreedom(self) -> int: ...
    @property
    def devianceResiduals(self) -> list[float]: ...
    @property
    def coefficientStandardErrors(self) -> list[float]: ...
    @property
    def tValues(self) -> list[float]: ...
    @property
    def pValues(self) -> list[float]: ...

class LinearRegressionTrainingSummary(LinearRegressionSummary):
    @property
    def objectiveHistory(self) -> list[float]: ...
    @property
    def totalIterations(self) -> int: ...

class _IsotonicRegressionParams(HasFeaturesCol, HasLabelCol, HasPredictionCol, HasWeightCol):
    isotonic: Param[bool]
    featureIndex: Param[int]
    def __init__(self, *args: Any) -> None: ...
    def getIsotonic(self) -> bool: ...
    def getFeatureIndex(self) -> int: ...

class IsotonicRegression(JavaEstimator, _IsotonicRegressionParams, HasWeightCol, JavaMLWritable, JavaMLReadable):
    def __init__(self, *, featuresCol: str = 'features', labelCol: str = 'label', predictionCol: str = 'prediction', weightCol: str | None = None, isotonic: bool = True, featureIndex: int = 0) -> None: ...
    def setParams(self, *, featuresCol: str = 'features', labelCol: str = 'label', predictionCol: str = 'prediction', weightCol: str | None = None, isotonic: bool = True, featureIndex: int = 0) -> IsotonicRegression: ...
    def setIsotonic(self, value: bool) -> IsotonicRegression: ...
    def setFeatureIndex(self, value: int) -> IsotonicRegression: ...
    def setFeaturesCol(self, value: str) -> IsotonicRegression: ...
    def setPredictionCol(self, value: str) -> IsotonicRegression: ...
    def setLabelCol(self, value: str) -> IsotonicRegression: ...
    def setWeightCol(self, value: str) -> IsotonicRegression: ...

class IsotonicRegressionModel(JavaModel, _IsotonicRegressionParams, JavaMLWritable, JavaMLReadable['IsotonicRegressionModel']):
    def setFeaturesCol(self, value: str) -> IsotonicRegressionModel: ...
    def setPredictionCol(self, value: str) -> IsotonicRegressionModel: ...
    def setFeatureIndex(self, value: int) -> IsotonicRegressionModel: ...
    @property
    def boundaries(self) -> Vector: ...
    @property
    def predictions(self) -> Vector: ...
    @property
    def numFeatures(self) -> int: ...
    def predict(self, value: float) -> float: ...

class _DecisionTreeRegressorParams(_DecisionTreeParams, _TreeRegressorParams, HasVarianceCol):
    def __init__(self, *args: Any) -> None: ...

class DecisionTreeRegressor(_JavaRegressor['DecisionTreeRegressionModel'], _DecisionTreeRegressorParams, JavaMLWritable, JavaMLReadable['DecisionTreeRegressor']):
    def __init__(self, *, featuresCol: str = 'features', labelCol: str = 'label', predictionCol: str = 'prediction', maxDepth: int = 5, maxBins: int = 32, minInstancesPerNode: int = 1, minInfoGain: float = 0.0, maxMemoryInMB: int = 256, cacheNodeIds: bool = False, checkpointInterval: int = 10, impurity: str = 'variance', seed: int | None = None, varianceCol: str | None = None, weightCol: str | None = None, leafCol: str = '', minWeightFractionPerNode: float = 0.0) -> None: ...
    def setParams(self, *, featuresCol: str = 'features', labelCol: str = 'label', predictionCol: str = 'prediction', maxDepth: int = 5, maxBins: int = 32, minInstancesPerNode: int = 1, minInfoGain: float = 0.0, maxMemoryInMB: int = 256, cacheNodeIds: bool = False, checkpointInterval: int = 10, impurity: str = 'variance', seed: int | None = None, varianceCol: str | None = None, weightCol: str | None = None, leafCol: str = '', minWeightFractionPerNode: float = 0.0) -> DecisionTreeRegressor: ...
    def setMaxDepth(self, value: int) -> DecisionTreeRegressor: ...
    def setMaxBins(self, value: int) -> DecisionTreeRegressor: ...
    def setMinInstancesPerNode(self, value: int) -> DecisionTreeRegressor: ...
    def setMinWeightFractionPerNode(self, value: float) -> DecisionTreeRegressor: ...
    def setMinInfoGain(self, value: float) -> DecisionTreeRegressor: ...
    def setMaxMemoryInMB(self, value: int) -> DecisionTreeRegressor: ...
    def setCacheNodeIds(self, value: bool) -> DecisionTreeRegressor: ...
    def setImpurity(self, value: str) -> DecisionTreeRegressor: ...
    def setCheckpointInterval(self, value: int) -> DecisionTreeRegressor: ...
    def setSeed(self, value: int) -> DecisionTreeRegressor: ...
    def setWeightCol(self, value: str) -> DecisionTreeRegressor: ...
    def setVarianceCol(self, value: str) -> DecisionTreeRegressor: ...

class DecisionTreeRegressionModel(_JavaRegressionModel, _DecisionTreeModel, _DecisionTreeRegressorParams, JavaMLWritable, JavaMLReadable['DecisionTreeRegressionModel']):
    def setVarianceCol(self, value: str) -> DecisionTreeRegressionModel: ...
    @property
    def featureImportances(self) -> Vector: ...

class _RandomForestRegressorParams(_RandomForestParams, _TreeRegressorParams):
    def __init__(self, *args: Any) -> None: ...

class RandomForestRegressor(_JavaRegressor['RandomForestRegressionModel'], _RandomForestRegressorParams, JavaMLWritable, JavaMLReadable['RandomForestRegressor']):
    def __init__(self, *, featuresCol: str = 'features', labelCol: str = 'label', predictionCol: str = 'prediction', maxDepth: int = 5, maxBins: int = 32, minInstancesPerNode: int = 1, minInfoGain: float = 0.0, maxMemoryInMB: int = 256, cacheNodeIds: bool = False, checkpointInterval: int = 10, impurity: str = 'variance', subsamplingRate: float = 1.0, seed: int | None = None, numTrees: int = 20, featureSubsetStrategy: str = 'auto', leafCol: str = '', minWeightFractionPerNode: float = 0.0, weightCol: str | None = None, bootstrap: bool | None = True) -> None: ...
    def setParams(self, *, featuresCol: str = 'features', labelCol: str = 'label', predictionCol: str = 'prediction', maxDepth: int = 5, maxBins: int = 32, minInstancesPerNode: int = 1, minInfoGain: float = 0.0, maxMemoryInMB: int = 256, cacheNodeIds: bool = False, checkpointInterval: int = 10, impurity: str = 'variance', subsamplingRate: float = 1.0, seed: int | None = None, numTrees: int = 20, featureSubsetStrategy: str = 'auto', leafCol: str = '', minWeightFractionPerNode: float = 0.0, weightCol: str | None = None, bootstrap: bool | None = True) -> RandomForestRegressor: ...
    def setMaxDepth(self, value: int) -> RandomForestRegressor: ...
    def setMaxBins(self, value: int) -> RandomForestRegressor: ...
    def setMinInstancesPerNode(self, value: int) -> RandomForestRegressor: ...
    def setMinInfoGain(self, value: float) -> RandomForestRegressor: ...
    def setMaxMemoryInMB(self, value: int) -> RandomForestRegressor: ...
    def setCacheNodeIds(self, value: bool) -> RandomForestRegressor: ...
    def setImpurity(self, value: str) -> RandomForestRegressor: ...
    def setNumTrees(self, value: int) -> RandomForestRegressor: ...
    def setBootstrap(self, value: bool) -> RandomForestRegressor: ...
    def setSubsamplingRate(self, value: float) -> RandomForestRegressor: ...
    def setFeatureSubsetStrategy(self, value: str) -> RandomForestRegressor: ...
    def setCheckpointInterval(self, value: int) -> RandomForestRegressor: ...
    def setSeed(self, value: int) -> RandomForestRegressor: ...
    def setWeightCol(self, value: str) -> RandomForestRegressor: ...
    def setMinWeightFractionPerNode(self, value: float) -> RandomForestRegressor: ...

class RandomForestRegressionModel(_JavaRegressionModel[Vector], _TreeEnsembleModel, _RandomForestRegressorParams, JavaMLWritable, JavaMLReadable['RandomForestRegressionModel']):
    @property
    def trees(self) -> list[DecisionTreeRegressionModel]: ...
    @property
    def featureImportances(self) -> Vector: ...

class _GBTRegressorParams(_GBTParams, _TreeRegressorParams):
    supportedLossTypes: list[str]
    lossType: Param[str]
    def __init__(self, *args: Any) -> None: ...
    def getLossType(self) -> str: ...

class GBTRegressor(_JavaRegressor['GBTRegressionModel'], _GBTRegressorParams, JavaMLWritable, JavaMLReadable['GBTRegressor']):
    def __init__(self, *, featuresCol: str = 'features', labelCol: str = 'label', predictionCol: str = 'prediction', maxDepth: int = 5, maxBins: int = 32, minInstancesPerNode: int = 1, minInfoGain: float = 0.0, maxMemoryInMB: int = 256, cacheNodeIds: bool = False, subsamplingRate: float = 1.0, checkpointInterval: int = 10, lossType: str = 'squared', maxIter: int = 20, stepSize: float = 0.1, seed: int | None = None, impurity: str = 'variance', featureSubsetStrategy: str = 'all', validationTol: float = 0.1, validationIndicatorCol: str | None = None, leafCol: str = '', minWeightFractionPerNode: float = 0.0, weightCol: str | None = None) -> None: ...
    def setParams(self, *, featuresCol: str = 'features', labelCol: str = 'label', predictionCol: str = 'prediction', maxDepth: int = 5, maxBins: int = 32, minInstancesPerNode: int = 1, minInfoGain: float = 0.0, maxMemoryInMB: int = 256, cacheNodeIds: bool = False, subsamplingRate: float = 1.0, checkpointInterval: int = 10, lossType: str = 'squared', maxIter: int = 20, stepSize: float = 0.1, seed: int | None = None, impurity: str = 'variance', featureSubsetStrategy: str = 'all', validationTol: float = 0.1, validationIndicatorCol: str | None = None, leafCol: str = '', minWeightFractionPerNode: float = 0.0, weightCol: str | None = None) -> GBTRegressor: ...
    def setMaxDepth(self, value: int) -> GBTRegressor: ...
    def setMaxBins(self, value: int) -> GBTRegressor: ...
    def setMinInstancesPerNode(self, value: int) -> GBTRegressor: ...
    def setMinInfoGain(self, value: float) -> GBTRegressor: ...
    def setMaxMemoryInMB(self, value: int) -> GBTRegressor: ...
    def setCacheNodeIds(self, value: bool) -> GBTRegressor: ...
    def setImpurity(self, value: str) -> GBTRegressor: ...
    def setLossType(self, value: str) -> GBTRegressor: ...
    def setSubsamplingRate(self, value: float) -> GBTRegressor: ...
    def setFeatureSubsetStrategy(self, value: str) -> GBTRegressor: ...
    def setValidationIndicatorCol(self, value: str) -> GBTRegressor: ...
    def setMaxIter(self, value: int) -> GBTRegressor: ...
    def setCheckpointInterval(self, value: int) -> GBTRegressor: ...
    def setSeed(self, value: int) -> GBTRegressor: ...
    def setStepSize(self, value: float) -> GBTRegressor: ...
    def setWeightCol(self, value: str) -> GBTRegressor: ...
    def setMinWeightFractionPerNode(self, value: float) -> GBTRegressor: ...

class GBTRegressionModel(_JavaRegressionModel[Vector], _TreeEnsembleModel, _GBTRegressorParams, JavaMLWritable, JavaMLReadable['GBTRegressionModel']):
    @property
    def featureImportances(self) -> Vector: ...
    @property
    def trees(self) -> list[DecisionTreeRegressionModel]: ...
    def evaluateEachIteration(self, dataset: DataFrame, loss: str) -> list[float]: ...

class _AFTSurvivalRegressionParams(_PredictorParams, HasMaxIter, HasTol, HasFitIntercept, HasAggregationDepth, HasMaxBlockSizeInMB):
    censorCol: Param[str]
    quantileProbabilities: Param[list[float]]
    quantilesCol: Param[str]
    def __init__(self, *args: Any) -> None: ...
    def getCensorCol(self) -> str: ...
    def getQuantileProbabilities(self) -> list[float]: ...
    def getQuantilesCol(self) -> str: ...

class AFTSurvivalRegression(_JavaRegressor['AFTSurvivalRegressionModel'], _AFTSurvivalRegressionParams, JavaMLWritable, JavaMLReadable['AFTSurvivalRegression']):
    def __init__(self, *, featuresCol: str = 'features', labelCol: str = 'label', predictionCol: str = 'prediction', fitIntercept: bool = True, maxIter: int = 100, tol: float = 1e-06, censorCol: str = 'censor', quantileProbabilities: list[float] = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99], quantilesCol: str | None = None, aggregationDepth: int = 2, maxBlockSizeInMB: float = 0.0) -> None: ...
    def setParams(self, *, featuresCol: str = 'features', labelCol: str = 'label', predictionCol: str = 'prediction', fitIntercept: bool = True, maxIter: int = 100, tol: float = 1e-06, censorCol: str = 'censor', quantileProbabilities: list[float] = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99], quantilesCol: str | None = None, aggregationDepth: int = 2, maxBlockSizeInMB: float = 0.0) -> AFTSurvivalRegression: ...
    def setCensorCol(self, value: str) -> AFTSurvivalRegression: ...
    def setQuantileProbabilities(self, value: list[float]) -> AFTSurvivalRegression: ...
    def setQuantilesCol(self, value: str) -> AFTSurvivalRegression: ...
    def setMaxIter(self, value: int) -> AFTSurvivalRegression: ...
    def setTol(self, value: float) -> AFTSurvivalRegression: ...
    def setFitIntercept(self, value: bool) -> AFTSurvivalRegression: ...
    def setAggregationDepth(self, value: int) -> AFTSurvivalRegression: ...
    def setMaxBlockSizeInMB(self, value: int) -> AFTSurvivalRegression: ...

class AFTSurvivalRegressionModel(_JavaRegressionModel[Vector], _AFTSurvivalRegressionParams, JavaMLWritable, JavaMLReadable['AFTSurvivalRegressionModel']):
    def setQuantileProbabilities(self, value: list[float]) -> AFTSurvivalRegressionModel: ...
    def setQuantilesCol(self, value: str) -> AFTSurvivalRegressionModel: ...
    @property
    def coefficients(self) -> Vector: ...
    @property
    def intercept(self) -> float: ...
    @property
    def scale(self) -> float: ...
    def predictQuantiles(self, features: Vector) -> Vector: ...

class _GeneralizedLinearRegressionParams(_PredictorParams, HasFitIntercept, HasMaxIter, HasTol, HasRegParam, HasWeightCol, HasSolver, HasAggregationDepth):
    family: Param[str]
    link: Param[str]
    linkPredictionCol: Param[str]
    variancePower: Param[float]
    linkPower: Param[float]
    solver: Param[str]
    offsetCol: Param[str]
    def __init__(self, *args: Any) -> None: ...
    def getFamily(self) -> str: ...
    def getLinkPredictionCol(self) -> str: ...
    def getLink(self) -> str: ...
    def getVariancePower(self) -> float: ...
    def getLinkPower(self) -> float: ...
    def getOffsetCol(self) -> str: ...

class GeneralizedLinearRegression(_JavaRegressor['GeneralizedLinearRegressionModel'], _GeneralizedLinearRegressionParams, JavaMLWritable, JavaMLReadable['GeneralizedLinearRegression']):
    def __init__(self, *, labelCol: str = 'label', featuresCol: str = 'features', predictionCol: str = 'prediction', family: str = 'gaussian', link: str | None = None, fitIntercept: bool = True, maxIter: int = 25, tol: float = 1e-06, regParam: float = 0.0, weightCol: str | None = None, solver: str = 'irls', linkPredictionCol: str | None = None, variancePower: float = 0.0, linkPower: float | None = None, offsetCol: str | None = None, aggregationDepth: int = 2) -> None: ...
    def setParams(self, *, labelCol: str = 'label', featuresCol: str = 'features', predictionCol: str = 'prediction', family: str = 'gaussian', link: str | None = None, fitIntercept: bool = True, maxIter: int = 25, tol: float = 1e-06, regParam: float = 0.0, weightCol: str | None = None, solver: str = 'irls', linkPredictionCol: str | None = None, variancePower: float = 0.0, linkPower: float | None = None, offsetCol: str | None = None, aggregationDepth: int = 2) -> GeneralizedLinearRegression: ...
    def setFamily(self, value: str) -> GeneralizedLinearRegression: ...
    def setLinkPredictionCol(self, value: str) -> GeneralizedLinearRegression: ...
    def setLink(self, value: str) -> GeneralizedLinearRegression: ...
    def setVariancePower(self, value: float) -> GeneralizedLinearRegression: ...
    def setLinkPower(self, value: float) -> GeneralizedLinearRegression: ...
    def setOffsetCol(self, value: str) -> GeneralizedLinearRegression: ...
    def setMaxIter(self, value: int) -> GeneralizedLinearRegression: ...
    def setRegParam(self, value: float) -> GeneralizedLinearRegression: ...
    def setTol(self, value: float) -> GeneralizedLinearRegression: ...
    def setFitIntercept(self, value: bool) -> GeneralizedLinearRegression: ...
    def setWeightCol(self, value: str) -> GeneralizedLinearRegression: ...
    def setSolver(self, value: str) -> GeneralizedLinearRegression: ...
    def setAggregationDepth(self, value: int) -> GeneralizedLinearRegression: ...

class GeneralizedLinearRegressionModel(_JavaRegressionModel[Vector], _GeneralizedLinearRegressionParams, JavaMLWritable, JavaMLReadable['GeneralizedLinearRegressionModel'], HasTrainingSummary['GeneralizedLinearRegressionTrainingSummary']):
    def setLinkPredictionCol(self, value: str) -> GeneralizedLinearRegressionModel: ...
    @property
    def coefficients(self) -> Vector: ...
    @property
    def intercept(self) -> float: ...
    @property
    def summary(self) -> GeneralizedLinearRegressionTrainingSummary: ...
    def evaluate(self, dataset: DataFrame) -> GeneralizedLinearRegressionSummary: ...

class GeneralizedLinearRegressionSummary(JavaWrapper):
    @property
    def predictions(self) -> DataFrame: ...
    @property
    def predictionCol(self) -> str: ...
    @property
    def numInstances(self) -> int: ...
    @property
    def rank(self) -> int: ...
    @property
    def degreesOfFreedom(self) -> int: ...
    @property
    def residualDegreeOfFreedom(self) -> int: ...
    @property
    def residualDegreeOfFreedomNull(self) -> int: ...
    def residuals(self, residualsType: str = 'deviance') -> DataFrame: ...
    @property
    def nullDeviance(self) -> float: ...
    @property
    def deviance(self) -> float: ...
    @property
    def dispersion(self) -> float: ...
    @property
    def aic(self) -> float: ...

class GeneralizedLinearRegressionTrainingSummary(GeneralizedLinearRegressionSummary):
    @property
    def numIterations(self) -> int: ...
    @property
    def solver(self) -> str: ...
    @property
    def coefficientStandardErrors(self) -> list[float]: ...
    @property
    def tValues(self) -> list[float]: ...
    @property
    def pValues(self) -> list[float]: ...

class _FactorizationMachinesParams(_PredictorParams, HasMaxIter, HasStepSize, HasTol, HasSolver, HasSeed, HasFitIntercept, HasRegParam, HasWeightCol):
    factorSize: Param[int]
    fitLinear: Param[bool]
    miniBatchFraction: Param[float]
    initStd: Param[float]
    solver: Incomplete
    def __init__(self, *args: Any) -> None: ...
    def getFactorSize(self) -> int: ...
    def getFitLinear(self) -> bool: ...
    def getMiniBatchFraction(self) -> float: ...
    def getInitStd(self) -> float: ...

class FMRegressor(_JavaRegressor['FMRegressionModel'], _FactorizationMachinesParams, JavaMLWritable, JavaMLReadable['FMRegressor']):
    def __init__(self, *, featuresCol: str = 'features', labelCol: str = 'label', predictionCol: str = 'prediction', factorSize: int = 8, fitIntercept: bool = True, fitLinear: bool = True, regParam: float = 0.0, miniBatchFraction: float = 1.0, initStd: float = 0.01, maxIter: int = 100, stepSize: float = 1.0, tol: float = 1e-06, solver: str = 'adamW', seed: int | None = None) -> None: ...
    def setParams(self, *, featuresCol: str = 'features', labelCol: str = 'label', predictionCol: str = 'prediction', factorSize: int = 8, fitIntercept: bool = True, fitLinear: bool = True, regParam: float = 0.0, miniBatchFraction: float = 1.0, initStd: float = 0.01, maxIter: int = 100, stepSize: float = 1.0, tol: float = 1e-06, solver: str = 'adamW', seed: int | None = None) -> FMRegressor: ...
    def setFactorSize(self, value: int) -> FMRegressor: ...
    def setFitLinear(self, value: bool) -> FMRegressor: ...
    def setMiniBatchFraction(self, value: float) -> FMRegressor: ...
    def setInitStd(self, value: float) -> FMRegressor: ...
    def setMaxIter(self, value: int) -> FMRegressor: ...
    def setStepSize(self, value: float) -> FMRegressor: ...
    def setTol(self, value: float) -> FMRegressor: ...
    def setSolver(self, value: str) -> FMRegressor: ...
    def setSeed(self, value: int) -> FMRegressor: ...
    def setFitIntercept(self, value: bool) -> FMRegressor: ...
    def setRegParam(self, value: float) -> FMRegressor: ...

class FMRegressionModel(_JavaRegressionModel, _FactorizationMachinesParams, JavaMLWritable, JavaMLReadable['FMRegressionModel']):
    @property
    def intercept(self) -> float: ...
    @property
    def linear(self) -> Vector: ...
    @property
    def factors(self) -> Matrix: ...