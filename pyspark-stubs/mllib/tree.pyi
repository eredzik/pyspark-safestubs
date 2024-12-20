from pyspark.mllib._typing import VectorLike
from pyspark.mllib.common import JavaModelWrapper
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.util import JavaLoader, JavaSaveable
from pyspark.rdd import RDD
from typing import overload

__all__ = ['DecisionTreeModel', 'DecisionTree', 'RandomForestModel', 'RandomForest', 'GradientBoostedTreesModel', 'GradientBoostedTrees']

class TreeEnsembleModel(JavaModelWrapper, JavaSaveable):
    @overload
    def predict(self, x: VectorLike) -> float: ...
    @overload
    def predict(self, x: RDD['VectorLike']) -> RDD[float]: ...
    def numTrees(self) -> int: ...
    def totalNumNodes(self) -> int: ...
    def toDebugString(self) -> str: ...

class DecisionTreeModel(JavaModelWrapper, JavaSaveable, JavaLoader['DecisionTreeModel']):
    @overload
    def predict(self, x: VectorLike) -> float: ...
    @overload
    def predict(self, x: RDD['VectorLike']) -> RDD[float]: ...
    def numNodes(self) -> int: ...
    def depth(self) -> int: ...
    def toDebugString(self) -> str: ...

class DecisionTree:
    @classmethod
    def trainClassifier(cls, data: RDD[LabeledPoint], numClasses: int, categoricalFeaturesInfo: dict[int, int], impurity: str = 'gini', maxDepth: int = 5, maxBins: int = 32, minInstancesPerNode: int = 1, minInfoGain: float = 0.0) -> DecisionTreeModel: ...
    @classmethod
    def trainRegressor(cls, data: RDD[LabeledPoint], categoricalFeaturesInfo: dict[int, int], impurity: str = 'variance', maxDepth: int = 5, maxBins: int = 32, minInstancesPerNode: int = 1, minInfoGain: float = 0.0) -> DecisionTreeModel: ...

class RandomForestModel(TreeEnsembleModel, JavaLoader['RandomForestModel']): ...

class RandomForest:
    supportedFeatureSubsetStrategies: tuple[str, ...]
    @classmethod
    def trainClassifier(cls, data: RDD[LabeledPoint], numClasses: int, categoricalFeaturesInfo: dict[int, int], numTrees: int, featureSubsetStrategy: str = 'auto', impurity: str = 'gini', maxDepth: int = 4, maxBins: int = 32, seed: int | None = None) -> RandomForestModel: ...
    @classmethod
    def trainRegressor(cls, data: RDD[LabeledPoint], categoricalFeaturesInfo: dict[int, int], numTrees: int, featureSubsetStrategy: str = 'auto', impurity: str = 'variance', maxDepth: int = 4, maxBins: int = 32, seed: int | None = None) -> RandomForestModel: ...

class GradientBoostedTreesModel(TreeEnsembleModel, JavaLoader['GradientBoostedTreesModel']): ...

class GradientBoostedTrees:
    @classmethod
    def trainClassifier(cls, data: RDD[LabeledPoint], categoricalFeaturesInfo: dict[int, int], loss: str = 'logLoss', numIterations: int = 100, learningRate: float = 0.1, maxDepth: int = 3, maxBins: int = 32) -> GradientBoostedTreesModel: ...
    @classmethod
    def trainRegressor(cls, data: RDD[LabeledPoint], categoricalFeaturesInfo: dict[int, int], loss: str = 'leastSquaresError', numIterations: int = 100, learningRate: float = 0.1, maxDepth: int = 3, maxBins: int = 32) -> GradientBoostedTreesModel: ...
