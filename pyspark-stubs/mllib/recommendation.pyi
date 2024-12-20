import array
from pyspark import SparkContext
from pyspark.mllib.common import JavaModelWrapper
from pyspark.mllib.util import JavaLoader, JavaSaveable
from pyspark.rdd import RDD
from typing import NamedTuple

__all__ = ['MatrixFactorizationModel', 'ALS', 'Rating']

class Rating(NamedTuple):
    user: int
    product: int
    rating: float
    def __reduce__(self) -> tuple[type['Rating'], tuple[int, int, float]]: ...

class MatrixFactorizationModel(JavaModelWrapper, JavaSaveable, JavaLoader['MatrixFactorizationModel']):
    def predict(self, user: int, product: int) -> float: ...
    def predictAll(self, user_product: RDD[tuple[int, int]]) -> RDD[Rating]: ...
    def userFeatures(self) -> RDD[tuple[int, array.array]]: ...
    def productFeatures(self) -> RDD[tuple[int, array.array]]: ...
    def recommendUsers(self, product: int, num: int) -> list[Rating]: ...
    def recommendProducts(self, user: int, num: int) -> list[Rating]: ...
    def recommendProductsForUsers(self, num: int) -> RDD[tuple[int, tuple[Rating, ...]]]: ...
    def recommendUsersForProducts(self, num: int) -> RDD[tuple[int, tuple[Rating, ...]]]: ...
    @property
    def rank(self) -> int: ...
    @classmethod
    def load(cls, sc: SparkContext, path: str) -> MatrixFactorizationModel: ...

class ALS:
    @classmethod
    def train(cls, ratings: RDD[Rating] | RDD[tuple[int, int, float]], rank: int, iterations: int = 5, lambda_: float = 0.01, blocks: int = -1, nonnegative: bool = False, seed: int | None = None) -> MatrixFactorizationModel: ...
    @classmethod
    def trainImplicit(cls, ratings: RDD[Rating] | RDD[tuple[int, int, float]], rank: int, iterations: int = 5, lambda_: float = 0.01, blocks: int = -1, alpha: float = 0.01, nonnegative: bool = False, seed: int | None = None) -> MatrixFactorizationModel: ...
