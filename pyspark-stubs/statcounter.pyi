import math
from _typeshed import Incomplete
from typing import Iterable

maximum = max
minimum = min
sqrt = math.sqrt

class StatCounter:
    n: int
    mu: float
    m2: float
    maxValue: Incomplete
    minValue: Incomplete
    def __init__(self, values: Iterable[float] | None = None) -> None: ...
    def merge(self, value: float) -> StatCounter: ...
    def mergeStats(self, other: StatCounter) -> StatCounter: ...
    def copy(self) -> StatCounter: ...
    def count(self) -> int: ...
    def mean(self) -> float: ...
    def sum(self) -> float: ...
    def min(self) -> float: ...
    def max(self) -> float: ...
    def variance(self) -> float: ...
    def sampleVariance(self) -> float: ...
    def stdev(self) -> float: ...
    def sampleStdev(self) -> float: ...
    def asDict(self, sample: bool = False) -> dict[str, float]: ...
