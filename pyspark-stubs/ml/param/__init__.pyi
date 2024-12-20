from _typeshed import Incomplete
from abc import ABCMeta
from pyspark.ml._typing import ParamMap
from pyspark.ml.linalg import Matrix, Vector
from pyspark.ml.util import Identifiable
from typing import Any, Callable, Generic, TypeVar, overload

__all__ = ['Param', 'Params', 'TypeConverters']

T = TypeVar('T')
P = TypeVar('P', bound='Params')

class Param(Generic[T]):
    parent: Incomplete
    name: Incomplete
    doc: Incomplete
    typeConverter: Incomplete
    def __init__(self, parent: Identifiable, name: str, doc: str, typeConverter: Callable[[Any], T] | None = None) -> None: ...
    def __hash__(self) -> int: ...
    def __eq__(self, other: Any) -> bool: ...

class TypeConverters:
    @staticmethod
    def identity(value: T) -> T: ...
    @staticmethod
    def toList(value: Any) -> list: ...
    @staticmethod
    def toListFloat(value: Any) -> list[float]: ...
    @staticmethod
    def toListListFloat(value: Any) -> list[list[float]]: ...
    @staticmethod
    def toListInt(value: Any) -> list[int]: ...
    @staticmethod
    def toListString(value: Any) -> list[str]: ...
    @staticmethod
    def toVector(value: Any) -> Vector: ...
    @staticmethod
    def toMatrix(value: Any) -> Matrix: ...
    @staticmethod
    def toFloat(value: Any) -> float: ...
    @staticmethod
    def toInt(value: Any) -> int: ...
    @staticmethod
    def toString(value: Any) -> str: ...
    @staticmethod
    def toBoolean(value: Any) -> bool: ...

class Params(Identifiable, metaclass=ABCMeta):
    def __init__(self) -> None: ...
    @property
    def params(self) -> list[Param]: ...
    def explainParam(self, param: str | Param) -> str: ...
    def explainParams(self) -> str: ...
    def getParam(self, paramName: str) -> Param: ...
    def isSet(self, param: str | Param[Any]) -> bool: ...
    def hasDefault(self, param: str | Param[Any]) -> bool: ...
    def isDefined(self, param: str | Param[Any]) -> bool: ...
    def hasParam(self, paramName: str) -> bool: ...
    @overload
    def getOrDefault(self, param: str) -> Any: ...
    @overload
    def getOrDefault(self, param: Param[T]) -> T: ...
    def extractParamMap(self, extra: ParamMap | None = None) -> ParamMap: ...
    def copy(self, extra: ParamMap | None = None) -> P: ...
    def set(self, param: Param, value: Any) -> None: ...
    def clear(self, param: Param) -> None: ...
