import numpy as np
from _typeshed import Incomplete
from pyspark.mllib._typing import NormType
from pyspark.sql.types import StructType, UserDefinedType
from typing import Any, Iterable, overload

__all__ = ['Vector', 'DenseVector', 'SparseVector', 'Vectors', 'Matrix', 'DenseMatrix', 'SparseMatrix', 'Matrices']

class VectorUDT(UserDefinedType):
    @classmethod
    def sqlType(cls) -> StructType: ...
    @classmethod
    def module(cls) -> str: ...
    @classmethod
    def scalaUDT(cls) -> str: ...
    def serialize(self, obj: Vector) -> tuple[int, int | None, list[int] | None, list[float]]: ...
    def deserialize(self, datum: tuple[int, int | None, list[int] | None, list[float]]) -> Vector: ...
    def simpleString(self) -> str: ...

class MatrixUDT(UserDefinedType):
    @classmethod
    def sqlType(cls) -> StructType: ...
    @classmethod
    def module(cls) -> str: ...
    @classmethod
    def scalaUDT(cls) -> str: ...
    def serialize(self, obj: Matrix) -> tuple[int, int, int, list[int] | None, list[int] | None, list[float], bool]: ...
    def deserialize(self, datum: tuple[int, int, int, list[int] | None, list[int] | None, list[float], bool]) -> Matrix: ...
    def simpleString(self) -> str: ...

class Vector:
    __UDT__: Incomplete
    def toArray(self) -> np.ndarray: ...
    def __len__(self) -> int: ...

class DenseVector(Vector):
    array: Incomplete
    def __init__(self, ar: bytes | np.ndarray | Iterable[float]) -> None: ...
    def __reduce__(self) -> tuple[type['DenseVector'], tuple[bytes]]: ...
    def numNonzeros(self) -> int: ...
    def norm(self, p: NormType) -> np.float64: ...
    def dot(self, other: Iterable[float]) -> np.float64: ...
    def squared_distance(self, other: Iterable[float]) -> np.float64: ...
    def toArray(self) -> np.ndarray: ...
    @property
    def values(self) -> np.ndarray: ...
    @overload
    def __getitem__(self, item: int) -> np.float64: ...
    @overload
    def __getitem__(self, item: slice) -> np.ndarray: ...
    def __len__(self) -> int: ...
    def __eq__(self, other: Any) -> bool: ...
    def __ne__(self, other: Any) -> bool: ...
    def __hash__(self) -> int: ...
    def __getattr__(self, item: str) -> Any: ...
    def __neg__(self) -> DenseVector: ...
    __add__: Incomplete
    __sub__: Incomplete
    __mul__: Incomplete
    __div__: Incomplete
    __truediv__: Incomplete
    __mod__: Incomplete
    __radd__: Incomplete
    __rsub__: Incomplete
    __rmul__: Incomplete
    __rdiv__: Incomplete
    __rtruediv__: Incomplete
    __rmod__: Incomplete

class SparseVector(Vector):
    @overload
    def __init__(self, size: int, /, __indices: bytes, __values: bytes) -> None: ...
    @overload
    def __init__(self, size: int, *args: tuple[int, float]) -> None: ...
    @overload
    def __init__(self, size: int, /, __indices: Iterable[int], __values: Iterable[float]) -> None: ...
    @overload
    def __init__(self, /, size: int, __pairs: Iterable[tuple[int, float]]) -> None: ...
    @overload
    def __init__(self, /, size: int, __map: dict[int, float]) -> None: ...
    def numNonzeros(self) -> int: ...
    def norm(self, p: NormType) -> np.float64: ...
    def __reduce__(self) -> tuple[type['SparseVector'], tuple[int, bytes, bytes]]: ...
    def dot(self, other: Iterable[float]) -> np.float64: ...
    def squared_distance(self, other: Iterable[float]) -> np.float64: ...
    def toArray(self) -> np.ndarray: ...
    def __len__(self) -> int: ...
    def __eq__(self, other: Any) -> bool: ...
    def __getitem__(self, index: int) -> np.float64: ...
    def __ne__(self, other: Any) -> bool: ...
    def __hash__(self) -> int: ...

class Vectors:
    @staticmethod
    @overload
    def sparse(size: int, __indices: bytes, /, __values: bytes) -> SparseVector: ...
    @staticmethod
    @overload
    def sparse(size: int, *args: tuple[int, float]) -> SparseVector: ...
    @staticmethod
    @overload
    def sparse(size: int, __indices: Iterable[int], /, __values: Iterable[float]) -> SparseVector: ...
    @staticmethod
    @overload
    def sparse(size: int, /, __pairs: Iterable[tuple[int, float]]) -> SparseVector: ...
    @staticmethod
    @overload
    def sparse(size: int, /, __map: dict[int, float]) -> SparseVector: ...
    @overload
    @staticmethod
    def dense(*elements: float) -> DenseVector: ...
    @overload
    @staticmethod
    def dense(__arr: bytes, /) -> DenseVector: ...
    @overload
    @staticmethod
    def dense(__arr: Iterable[float], /) -> DenseVector: ...
    @staticmethod
    def squared_distance(v1: Vector, v2: Vector) -> np.float64: ...
    @staticmethod
    def norm(vector: Vector, p: NormType) -> np.float64: ...
    @staticmethod
    def zeros(size: int) -> DenseVector: ...

class Matrix:
    __UDT__: Incomplete
    numRows: Incomplete
    numCols: Incomplete
    isTransposed: Incomplete
    def __init__(self, numRows: int, numCols: int, isTransposed: bool = False) -> None: ...
    def toArray(self) -> np.ndarray: ...

class DenseMatrix(Matrix):
    values: Incomplete
    def __init__(self, numRows: int, numCols: int, values: bytes | Iterable[float], isTransposed: bool = False) -> None: ...
    def __reduce__(self) -> tuple[type['DenseMatrix'], tuple[int, int, bytes, int]]: ...
    def toArray(self) -> np.ndarray: ...
    def toSparse(self) -> SparseMatrix: ...
    def __getitem__(self, indices: tuple[int, int]) -> np.float64: ...
    def __eq__(self, other: Any) -> bool: ...

class SparseMatrix(Matrix):
    colPtrs: Incomplete
    rowIndices: Incomplete
    values: Incomplete
    def __init__(self, numRows: int, numCols: int, colPtrs: bytes | Iterable[int], rowIndices: bytes | Iterable[int], values: bytes | Iterable[float], isTransposed: bool = False) -> None: ...
    def __reduce__(self) -> tuple[type['SparseMatrix'], tuple[int, int, bytes, bytes, bytes, int]]: ...
    def __getitem__(self, indices: tuple[int, int]) -> np.float64: ...
    def toArray(self) -> np.ndarray: ...
    def toDense(self) -> DenseMatrix: ...
    def __eq__(self, other: Any) -> bool: ...

class Matrices:
    @staticmethod
    def dense(numRows: int, numCols: int, values: bytes | Iterable[float]) -> DenseMatrix: ...
    @staticmethod
    def sparse(numRows: int, numCols: int, colPtrs: bytes | Iterable[int], rowIndices: bytes | Iterable[int], values: bytes | Iterable[float]) -> SparseMatrix: ...
