import pandas as pd
import pyspark.pandas as ps
from pyspark.pandas.internal import InternalField as InternalField
from pyspark.sql.types import StructField as StructField
from typing import Any, Callable

class CategoricalAccessor:
    def __init__(self, series: ps.Series) -> None: ...
    @property
    def categories(self) -> pd.Index: ...
    @categories.setter
    def categories(self, categories: pd.Index | list) -> None: ...
    @property
    def ordered(self) -> bool: ...
    @property
    def codes(self) -> ps.Series: ...
    def add_categories(self, new_categories: pd.Index | Any | list) -> ps.Series | None: ...
    def as_ordered(self, inplace: bool = False) -> ps.Series | None: ...
    def as_unordered(self, inplace: bool = False) -> ps.Series | None: ...
    def remove_categories(self, removals: pd.Index | Any | list) -> ps.Series | None: ...
    def remove_unused_categories(self) -> ps.Series | None: ...
    def rename_categories(self, new_categories: list | dict | Callable) -> ps.Series | None: ...
    def reorder_categories(self, new_categories: pd.Index | list, ordered: bool | None = None) -> ps.Series | None: ...
    def set_categories(self, new_categories: pd.Index | list, ordered: bool | None = None, rename: bool = False) -> ps.Series | None: ...
