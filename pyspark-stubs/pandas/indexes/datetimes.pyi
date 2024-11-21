import datetime
from _typeshed import Incomplete
from pandas.tseries.offsets import DateOffset as DateOffset
from pyspark.pandas.indexes.base import Index as Index
from pyspark.pandas.missing.indexes import MissingPandasLikeDatetimeIndex as MissingPandasLikeDatetimeIndex
from pyspark.pandas.series import Series as Series, first_series as first_series
from pyspark.pandas.utils import verify_temp_column_name as verify_temp_column_name
from typing import Any

class DatetimeIndex(Index):
    def __new__(cls, data: Incomplete | None = None, freq=..., normalize: bool = False, closed: Incomplete | None = None, ambiguous: str = 'raise', dayfirst: bool = False, yearfirst: bool = False, dtype: Incomplete | None = None, copy: bool = False, name: Incomplete | None = None): ...
    def __getattr__(self, item: str) -> Any: ...
    @property
    def year(self) -> Index: ...
    @property
    def month(self) -> Index: ...
    @property
    def day(self) -> Index: ...
    @property
    def hour(self) -> Index: ...
    @property
    def minute(self) -> Index: ...
    @property
    def second(self) -> Index: ...
    @property
    def microsecond(self) -> Index: ...
    @property
    def week(self) -> Index: ...
    @property
    def weekofyear(self) -> Index: ...
    @property
    def dayofweek(self) -> Index: ...
    @property
    def day_of_week(self) -> Index: ...
    @property
    def weekday(self) -> Index: ...
    @property
    def dayofyear(self) -> Index: ...
    @property
    def day_of_year(self) -> Index: ...
    @property
    def quarter(self) -> Index: ...
    @property
    def is_month_start(self) -> Index: ...
    @property
    def is_month_end(self) -> Index: ...
    @property
    def is_quarter_start(self) -> Index: ...
    @property
    def is_quarter_end(self) -> Index: ...
    @property
    def is_year_start(self) -> Index: ...
    @property
    def is_year_end(self) -> Index: ...
    @property
    def is_leap_year(self) -> Index: ...
    @property
    def daysinmonth(self) -> Index: ...
    @property
    def days_in_month(self) -> Index: ...
    def ceil(self, freq: str | DateOffset, *args: Any, **kwargs: Any) -> DatetimeIndex: ...
    def floor(self, freq: str | DateOffset, *args: Any, **kwargs: Any) -> DatetimeIndex: ...
    def round(self, freq: str | DateOffset, *args: Any, **kwargs: Any) -> DatetimeIndex: ...
    def month_name(self, locale: str | None = None) -> Index: ...
    def day_name(self, locale: str | None = None) -> Index: ...
    def normalize(self) -> DatetimeIndex: ...
    def strftime(self, date_format: str) -> Index: ...
    def indexer_between_time(self, start_time: datetime.time | str, end_time: datetime.time | str, include_start: bool = True, include_end: bool = True) -> Index: ...
    def indexer_at_time(self, time: datetime.time | str, asof: bool = False) -> Index: ...
    def all(self, *args, **kwargs) -> None: ...

def disallow_nanoseconds(freq: str | DateOffset) -> None: ...