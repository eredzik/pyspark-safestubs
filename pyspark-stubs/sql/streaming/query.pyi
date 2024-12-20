from py4j.java_gateway import JavaObject
from pyspark.errors import StreamingQueryException
from pyspark.sql.streaming.listener import StreamingQueryListener
from typing import Any

__all__ = ['StreamingQuery', 'StreamingQueryManager']

class StreamingQuery:
    def __init__(self, jsq: JavaObject) -> None: ...
    @property
    def id(self) -> str: ...
    @property
    def runId(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def isActive(self) -> bool: ...
    def awaitTermination(self, timeout: int | None = None) -> bool | None: ...
    @property
    def status(self) -> dict[str, Any]: ...
    @property
    def recentProgress(self) -> list[dict[str, Any]]: ...
    @property
    def lastProgress(self) -> dict[str, Any] | None: ...
    def processAllAvailable(self) -> None: ...
    def stop(self) -> None: ...
    def explain(self, extended: bool = False) -> None: ...
    def exception(self) -> StreamingQueryException | None: ...

class StreamingQueryManager:
    def __init__(self, jsqm: JavaObject) -> None: ...
    @property
    def active(self) -> list[StreamingQuery]: ...
    def get(self, id: str) -> StreamingQuery | None: ...
    def awaitAnyTermination(self, timeout: int | None = None) -> bool | None: ...
    def resetTerminated(self) -> None: ...
    def addListener(self, listener: StreamingQueryListener) -> None: ...
    def removeListener(self, listener: StreamingQueryListener) -> None: ...
