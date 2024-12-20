from _typeshed import Incomplete
from inspect import Signature
from typing import Any

def get_logger() -> Any: ...

class PandasOnSparkUsageLogger:
    logger: Incomplete
    def __init__(self) -> None: ...
    def log_success(self, class_name: str, name: str, duration: float, signature: Signature | None = None) -> None: ...
    def log_failure(self, class_name: str, name: str, ex: Exception, duration: float, signature: Signature | None = None) -> None: ...
    def log_missing(self, class_name: str, name: str, is_deprecated: bool = False, signature: Signature | None = None) -> None: ...
