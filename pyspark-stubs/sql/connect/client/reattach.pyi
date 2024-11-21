import pyspark.sql.connect.proto as pb2
import pyspark.sql.connect.proto.base_pb2_grpc as grpc_lib
from collections.abc import Generator
from pyspark.sql.connect.utils import check_dependencies as check_dependencies
from typing import Any, Iterable

class ExecutePlanResponseReattachableIterator(Generator):
    @classmethod
    def shutdown(cls) -> None: ...
    def __init__(self, request: pb2.ExecutePlanRequest, stub: grpc_lib.SparkConnectServiceStub, retry_policy: dict[str, Any], metadata: Iterable[tuple[str, str]]) -> None: ...
    def send(self, value: Any) -> pb2.ExecutePlanResponse: ...
    def throw(self, type: Any = None, value: Any = None, traceback: Any = None) -> Any: ...
    def close(self) -> None: ...
    def __del__(self) -> None: ...

class RetryException(Exception): ...
