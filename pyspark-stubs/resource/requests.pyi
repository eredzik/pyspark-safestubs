from py4j.java_gateway import JVMView as JVMView, JavaObject as JavaObject
from typing import overload

class ExecutorResourceRequest:
    def __init__(self, resourceName: str, amount: int, discoveryScript: str = '', vendor: str = '') -> None: ...
    @property
    def resourceName(self) -> str: ...
    @property
    def amount(self) -> int: ...
    @property
    def discoveryScript(self) -> str: ...
    @property
    def vendor(self) -> str: ...

class ExecutorResourceRequests:
    @overload
    def __init__(self, _jvm: JVMView) -> None: ...
    @overload
    def __init__(self, _jvm: None = ..., _requests: dict[str, ExecutorResourceRequest] | None = ...) -> None: ...
    def memory(self, amount: str) -> ExecutorResourceRequests: ...
    def memoryOverhead(self, amount: str) -> ExecutorResourceRequests: ...
    def pysparkMemory(self, amount: str) -> ExecutorResourceRequests: ...
    def offheapMemory(self, amount: str) -> ExecutorResourceRequests: ...
    def cores(self, amount: int) -> ExecutorResourceRequests: ...
    def resource(self, resourceName: str, amount: int, discoveryScript: str = '', vendor: str = '') -> ExecutorResourceRequests: ...
    @property
    def requests(self) -> dict[str, ExecutorResourceRequest]: ...

class TaskResourceRequest:
    def __init__(self, resourceName: str, amount: float) -> None: ...
    @property
    def resourceName(self) -> str: ...
    @property
    def amount(self) -> float: ...

class TaskResourceRequests:
    @overload
    def __init__(self, _jvm: JVMView) -> None: ...
    @overload
    def __init__(self, _jvm: None = ..., _requests: dict[str, TaskResourceRequest] | None = ...) -> None: ...
    def cpus(self, amount: int) -> TaskResourceRequests: ...
    def resource(self, resourceName: str, amount: float) -> TaskResourceRequests: ...
    @property
    def requests(self) -> dict[str, TaskResourceRequest]: ...
