from py4j.java_gateway import JavaGateway, JavaObject
from pyspark.accumulators import Accumulator, AccumulatorParam
from pyspark.broadcast import Broadcast
from pyspark.conf import SparkConf
from pyspark.profiler import BasicProfiler, MemoryProfiler, ProfilerCollector, UDFBasicProfiler
from pyspark.rdd import RDD
from pyspark.resource.information import ResourceInformation
from pyspark.serializers import Serializer
from pyspark.status import StatusTracker
from types import TracebackType
from typing import Any, Callable, Iterable, NoReturn, Sequence, TypeVar

__all__ = ['SparkContext']

T = TypeVar('T')
U = TypeVar('U')

class SparkContext:
    serializer: Serializer
    profiler_collector: ProfilerCollector
    PACKAGE_EXTENSIONS: Iterable[str]
    def __init__(self, master: str | None = None, appName: str | None = None, sparkHome: str | None = None, pyFiles: list[str] | None = None, environment: dict[str, Any] | None = None, batchSize: int = 0, serializer: Serializer = ..., conf: SparkConf | None = None, gateway: JavaGateway | None = None, jsc: JavaObject | None = None, profiler_cls: type[BasicProfiler] = ..., udf_profiler_cls: type[UDFBasicProfiler] = ..., memory_profiler_cls: type[MemoryProfiler] = ...) -> None: ...
    def __getnewargs__(self) -> NoReturn: ...
    def __enter__(self) -> SparkContext: ...
    def __exit__(self, type: type[BaseException] | None, value: BaseException | None, trace: TracebackType | None) -> None: ...
    @classmethod
    def getOrCreate(cls, conf: SparkConf | None = None) -> SparkContext: ...
    def setLogLevel(self, logLevel: str) -> None: ...
    @classmethod
    def setSystemProperty(cls, key: str, value: str) -> None: ...
    @property
    def version(self) -> str: ...
    @property
    def applicationId(self) -> str: ...
    @property
    def uiWebUrl(self) -> str | None: ...
    @property
    def startTime(self) -> int: ...
    @property
    def defaultParallelism(self) -> int: ...
    @property
    def defaultMinPartitions(self) -> int: ...
    def stop(self) -> None: ...
    def emptyRDD(self) -> RDD[Any]: ...
    def range(self, start: int, end: int | None = None, step: int = 1, numSlices: int | None = None) -> RDD[int]: ...
    def parallelize(self, c: Iterable[T], numSlices: int | None = None) -> RDD[T]: ...
    def pickleFile(self, name: str, minPartitions: int | None = None) -> RDD[Any]: ...
    def textFile(self, name: str, minPartitions: int | None = None, use_unicode: bool = True) -> RDD[str]: ...
    def wholeTextFiles(self, path: str, minPartitions: int | None = None, use_unicode: bool = True) -> RDD[tuple[str, str]]: ...
    def binaryFiles(self, path: str, minPartitions: int | None = None) -> RDD[tuple[str, bytes]]: ...
    def binaryRecords(self, path: str, recordLength: int) -> RDD[bytes]: ...
    def sequenceFile(self, path: str, keyClass: str | None = None, valueClass: str | None = None, keyConverter: str | None = None, valueConverter: str | None = None, minSplits: int | None = None, batchSize: int = 0) -> RDD[tuple[T, U]]: ...
    def newAPIHadoopFile(self, path: str, inputFormatClass: str, keyClass: str, valueClass: str, keyConverter: str | None = None, valueConverter: str | None = None, conf: dict[str, str] | None = None, batchSize: int = 0) -> RDD[tuple[T, U]]: ...
    def newAPIHadoopRDD(self, inputFormatClass: str, keyClass: str, valueClass: str, keyConverter: str | None = None, valueConverter: str | None = None, conf: dict[str, str] | None = None, batchSize: int = 0) -> RDD[tuple[T, U]]: ...
    def hadoopFile(self, path: str, inputFormatClass: str, keyClass: str, valueClass: str, keyConverter: str | None = None, valueConverter: str | None = None, conf: dict[str, str] | None = None, batchSize: int = 0) -> RDD[tuple[T, U]]: ...
    def hadoopRDD(self, inputFormatClass: str, keyClass: str, valueClass: str, keyConverter: str | None = None, valueConverter: str | None = None, conf: dict[str, str] | None = None, batchSize: int = 0) -> RDD[tuple[T, U]]: ...
    def union(self, rdds: list[RDD[T]]) -> RDD[T]: ...
    def broadcast(self, value: T) -> Broadcast[T]: ...
    def accumulator(self, value: T, accum_param: AccumulatorParam[T] | None = None) -> Accumulator[T]: ...
    def addFile(self, path: str, recursive: bool = False) -> None: ...
    @property
    def listFiles(self) -> list[str]: ...
    def addPyFile(self, path: str) -> None: ...
    def addArchive(self, path: str) -> None: ...
    @property
    def listArchives(self) -> list[str]: ...
    def setCheckpointDir(self, dirName: str) -> None: ...
    def getCheckpointDir(self) -> str | None: ...
    def setJobGroup(self, groupId: str, description: str, interruptOnCancel: bool = False) -> None: ...
    def setInterruptOnCancel(self, interruptOnCancel: bool) -> None: ...
    def addJobTag(self, tag: str) -> None: ...
    def removeJobTag(self, tag: str) -> None: ...
    def getJobTags(self) -> set[str]: ...
    def clearJobTags(self) -> None: ...
    def setLocalProperty(self, key: str, value: str) -> None: ...
    def getLocalProperty(self, key: str) -> str | None: ...
    def setJobDescription(self, value: str) -> None: ...
    def sparkUser(self) -> str: ...
    def cancelJobGroup(self, groupId: str) -> None: ...
    def cancelJobsWithTag(self, tag: str) -> None: ...
    def cancelAllJobs(self) -> None: ...
    def statusTracker(self) -> StatusTracker: ...
    def runJob(self, rdd: RDD[T], partitionFunc: Callable[[Iterable[T]], Iterable[U]], partitions: Sequence[int] | None = None, allowLocal: bool = False) -> list[U]: ...
    def show_profiles(self) -> None: ...
    def dump_profiles(self, path: str) -> None: ...
    def getConf(self) -> SparkConf: ...
    @property
    def resources(self) -> dict[str, ResourceInformation]: ...
