import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark() -> SparkSession:
    return SparkSession.builder \
        .appName("pyspark-type-tests") \
        .master("local[1]") \
        .getOrCreate() 