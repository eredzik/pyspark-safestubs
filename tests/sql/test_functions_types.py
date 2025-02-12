from typing import TYPE_CHECKING, Literal, TypeVar

from pyspark.sql import functions as F
from pyspark.sql.column import Column

if TYPE_CHECKING:
    try:
        from typing import assert_type
    except ImportError:
        from typing_extensions import assert_type

T = TypeVar("T", bound=str)


def test_column_type_propagation() -> None:
    # Test that column name type is preserved
    col1 = F.col("test_col")
    assert_type(col1, Column[Literal["test_col"], Literal["test_col"]])

    # Test literal
    lit_col = F.lit(1)
    assert_type(lit_col, Column[Literal["lit"], Literal["expr"]])

    # Test that column operations preserve input type
    col2 = F.col("name")
    upper_col = F.upper(col2)
    assert_type(upper_col, Column[Literal["name"], Literal["expr"]])

    # Test multiple column operations
    col3 = F.col("age")
    sum_col = F.sum(col3)
    assert_type(sum_col, Column[Literal["age"], Literal["expr"]])

    # Test when condition
    when_col = F.when(F.col("flag") == F.lit(True), 1).otherwise(0)
    assert_type(when_col, Column[Literal["flag", 'lit'], Literal["expr"]])


def test_string_operations() -> None:
    # Test string functions
    str_col = F.col("text")
    concat = F.concat(str_col, F.lit(" suffix"))
    assert_type(concat, Column[Literal["text", "lit"], Literal["expr"]])

    # Test regexp
    regex = F.regexp_replace(str_col, "pattern", "replacement")
    assert_type(regex, Column[Literal["text"], Literal["expr"]])


def test_math_operations() -> None:
    # Test math functions
    num_col = F.col("number")
    abs_col = F.abs(num_col)
    assert_type(abs_col, Column[Literal["number"], Literal["expr"]])

    # Test round
    round_col = F.round(num_col, 2)
    assert_type(round_col, Column[Literal["number"], Literal["expr"]])


def test_aggregation_operations() -> None:
    # Test aggregation functions
    val_col = F.col("value")
    avg_col = F.avg(val_col)
    assert_type(avg_col, Column[Literal["value"], Literal["expr"]])

    # Test count
    count_col = F.count(val_col)
    assert_type(count_col, Column[Literal['lit'], Literal['expr']])


def test_boolean_operations() -> None:
    # Test negation operator
    bool_col = F.col("flag")
    negated = ~bool_col
    assert_type(negated, Column[Literal["flag"], Literal["flag"]])

    # Test combined boolean operations
    combined = ~(bool_col & F.col("other_flag"))
    assert_type(combined, Column[Literal["flag", "other_flag"], Literal["expr"]])


def test_regexp_operations() -> None:
    # Test regexp_like
    text_col = F.col("text")
    pattern = "^[A-Z].*"
    
    # Basic regexp_like
    matches = F.regexp_like(text_col, pattern)
    assert_type(matches, Column[Literal["text"], Literal["expr"]])
    
    # regexp_like with case sensitivity flag
    
    # Combining regexp_like with other operations
    combined = ~F.regexp_like(text_col, pattern)
    assert_type(combined, Column[Literal["text"], Literal["expr"]])


if TYPE_CHECKING:
    from pyspark.sql.dataframe import DataFrame
    from pyspark.sql.session import SparkSession

    def test_dataframe_pipeline_types(spark: SparkSession) -> None:
        data = [("John", 30, 5000.0, "IT")]
        df = spark.createDataFrame(data, ("name", "age", "salary", "dept"))
        assert_type(
            df,
            DataFrame[Literal["name", "age", "salary", "dept"]],
        )

        # Select and rename columns
        df2 = df.select(
            F.col("name"),
            F.col("age"),
            F.col("salary").alias("annual_salary"),
            F.lit("2024").alias("year"),
        )
        assert_type(df2, DataFrame[Literal["name", "age", "annual_salary", "year"]])

        # Add calculated columns
        df3 = (
            df2.withColumn("salary_monthly", F.col("annual_salary") / 12)
            .withColumn("age_next_year", F.col("age") + 1)
            .withColumn("name_upper", F.upper(F.col("name")))
        )
        assert_type(
            df3,
            DataFrame[
                Literal[
                    "name",
                    "age",
                    "annual_salary",
                    "year",
                    "salary_monthly",
                    "age_next_year",
                    "name_upper",
                ]
            ],
        )

        # Filter data
        df4 = df3.filter(
            (F.col("salary_monthly") > 400) & F.col("name_upper").startswith("A")
        )
        assert_type(
            df4,
            DataFrame[
                Literal[
                    "name",
                    "age",
                    "annual_salary",
                    "year",
                    "salary_monthly",
                    "age_next_year",
                    "name_upper",
                ]
            ],
        )

        # Aggregate operations
        summary = df4.groupBy("year").agg(
            F.count("*").alias("employee_count"),
            F.avg(F.col("salary_monthly")).alias("avg_salary"),
            F.collect_list(F.col("name_upper")).alias("employees"),
        )
        assert_type(
            summary,
            DataFrame[Literal["year", "employee_count", "avg_salary", "employees"]],
        )

        # Window functions
        from pyspark.sql import Window

        window_spec = Window.partitionBy("year").orderBy(F.col("salary_monthly").desc())
        df5 = df4.withColumn("salary_rank", F.rank().over(window_spec)).withColumn(
            "running_total", F.sum("salary_monthly").over(window_spec)
        )
        assert_type(
            df5,
            DataFrame[
                Literal[
                    "name",
                    "age",
                    "annual_salary",
                    "year",
                    "salary_monthly",
                    "age_next_year",
                    "name_upper",
                    "salary_rank",
                    "running_total",
                ]
            ],
        )
