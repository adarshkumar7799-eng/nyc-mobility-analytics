"""Unit tests for the Gold aggregation functions in nyc_mobility.gold."""

import sys
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Union

# pytest runs from tests/, so make the project's src/ package importable
# the same way the notebooks do (sys.path.append of the src root).
SRC_ROOT = str(Path(__file__).resolve().parent.parent / "src")
if SRC_ROOT not in sys.path:
    sys.path.append(SRC_ROOT)

from nyc_mobility import gold
from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampNTZType,
)

# Exactly the columns the Gold functions read; pickup timestamps are
# TIMESTAMP_NTZ and rows carry naive (timezone-less) datetime values.
SILVER_SCHEMA = StructType(
    [
        StructField("tpep_pickup_datetime", TimestampNTZType(), True),
        StructField("total_amount", DoubleType(), True),
        StructField("pickup_borough", StringType(), True),
        StructField("pickup_hour", IntegerType(), True),
        StructField("is_weekend", BooleanType(), True),
    ]
)

SILVER_DATA = [
    # Three Manhattan trips on Monday 2024-01-15.
    (datetime(2024, 1, 15, 8, 0), 10.00, "Manhattan", 8, False),
    (datetime(2024, 1, 15, 14, 0), 20.00, "Manhattan", 14, False),
    (datetime(2024, 1, 15, 20, 0), 30.00, "Manhattan", 20, False),
    # Two Brooklyn trips on Saturday 2024-01-20.
    (datetime(2024, 1, 20, 10, 0), 40.00, "Brooklyn", 10, True),
    (datetime(2024, 1, 20, 15, 0), 15.00, "Brooklyn", 15, True),
    # One Manhattan trip on Saturday 2024-01-20.
    (datetime(2024, 1, 20, 8, 0), 25.00, "Manhattan", 8, True),
]

TOTAL_TRIPS = 6
TOTAL_REVENUE = Decimal("140.00")


def assert_single_matching_row(df: DataFrame, condition: Column) -> None:
    """Assert exactly one row of the DataFrame satisfies the condition."""
    assert df.filter(condition).count() == 1


def assert_column_sum(df: DataFrame, column: str, expected: Union[int, Decimal]) -> None:
    """Assert the sum of one column over all rows equals the expected value."""
    summed = df.agg(F.sum(column).alias("total"))
    assert summed.filter(F.col("total") == expected).count() == 1


def test_gold_aggregations(spark: SparkSession, tmp_volume: Path) -> None:
    """Run all three gold transforms on one shared silver source.

    Asserts each aggregation's measures, the returned row counts, and the
    conservation of trip totals and revenue across the written tables.
    """
    source_table = f"nyc_mobility.bronze.test_gold_{tmp_volume.name}_source"
    revenue_table = f"nyc_mobility.bronze.test_gold_{tmp_volume.name}_revenue"
    borough_table = f"nyc_mobility.bronze.test_gold_{tmp_volume.name}_borough"
    hour_weekend_table = f"nyc_mobility.bronze.test_gold_{tmp_volume.name}_hour_weekend"

    try:
        silver_df = spark.createDataFrame(SILVER_DATA, SILVER_SCHEMA)
        silver_df.write.mode("overwrite").saveAsTable(source_table)

        # Revenue by date.
        rows_revenue = gold.transform_gold_revenue_by_date(
            spark=spark,
            source_table=source_table,
            target_table=revenue_table,
        )
        assert rows_revenue == 2

        revenue_df = spark.table(revenue_table)
        assert revenue_df.count() == 2
        assert_single_matching_row(
            revenue_df,
            (F.col("pickup_date") == date(2024, 1, 15))
            & (F.col("total_revenue") == Decimal("60.00"))
            & (F.col("trip_count") == 3),
        )
        assert_single_matching_row(
            revenue_df,
            (F.col("pickup_date") == date(2024, 1, 20))
            & (F.col("total_revenue") == Decimal("80.00"))
            & (F.col("trip_count") == 3),
        )
        assert_column_sum(revenue_df, "trip_count", TOTAL_TRIPS)
        assert_column_sum(revenue_df, "total_revenue", TOTAL_REVENUE)

        # Trips by borough.
        rows_borough = gold.transform_gold_trips_by_borough(
            spark=spark,
            source_table=source_table,
            target_table=borough_table,
        )
        assert rows_borough == 2

        borough_df = spark.table(borough_table)
        assert borough_df.count() == 2
        assert_single_matching_row(
            borough_df,
            (F.col("pickup_borough") == "Manhattan")
            & (F.col("trip_count") == 4)
            & (F.col("total_revenue") == Decimal("85.00")),
        )
        assert_single_matching_row(
            borough_df,
            (F.col("pickup_borough") == "Brooklyn")
            & (F.col("trip_count") == 2)
            & (F.col("total_revenue") == Decimal("55.00")),
        )
        assert_column_sum(borough_df, "trip_count", TOTAL_TRIPS)
        assert_column_sum(borough_df, "total_revenue", TOTAL_REVENUE)

        # Trips by hour and weekend flag.
        rows_hour_weekend = gold.transform_gold_trips_by_hour_weekend(
            spark=spark,
            source_table=source_table,
            target_table=hour_weekend_table,
        )
        assert rows_hour_weekend == 6

        hour_weekend_df = spark.table(hour_weekend_table)
        assert hour_weekend_df.count() == 6
        assert hour_weekend_df.filter(F.col("trip_count") == 1).count() == 6
        assert_column_sum(hour_weekend_df, "trip_count", TOTAL_TRIPS)
    finally:
        for table_name in (
            source_table,
            revenue_table,
            borough_table,
            hour_weekend_table,
        ):
            spark.sql(f"DROP TABLE IF EXISTS {table_name}")