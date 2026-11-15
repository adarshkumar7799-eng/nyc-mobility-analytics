"""Gold layer functions: aggregations and business-ready tables."""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def transform_gold_revenue_by_date(
    spark: SparkSession,
    source_table: str,
    target_table: str,
) -> int:
    """Aggregate silver trips into one row of revenue and trips per pickup date.

    Groups by the date portion of tpep_pickup_datetime, sums total_amount
    cast to DECIMAL(12,2) rounded once outside the sum, overwrites
    target_table, and returns the number of rows written.
    """
    silver_df = spark.table(source_table)

    revenue_by_date_df = (
        silver_df
        .groupBy(F.to_date(F.col("tpep_pickup_datetime")).alias("pickup_date"))
        .agg(
            F.round(
                F.sum(F.col("total_amount").cast("DECIMAL(12,2)")), 2
            ).alias("total_revenue"),
            F.count("*").alias("trip_count"),
        )
        .orderBy(F.col("pickup_date").asc())
    )

    (
        revenue_by_date_df.write
        .format("delta")
        .mode("overwrite")
        .saveAsTable(target_table)
    )

    row_count: int = spark.table(target_table).count()

    return row_count


def transform_gold_trips_by_borough(
    spark: SparkSession,
    source_table: str,
    target_table: str,
) -> int:
    """Aggregate silver trips into one row of trips and revenue per pickup borough.

    Groups by pickup_borough keeping null and unknown keys as their own
    groups, sums total_amount cast to DECIMAL(12,2) rounded once outside
    the sum, overwrites target_table, and returns the number of rows written.
    """
    silver_df = spark.table(source_table)

    trips_by_borough_df = (
        silver_df
        .groupBy(F.col("pickup_borough"))
        .agg(
            F.count("*").alias("trip_count"),
            F.round(
                F.sum(F.col("total_amount").cast("DECIMAL(12,2)")), 2
            ).alias("total_revenue"),
        )
        .orderBy(F.col("trip_count").desc())
    )

    (
        trips_by_borough_df.write
        .format("delta")
        .mode("overwrite")
        .saveAsTable(target_table)
    )

    row_count: int = spark.table(target_table).count()

    return row_count


def transform_gold_trips_by_hour_weekend(
    spark: SparkSession,
    source_table: str,
    target_table: str,
) -> int:
    """Aggregate silver trips into one row of trips per pickup hour and weekend flag.

    Groups by pickup_hour and is_weekend keeping all key combinations,
    overwrites target_table, and returns the number of rows written.
    """
    silver_df = spark.table(source_table)

    trips_by_hour_weekend_df = (
        silver_df
        .groupBy(F.col("pickup_hour"), F.col("is_weekend"))
        .agg(F.count("*").alias("trip_count"))
        .orderBy(F.col("pickup_hour").asc(), F.col("is_weekend").asc())
    )

    (
        trips_by_hour_weekend_df.write
        .format("delta")
        .mode("overwrite")
        .saveAsTable(target_table)
    )

    row_count: int = spark.table(target_table).count()

    return row_count
