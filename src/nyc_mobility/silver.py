"""Silver layer functions: data quality filters and enrichment."""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def transform_silver_trips(
    spark: SparkSession,
    source_table: str,
    lookup_table: str,
    target_table: str,
) -> int:
    """Build the silver trips table from bronze trips and the zone lookup.

    Applies data quality filters, joins the zone lookup twice for pickup
    and dropoff enrichment, adds derived duration and calendar columns,
    and overwrites target_table. Returns the number of rows written.
    """
    trips_df = spark.table(source_table)
    zones_df = spark.table(lookup_table)

    filtered_df = trips_df.filter(
        (F.col("fare_amount") > 0)
        & (F.col("trip_distance") > 0)
        & (F.col("passenger_count") > 0)
        & (F.col("passenger_count") <= 6)
        & F.col("tpep_pickup_datetime").isNotNull()
        & F.col("tpep_dropoff_datetime").isNotNull()
        & (F.col("tpep_dropoff_datetime") > F.col("tpep_pickup_datetime"))
    )

    # Join the zone lookup twice with distinct aliases so the pickup and
    # dropoff copies do not collide on column names.
    pu_zones = zones_df.alias("pu")
    do_zones = zones_df.alias("do")

    enriched_df = (
        filtered_df
        .join(
            pu_zones,
            F.col("PULocationID") == F.col("pu.location_id"),
            "left",
        )
        .join(
            do_zones,
            F.col("DOLocationID") == F.col("do.location_id"),
            "left",
        )
        .select(
            *[F.col(c) for c in trips_df.columns],
            F.col("pu.Zone").alias("pickup_zone"),
            F.col("pu.Borough").alias("pickup_borough"),
            F.col("do.Zone").alias("dropoff_zone"),
            F.col("do.Borough").alias("dropoff_borough"),
        )
    )

    silver_df = (
        enriched_df
        .withColumn(
            "trip_duration_minutes",
            (
                F.unix_timestamp(F.col("tpep_dropoff_datetime"))
                - F.unix_timestamp(F.col("tpep_pickup_datetime"))
            )
            / 60.0,
        )
        .withColumn("pickup_hour", F.hour(F.col("tpep_pickup_datetime")))
        .withColumn(
            "pickup_day_of_week",
            F.dayofweek(F.col("tpep_pickup_datetime")),
        )
        .withColumn("is_weekend", F.col("pickup_day_of_week").isin(1, 7))
    )

    (
        silver_df.write
        .format("delta")
        .mode("overwrite")
        .saveAsTable(target_table)
    )

    row_count: int = spark.table(target_table).count()

    return row_count
