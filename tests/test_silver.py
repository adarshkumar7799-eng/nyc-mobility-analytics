"""Unit tests for nyc_mobility.silver.transform_silver_trips."""

import sys
from datetime import datetime
from pathlib import Path

# pytest runs from tests/, so make the project's src/ package importable
# the same way the notebooks do (sys.path.append of the src root).
SRC_ROOT = str(Path(__file__).resolve().parent.parent / "src")
if SRC_ROOT not in sys.path:
    sys.path.append(SRC_ROOT)

from nyc_mobility import silver
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampNTZType,
)

# Matches the nyc_mobility.bronze.trips schema; the two trip timestamps
# are TIMESTAMP_NTZ and rows carry naive (timezone-less) datetimes.
TRIPS_SCHEMA = StructType(
    [
        StructField("VendorID", IntegerType(), True),
        StructField("tpep_pickup_datetime", TimestampNTZType(), True),
        StructField("tpep_dropoff_datetime", TimestampNTZType(), True),
        StructField("passenger_count", LongType(), True),
        StructField("trip_distance", DoubleType(), True),
        StructField("RatecodeID", LongType(), True),
        StructField("store_and_fwd_flag", StringType(), True),
        StructField("PULocationID", IntegerType(), True),
        StructField("DOLocationID", IntegerType(), True),
        StructField("payment_type", LongType(), True),
        StructField("fare_amount", DoubleType(), True),
        StructField("extra", DoubleType(), True),
        StructField("mta_tax", DoubleType(), True),
        StructField("tip_amount", DoubleType(), True),
        StructField("tolls_amount", DoubleType(), True),
        StructField("improvement_surcharge", DoubleType(), True),
        StructField("total_amount", DoubleType(), True),
        StructField("congestion_surcharge", DoubleType(), True),
        StructField("Airport_fee", DoubleType(), True),
    ]
)

ZONES_SCHEMA = StructType(
    [
        StructField("LocationID", IntegerType(), True),
        StructField("Borough", StringType(), True),
        StructField("Zone", StringType(), True),
        StructField("service_zone", StringType(), True),
    ]
)

TRIPS_DATA = [
    # --- 5 valid rows that pass every quality filter ---
    # Saturday pickup: is_weekend must come out true.
    (1, datetime(2024, 1, 6, 10, 0), datetime(2024, 1, 6, 10, 30), 1, 2.5, 1, "N", 1, 2, 1, 15.5, 0.5, 0.5, 2.0, 0.0, 1.0, 19.5, 2.5, 0.0),
    # Sunday pickup: is_weekend must come out true.
    (1, datetime(2024, 1, 7, 11, 0), datetime(2024, 1, 7, 11, 45), 3, 5.0, 1, "N", 2, 1, 1, 25.0, 0.5, 0.5, 3.0, 0.0, 1.0, 30.0, 2.5, 0.0),
    # Weekday (Monday) pickup.
    (1, datetime(2024, 1, 8, 8, 0), datetime(2024, 1, 8, 8, 20), 2, 1.8, 1, "N", 1, 1, 1, 12.0, 0.5, 0.5, 1.5, 0.0, 1.0, 15.5, 2.5, 0.0),
    # Weekday (Tuesday) pickup, 5 passengers (upper bound of the range).
    (1, datetime(2024, 1, 9, 14, 0), datetime(2024, 1, 9, 15, 10), 5, 6.4, 1, "N", 2, 2, 1, 30.0, 0.5, 0.5, 4.0, 0.0, 1.0, 36.0, 2.5, 0.0),
    # Weekday (Wednesday) pickup.
    (1, datetime(2024, 1, 10, 9, 0), datetime(2024, 1, 10, 9, 15), 4, 1.2, 1, "N", 1, 2, 1, 9.75, 0.5, 0.5, 1.0, 0.0, 1.0, 12.75, 2.5, 0.0),
    # --- 5 invalid rows, one per violated quality filter ---
    # Violates fare_amount > 0: negative fare.
    (1, datetime(2024, 1, 6, 12, 0), datetime(2024, 1, 6, 12, 30), 1, 2.0, 1, "N", 1, 2, 1, -5.0, 0.5, 0.5, 0.0, 0.0, 1.0, -3.0, 2.5, 0.0),
    # Violates trip_distance > 0: zero distance.
    (1, datetime(2024, 1, 6, 12, 30), datetime(2024, 1, 6, 13, 0), 1, 0.0, 1, "N", 1, 2, 1, 10.0, 0.5, 0.5, 0.0, 0.0, 1.0, 12.0, 2.5, 0.0),
    # Violates passenger_count > 0: zero passengers.
    (1, datetime(2024, 1, 8, 10, 0), datetime(2024, 1, 8, 10, 30), 0, 2.0, 1, "N", 1, 2, 1, 10.0, 0.5, 0.5, 0.0, 0.0, 1.0, 12.0, 2.5, 0.0),
    # Violates passenger_count <= 6: eight passengers.
    (1, datetime(2024, 1, 8, 10, 30), datetime(2024, 1, 8, 11, 0), 8, 2.0, 1, "N", 1, 2, 1, 10.0, 0.5, 0.5, 0.0, 0.0, 1.0, 12.0, 2.5, 0.0),
    # Violates dropoff > pickup: dropoff 30 minutes before pickup.
    (1, datetime(2024, 1, 8, 11, 0), datetime(2024, 1, 8, 10, 30), 1, 2.0, 1, "N", 1, 2, 1, 10.0, 0.5, 0.5, 0.0, 0.0, 1.0, 12.0, 2.5, 0.0),
]

ZONES_DATA = [
    (1, "Manhattan", "Times Square", "Yellow Zone"),
    (2, "Brooklyn", "Williamsburg", "Boro Zone"),
]

ENRICHMENT_COLUMNS = {"pickup_zone", "pickup_borough", "dropoff_zone", "dropoff_borough"}

DERIVED_COLUMNS = {
    "trip_duration_minutes",
    "pickup_hour",
    "pickup_day_of_week",
    "is_weekend",
}


def test_transform_silver_trips_filters_and_enriches(
    spark: SparkSession,
    tmp_volume: Path,
) -> None:
    """Run transform_silver_trips on synthetic bronze inputs and check output.

    Builds 10 mock trip rows (5 valid, 5 invalid) and 2 zone rows, runs the
    silver transformation against them, and asserts only the valid rows
    survive, carrying the expected enrichment and derived columns.
    """
    trips_table = f"nyc_mobility.bronze.test_trips_{tmp_volume.name}"
    zones_table = f"nyc_mobility.bronze.test_zones_{tmp_volume.name}"
    target_table = f"nyc_mobility.bronze.test_silver_{tmp_volume.name}"

    try:
        trips_df = spark.createDataFrame(TRIPS_DATA, TRIPS_SCHEMA)
        zones_df = spark.createDataFrame(ZONES_DATA, ZONES_SCHEMA)

        trips_df.write.mode("overwrite").saveAsTable(trips_table)
        zones_df.write.mode("overwrite").saveAsTable(zones_table)

        rows_written = silver.transform_silver_trips(
            spark=spark,
            source_table=trips_table,
            lookup_table=zones_table,
            target_table=target_table,
        )

        # Only the 5 valid rows should survive the quality filters.
        assert rows_written == 5

        silver_df = spark.table(target_table)
        assert silver_df.count() == 5

        written_columns = set(silver_df.columns)
        assert ENRICHMENT_COLUMNS <= written_columns
        assert DERIVED_COLUMNS <= written_columns

        # The valid rows include Saturday 2024-01-06 and Sunday 2024-01-07
        # pickups, so at least one output row must be flagged as weekend.
        weekend_rows = silver_df.filter(F.col("is_weekend")).count()
        assert weekend_rows >= 1
    finally:
        for table_name in (trips_table, zones_table, target_table):
            spark.sql(f"DROP TABLE IF EXISTS {table_name}")