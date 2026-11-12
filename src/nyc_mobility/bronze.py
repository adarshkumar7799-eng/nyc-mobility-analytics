"""Bronze layer functions: raw ingestion of NYC Yellow Taxi data."""

import uuid

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def ingest_yellow_taxi_to_bronze(
    spark: SparkSession,
    source_path: str,
    target_table: str,
) -> int:
    """Ingest raw Yellow Taxi Parquet into a bronze Delta table.

    Reads the Parquet file at source_path, stamps every row with ingestion
    metadata (timestamp, source file, run id), and overwrites target_table.
    Returns the number of rows written.
    """
    run_id: str = str(uuid.uuid4())

    bronze_df = (
        spark.read.parquet(source_path)
        .withColumn("ingestion_timestamp", F.current_timestamp())
        .withColumn("source_file", F.col("_metadata.file_path"))
        .withColumn("pipeline_run_id", F.lit(run_id))
    )

    row_count: int = bronze_df.count()

    (
        bronze_df.write
        .format("delta")
        .mode("overwrite")
        .saveAsTable(target_table)
    )

    return row_count