"""Shared pytest fixtures for the NYC Mobility test suite."""

import uuid
from pathlib import Path

import pytest
from pyspark.sql import SparkSession

TEST_ARTIFACTS_ROOT = Path("/Volumes/nyc_mobility/bronze/test_artifacts")


@pytest.fixture(scope="session")
def spark() -> SparkSession:
    """Return the workspace's existing Spark session.

    Reuses the compute-attached session (Spark Connect) instead of
    building a new local one.
    """
    return SparkSession.builder.getOrCreate()


@pytest.fixture
def tmp_volume() -> Path:
    """Return a unique directory path under the bronze test_artifacts Volume.

    The short uuid suffix ensures repeated and concurrent test
    runs never write into the same directory.
    """
    path = TEST_ARTIFACTS_ROOT / f"test_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    return path
