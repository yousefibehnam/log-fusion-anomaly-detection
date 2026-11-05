from pathlib import Path

import pytest

DATA = Path(__file__).parent / "data"


@pytest.fixture
def hdfs_2k_sample() -> Path:
    return DATA / "loghub" / "HDFS_2k.log_structured.csv"
