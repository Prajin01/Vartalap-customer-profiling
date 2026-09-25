import pytest

from src.config import path


@pytest.fixture(scope="session")
def con():
    """Read-only connection to the built warehouse; skips if not built yet."""
    db_file = path("duckdb")
    if not db_file.exists():
        pytest.skip("Warehouse not built. Run: python -m src.ingest && python -m src.build_warehouse")
    import duckdb
    c = duckdb.connect(str(db_file), read_only=True)
    yield c
    c.close()
