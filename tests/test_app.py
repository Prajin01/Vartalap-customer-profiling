"""Smoke test: the Streamlit app renders every tab without raising (skips if streamlit is missing)."""
from pathlib import Path

import pytest

testing = pytest.importorskip("streamlit.testing.v1")

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def test_app_renders_without_errors(con):   # `con` fixture skips when the warehouse is missing
    exists = con.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'marts' "
                         "AND table_name = 'customer_profile_base'").fetchone()[0]
    if not exists:
        pytest.skip("run `python -m src.profile_engine` first")
    at = testing.AppTest.from_file(str(APP), default_timeout=180).run()
    assert not at.exception, at.exception
