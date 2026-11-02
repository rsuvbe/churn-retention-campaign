"""Smoke test: the Streamlit app renders and scores a customer without errors."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (ROOT / "models" / "catboost_churn.cbm").exists(),
                    reason="run notebooks/02_modeling.ipynb first")
def test_app_renders():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=180).run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "Churn probability" in labels and "Customers to contact" in labels
