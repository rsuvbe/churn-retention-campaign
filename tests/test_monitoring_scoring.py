"""Tests for drift monitoring, calibrator selection and batch-scoring input handling."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from src import modeling as M
from src.data import load_raw, prepare_for_scoring
from src.monitoring import psi, psi_categorical, status

ROOT = Path(__file__).resolve().parents[1]


def test_psi_zero_for_same_distribution_and_large_for_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 20_000)
    assert psi(ref, rng.normal(0, 1, 20_000)) < 0.01
    assert psi(ref, rng.normal(1, 1, 20_000)) > 0.25


def test_psi_categorical_and_status():
    assert psi_categorical(list("aabb"), list("aabb")) == pytest.approx(0, abs=1e-9)
    assert psi_categorical(list("aaab"), list("abbb")) > 0.25
    assert status(0.05) == "stable" and status(0.15) == "investigate" and status(0.3) == "ALERT"


def test_select_calibrator_prefers_fix_for_miscalibrated_scores():
    rng = np.random.default_rng(1)
    true_p = rng.uniform(0, 1, 5000)
    y = rng.binomial(1, true_p)
    overconfident = true_p ** 0.3          # systematically too high
    best, table = M.select_calibrator(overconfident, y, seed=0)
    assert best in ("platt", "isotonic")
    assert table.loc[best, ("brier", "mean")] < table.loc["none", ("brier", "mean")]


def test_prepare_for_scoring_accepts_raw_columns_without_target():
    raw = load_raw().drop(columns=["Attrition_Flag"]).head(5)
    out = prepare_for_scoring(raw)
    assert "churned" not in out and "avg_trans_amt" in out


def test_prepare_for_scoring_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        prepare_for_scoring(pd.DataFrame({"CLIENTNUM": [1]}))


@pytest.mark.skipif(not (ROOT / "models" / "catboost_churn.cbm").exists(),
                    reason="run notebooks/02_modeling.ipynb first")
def test_score_cli_end_to_end(tmp_path):
    from src.score import main
    src_csv = tmp_path / "in.csv"
    load_raw().drop(columns=["Attrition_Flag"]).head(200).to_csv(src_csv, index=False)
    out_csv = tmp_path / "out.csv"
    main(["--input", str(src_csv), "--output", str(out_csv)])
    scored = pd.read_csv(out_csv)
    assert len(scored) == 200
    assert scored.churn_probability.between(0, 1).all()
    assert (scored.contact == (scored.expected_profit > 0)).all()
