"""Unit tests for the parts of src/ where a silent bug would change conclusions."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from src import ab_testing as AB
from src import business as B
from src import modeling as M
from src.data import LEAKY_PREFIX, load_raw


def test_leaky_columns_are_dropped():
    assert not any(c.startswith(LEAKY_PREFIX) for c in load_raw().columns)


def test_recall_at_top_k():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1])
    s = np.array([.9, .8, .7, .1, .1, .1, .1, .1, .1, .05])
    # top 30% = 3 customers -> catches 2 of 3 churners
    assert M.recall_at_top_k(y, s, 0.3) == pytest.approx(2 / 3)
    assert M.precision_at_top_k(y, s, 0.3) == pytest.approx(2 / 3)


def test_sample_size_matches_textbook_formula():
    n_sm = AB.sample_size_per_arm(0.5, 0.15, 0.05, 0.8)
    n_tb = AB.sample_size_formula(0.5, 0.15, 0.05, 0.8)
    assert abs(n_sm - n_tb) / n_tb < 0.03


def test_mde_inverts_sample_size():
    n = AB.sample_size_per_arm(0.5, 0.15, 0.05, 0.8)
    assert AB.mde_for_sample(0.5, n, 0.05, 0.8) == pytest.approx(0.15, abs=0.005)


def test_aa_false_positive_rate_close_to_alpha():
    rng = np.random.default_rng(0)
    fpr = AB.monte_carlo_power(0.4, 0.4, 500, 20_000, rng, 0.05)
    assert 0.04 < fpr < 0.06


def test_profit_curve_matches_manual_calculation():
    a = B.Assumptions(0.015, 0.15, 2, 150, contact_cost=10, incentive_cost=0,
                      incentive_accept_rate=0, retention_success_rate=0.5)
    y = np.array([1, 0, 1])
    clv = np.array([100.0, 100.0, 300.0])
    score = np.array([0.9, 0.5, 0.1])
    curve = B.profit_curve(y, score, clv, a)
    # contact 1: 0.5*100 - 10 = 40; contact 2: 40 - 10 = 30; contact 3: 30 + 0.5*300 - 10 = 170
    assert curve.profit.tolist() == pytest.approx([0, 40, 30, 170])


def test_customer_value_floor():
    a = B.Assumptions(0.015, 0.15, 2, 150, 15, 40, 0.5, 0.25)
    df = pd.DataFrame({"total_trans_amt": [0, 10_000], "total_revolving_bal": [0, 1000]})
    v = B.customer_value(df, a)
    assert v.iloc[0] == 150
    assert v.iloc[1] == pytest.approx((150 + 150) * 2)
