"""Sanity checks on synthetic data where the right answer is known.

Run with:  python -m pytest -q
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import metrics  # noqa: E402

rng = np.random.default_rng(42)
N = 50_000


def calibrated_sample():
    p = rng.uniform(0.01, 0.99, N)
    y = (rng.uniform(size=N) < p).astype(float)
    return p, y


def test_perfectly_calibrated_market():
    p, y = calibrated_sample()
    fit = metrics.calibration_fit(p, y)
    assert abs(fit["slope"] - 1) < 0.05
    assert abs(fit["intercept"]) < 0.05
    assert metrics.murphy_decomposition(p, y)["reliability"] < 0.001
    for row in metrics.calibration_table(p, y):
        assert row["ci_low"] - 0.01 <= row["mean_price"] <= row["ci_high"] + 0.01


def test_overconfident_market_has_slope_below_one():
    # True probability is q, but the market quotes a more extreme price.
    q = rng.uniform(0.05, 0.95, N)
    y = (rng.uniform(size=N) < q).astype(float)
    p = 1 / (1 + np.exp(-2 * np.log(q / (1 - q))))
    assert metrics.calibration_fit(p, y)["slope"] < 0.6


def test_decomposition_close_to_brier():
    p, y = calibrated_sample()
    d = metrics.murphy_decomposition(p, y)
    approx = d["reliability"] - d["resolution"] + d["uncertainty"]
    # Price bins leave a small within-bin term, so allow a tolerance.
    assert abs(approx - metrics.brier(p, y)) < 0.01


def test_scores_on_simple_cases():
    p = np.array([1.0, 0.0])
    y = np.array([1.0, 0.0])
    assert metrics.brier(p, y) == 0
    assert metrics.brier(np.array([0.5]), np.array([1.0])) == 0.25
    lo, hi = metrics.wilson(50, 100)
    assert lo < 0.5 < hi


def test_favourite_folding():
    p = np.array([0.9, 0.1, 0.8])
    y = np.array([1.0, 0.0, 0.0])
    rows = metrics.favourite_table(p, y)
    total = sum(r["n"] for r in rows)
    wins = sum(r["yes_rate"] * r["n"] for r in rows)
    assert total == 3 and round(wins) == 2
