"""Calibration tests for the overfit gate. These exist because it was once wrong.

Until 2026-07-30 `deflated_sharpe` subtracted the dimensionless BLdP bracket directly
from a per-observation Sharpe and then scaled the difference by sqrt(T-1)/denom,
inflating the penalty ~sqrt(T-1)-fold. The result was a CONSTANT FAIL: prob 0.0000 on
best-of-N pure noise, and 0.000 pass rate even at a planted true per-observation SR of
0.30 (~4.8 annualised). Every `deflated_sharpe_ratio <= 0` verdict produced before that
date carries no information.

The bug was invisible because the n_trials=1 path is correct and every unit test used
it. So the tests below deliberately exercise n_trials > 1, and they test the property
that DEFINES deflation rather than a hard-coded number:

    a deflated Sharpe removes exactly the advantage of having searched N times --
    no more. So the best of N pure-noise trials must score ~0.5, not ~0.0.

This module is imported by nothing; it exists to stop a silent regression. The same
tests live in copper_brain and hq-trading-system, which hold the other two copies of
this math.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from macro_gpu_lab import config
from macro_gpu_lab.validate import (
    deflated_sharpe,
    evaluate_gate,
    expected_max_sr_units,
    pbo_cscv,
)

T = 210


def _best_of_n_noise(rng, n_trials, t=T, planted_sr=0.0):
    trials = rng.standard_normal((n_trials, t)) * 0.01
    if planted_sr:
        trials[0] += planted_sr * 0.01
    srs = trials.mean(axis=1) / trials.std(axis=1, ddof=1)
    return trials[int(np.argmax(srs))]


@pytest.mark.parametrize("n_trials", [2, 8, 42])
def test_deflated_sharpe_null_calibration(n_trials):
    """THE test. Best-of-N on pure noise must average ~0.5, never ~0.0."""
    rng = np.random.default_rng(11)
    probs = [deflated_sharpe(_best_of_n_noise(rng, n_trials), n_trials=n_trials)["prob"]
             for _ in range(400)]
    mean = float(np.mean(probs))
    assert abs(mean - 0.5) < 0.10, (
        f"n_trials={n_trials}: mean DSR probability {mean:.4f} on PURE NOISE. A correct "
        "deflated Sharpe returns ~0.50 here. Near 0.0 means the penalty is inflated "
        "(the pre-2026-07-30 bug); near 1.0 means it is missing."
    )


def test_deflated_sharpe_has_power():
    """A planted, genuinely large edge must pass most of the time."""
    rng = np.random.default_rng(17)
    passes = [deflated_sharpe(_best_of_n_noise(rng, 8, planted_sr=0.30),
                              n_trials=8)["ratio"] > 0
              for _ in range(300)]
    assert float(np.mean(passes)) > 0.90


def test_power_rises_with_the_true_edge():
    rng = np.random.default_rng(23)
    rates = []
    for planted in (0.0, 0.15, 0.30):
        passes = [deflated_sharpe(_best_of_n_noise(rng, 8, planted_sr=planted),
                                  n_trials=8)["ratio"] > 0
                  for _ in range(200)]
        rates.append(float(np.mean(passes)))
    assert rates == sorted(rates), rates


def test_bracket_is_dimensionless_and_monotonic():
    assert expected_max_sr_units(1) == 0.0
    vals = [expected_max_sr_units(n) for n in (2, 8, 42, 132)]
    assert vals == sorted(vals)
    assert all(0 < v < 4 for v in vals)


def test_penalty_is_not_scaled_by_sqrt_T():
    """The signature of the old bug: the penalty must NOT grow with sample size.

    With the same per-observation Sharpe, a longer sample should make the ratio MORE
    favourable (more evidence), never dramatically less.
    """
    rng = np.random.default_rng(5)
    base = rng.standard_normal(100) * 0.01
    # centre first, THEN add the drift, so the per-observation Sharpe is genuinely
    # +0.2 rather than whatever the random draw's own mean happened to be.
    short = base - base.mean() + 0.002
    long = np.concatenate([short] * 8)          # same moments, 8x the observations
    r_short = deflated_sharpe(short, n_trials=8)["ratio"]
    r_long = deflated_sharpe(long, n_trials=8)["ratio"]
    assert r_long > r_short, (r_short, r_long)


def test_single_trial_is_the_psr_z_score():
    rng = np.random.default_rng(7)
    pnl = rng.standard_normal(300) * 0.01 + 0.0004
    d = deflated_sharpe(pnl, n_trials=1)
    assert d["bracket"] == 0.0
    assert d["ratio"] == pytest.approx(d["psr_z"], abs=1e-9)


def test_ratio_equals_psr_z_minus_bracket():
    rng = np.random.default_rng(9)
    pnl = rng.standard_normal(250) * 0.01 + 0.0006
    for n in (2, 8, 42):
        d = deflated_sharpe(pnl, n_trials=n)
        # all three fields are rounded to 4dp on the way out, so the identity holds
        # only to that precision -- tightening this is testing the rounding, not the math
        assert d["ratio"] == pytest.approx(d["psr_z"] - d["bracket"], abs=2e-4)
        # every field is rounded for the scorecard, and the bracket multiplies that
        # rounding error, so this identity is checked relatively rather than absolutely
        assert d["sr_star"] == pytest.approx(d["sigma_sr"] * d["bracket"], rel=1e-4)


def test_degenerate_inputs_return_none():
    assert deflated_sharpe([0.1] * 4, n_trials=2) is None          # too few
    # A CONSTANT series has std == 0 mathematically but ~1.4e-17 in floating point,
    # which used to yield sr ~ 7e15 and prob 1.0 -- a constant PnL passing the gate.
    assert deflated_sharpe([0.1] * 20, n_trials=2) is None
    assert deflated_sharpe([0.0] * 20, n_trials=2) is None
    assert deflated_sharpe([0.1] * 19 + [float("nan")], n_trials=2) is None
    assert deflated_sharpe([0.1] * 19 + [float("inf")], n_trials=2) is None


def test_gate_threshold_is_the_significance_bar():
    """Raised 0.0 -> 1.645 on 2026-07-30: ratio>0 is a median test, not a test."""
    assert config.DEFLATED_SHARPE_MIN == pytest.approx(1.645)
    assert config.PBO_MAX == 0.5


def test_gate_rejects_noise_at_the_raised_bar():
    """Pure noise must almost never clear the promotion gate."""
    rng = np.random.default_rng(29)
    passes = 0
    for _ in range(200):
        pnl = _best_of_n_noise(rng, 8)
        g = evaluate_gate(pnl, n_trials=8)
        passes += bool(g["passes"])
    assert passes / 200 < 0.05, f"{passes}/200 noise draws cleared the gate"


def test_pbo_is_unchanged_by_the_dsr_fix():
    """The fix touched deflated_sharpe only -- PBO must behave as before."""
    rng = np.random.default_rng(31)
    pnl = rng.standard_normal(400) * 0.01
    p = pbo_cscv(pnl)
    assert p is None or 0.0 <= p <= 1.0
