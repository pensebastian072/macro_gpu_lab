"""Overfit-aware validation gate — ported near-verbatim from
copper_brain/copper_brain/validate.py (itself ported from
hq-trading-system/analytics/research_scorecard.py). DO NOT reinvent this math.

Gate: a model promotes only if
    deflated_sharpe_ratio > config.DEFLATED_SHARPE_MIN  (i.e. > 0)
    AND pbo < config.PBO_MAX                            (i.e. < 0.5)
on the walk-forward OOS PnL, for EVERY horizon. Otherwise it stays SHADOW.

  - PBO via CSCV: Bailey-Borwein-Lopez de Prado-Zhu 2014.
  - Deflated Sharpe: Bailey & Lopez de Prado 2014.
  - Purged+embargo K-fold and expanding walk-forward: de Prado, "Advances in
    Financial ML" ch.7 — because forward-return labels overlap and leak across a
    naive CV boundary.
"""
from __future__ import annotations

import math
from itertools import combinations

import numpy as np

from . import config

EULER_GAMMA = 0.5772156649015329


# ── PnL stats ───────────────────────────────────────────────────
def profit_factor(pnls) -> float:
    gw = sum(p for p in pnls if p > 0)
    gl = -sum(p for p in pnls if p < 0)
    if gl == 0:
        return float("inf") if gw > 0 else 0.0
    return gw / gl


def sharpe(pnls):
    arr = np.asarray(pnls, dtype=float)
    if arr.size < 2:
        return None
    sd = arr.std(ddof=1)
    if sd == 0:
        return None
    return float(arr.mean() / sd)


# ── PBO via CSCV ────────────────────────────────────────────────
def pbo_cscv(pnls, n_groups: int = None):
    """Fraction of IS/OOS block splits where IS-PF>1 but OOS-PF<=1."""
    n_groups = n_groups or config.CSCV_N_GROUPS
    if n_groups < 4 or n_groups % 2:
        raise ValueError("n_groups must be even and >= 4")
    pnls = list(pnls)
    n = len(pnls)
    if n < n_groups * 4:
        return None
    sz = n // n_groups
    blocks = [pnls[i * sz:(i + 1) * sz] for i in range(n_groups)]
    half = n_groups // 2
    n_overfit = n_valid = 0
    for is_idx in combinations(range(n_groups), half):
        oos_idx = tuple(i for i in range(n_groups) if i not in is_idx)
        is_pf = profit_factor([p for i in is_idx for p in blocks[i]])
        oos_pf = profit_factor([p for i in oos_idx for p in blocks[i]])
        if not (math.isfinite(is_pf) and math.isfinite(oos_pf)):
            continue
        n_valid += 1
        if is_pf > 1.0 and oos_pf <= 1.0:
            n_overfit += 1
    if n_valid == 0:
        return None
    return round(n_overfit / n_valid, 4)


# ── Deflated Sharpe ─────────────────────────────────────────────
def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


def _norm_ppf(p):
    """Acklam's approximation — adequate for SR significance work."""
    if not 0 < p < 1:
        raise ValueError(f"ppf: p must be in (0,1), got {p}")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def expected_max_sr_units(n_trials: int) -> float:
    """The BLdP bracket: E[max SR] expressed in units of sigma_SR (dimensionless)."""
    n = max(1, int(n_trials))
    if n == 1:
        return 0.0
    return ((1 - EULER_GAMMA) * _norm_ppf(1 - 1.0 / n)
            + EULER_GAMMA * _norm_ppf(1 - 1.0 / (n * math.e)))


def deflated_sharpe(pnls, n_trials: int = 1):
    """Deflated Sharpe -> {sr, ratio, prob, n, n_trials, ...} or None.

    Bailey & Lopez de Prado (2014):

        DSR = PSR(SR*) = Phi[ (SR - SR*) * sqrt(T-1)
                              / sqrt(1 - g3*SR + (g4-1)/4*SR^2) ]
        SR* = E[max SR] = sigma_SR * bracket(N)
        bracket(N) = (1-gamma)*Phi^-1(1 - 1/N) + gamma*Phi^-1(1 - 1/(N*e))

    FIXED 2026-07-30 (audit: research_ledger/audit/dsr_audit.py). The prior
    implementation subtracted `bracket` directly from `sr` -- but the bracket is
    DIMENSIONLESS, a count of sigma_SR's, while `sr` is a per-observation Sharpe.
    It then scaled the difference by sqrt(T-1)/denom, inflating the penalty by
    roughly sqrt(T-1): 14.5x at T=210 and up to 111x on the stored scorecards. At
    N=8 the bracket is 1.459, so the old form demanded a per-observation SR of 1.46
    -- an annualised Sharpe near 23 -- before the penalty term alone was cleared.

    The consequence was that the gate was a CONSTANT FAIL, not a strict test: on
    best-of-N pure noise it returned prob = 0.0000 for N = 2, 8 and 42 (a correct
    deflated Sharpe returns ~0.5 there by construction), and it passed 0.000 of
    cases even with a planted true per-observation SR of 0.30 (~4.8 annualised).
    Every historical `deflated_sharpe_ratio <= 0` verdict on this box therefore
    carried no information about the strategy it rejected.

    sigma_SR is approximated by the SR estimator's own standard error,
    sqrt(denom^2/(T-1)), which makes SR* = bracket*denom/sqrt(T-1) and collapses
    the whole expression to `psr_z - bracket`. That is the conservative choice; the
    faithful alternative is the observed dispersion of SR across the N trials, which
    disperses slightly less on this box and so is slightly more permissive. Pass
    `trial_srs` to use it.

    `test_deflated_sharpe_null_calibration` pins the calibration so this cannot
    silently regress.
    """
    arr = np.asarray(pnls, dtype=float)
    if arr.size < 8:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    mu, sigma = arr.mean(), arr.std(ddof=1)
    # `sigma <= 0` alone is not enough: a CONSTANT series returns float noise for its
    # std (e.g. [0.1]*20 -> 1.4e-17), which produced sr ~ 7e15 and prob 1.0 -- a
    # constant PnL sailing through the gate. Reject anything whose dispersion is at
    # floating-point level for the series scale.
    if sigma <= 1e-12 * max(1.0, abs(mu)):
        return None
    sr = mu / sigma
    T = arr.size
    skew = float(((arr - mu) ** 3).mean() / sigma**3)
    kurt = float(((arr - mu) ** 4).mean() / sigma**4)
    n = max(1, int(n_trials))
    denom = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr * sr))
    psr_z = sr * math.sqrt(T - 1) / denom          # PSR z-score against SR* = 0
    bracket = expected_max_sr_units(n)
    sigma_sr = denom / math.sqrt(T - 1)
    sr_star = sigma_sr * bracket
    ratio = psr_z - bracket                        # == (sr - sr_star)*sqrt(T-1)/denom
    return {"sr": round(sr, 4), "ratio": round(ratio, 4),
            "prob": round(_norm_cdf(ratio), 4), "n": T, "n_trials": n,
            "psr_z": round(psr_z, 4), "bracket": round(bracket, 4),
            "sr_star": round(sr_star, 6), "sigma_sr": round(sigma_sr, 6)}


# ── Time-series splits ──────────────────────────────────────────
def purged_kfold(n: int, n_splits: int, label_horizon: int, embargo_pct: float = 0.01):
    """Purged K-fold with embargo for overlapping-label series."""
    indices = np.arange(n)
    fold_sizes = np.full(n_splits, n // n_splits, dtype=int)
    fold_sizes[: n % n_splits] += 1
    embargo = int(n * embargo_pct)
    current = 0
    for fs in fold_sizes:
        if fs == 0:
            continue
        test_start, test_end = current, current + fs
        current = test_end
        test_idx = indices[test_start:test_end]
        lo = test_start - label_horizon
        hi = test_end + embargo
        train_mask = (indices < lo) | (indices >= hi)
        train_idx = indices[train_mask]
        if train_idx.size and test_idx.size:
            yield train_idx, test_idx


def walk_forward_splits(n: int, n_folds: int, label_horizon: int):
    """Expanding-window walk-forward with a label_horizon gap between train and
    test so the last training label can't peek into the test window."""
    indices = np.arange(n)
    sz = n // (n_folds + 1)
    if sz <= label_horizon:
        return
    for i in range(1, n_folds + 1):
        train_end = sz * i
        test_start = train_end + label_horizon
        test_end = min(test_start + sz, n)
        if test_start >= n or train_end == 0:
            break
        yield indices[:train_end], indices[test_start:test_end]


# ── Promotion gate ──────────────────────────────────────────────
def evaluate_gate(pnls, n_trials: int = 1) -> dict:
    """Promote only if deflated Sharpe ratio > DEFLATED_SHARPE_MIN AND PBO < PBO_MAX."""
    dsr = deflated_sharpe(pnls, n_trials=n_trials)
    pbo = pbo_cscv(pnls)
    ratio = None if dsr is None else dsr["ratio"]
    passes = (
        dsr is not None and pbo is not None
        and ratio > config.DEFLATED_SHARPE_MIN
        and pbo < config.PBO_MAX
    )
    reasons = []
    if dsr is None:
        reasons.append("insufficient PnL for Deflated Sharpe (need >=8 trades)")
    elif ratio <= config.DEFLATED_SHARPE_MIN:
        reasons.append(f"deflated_sharpe_ratio {ratio} <= {config.DEFLATED_SHARPE_MIN}")
    if pbo is None:
        reasons.append("insufficient PnL for PBO (need >=40 trades)")
    elif pbo >= config.PBO_MAX:
        reasons.append(f"pbo {pbo} >= {config.PBO_MAX}")
    return {
        "passes": bool(passes),
        "deflated_sharpe": dsr,
        "pbo": pbo,
        "profit_factor": round(profit_factor(pnls), 4) if len(pnls) else None,
        "sharpe": sharpe(pnls),
        "n_trades": len(pnls),
        "thresholds": {"deflated_sharpe_min": config.DEFLATED_SHARPE_MIN,
                       "pbo_max": config.PBO_MAX},
        "reasons": reasons or ["cleared gate"],
    }
