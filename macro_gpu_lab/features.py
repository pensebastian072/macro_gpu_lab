"""Cross-asset relationship features — daily, backward-looking only.

Daily-panel versions of the hq-trading-system macro_engine computations: rolling
correlation + beta to the SPY/DXY bases, z-scored divergence spreads, and the
Kritzman absorption ratio (systemic-risk / one-factor detector). Everything here
uses only information available up to and including date t — the no-leakage tests
assert this — so the forward-return labels attached in labels.py are honest.

Output is a LONG feature frame indexed by (date, asset): each row is one asset on
one day with its own momentum/vol features plus the shared macro context and the
relationship features that let a model learn which cross-asset states precede
moves — i.e. the "monetizable differences" we are hunting.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import config


def log_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """Elementwise daily log returns (NaN where a close is missing)."""
    return np.log(panel).diff()


# ────────────────────────────────────────────────────────────────
# Shared market-wide features (same value for every asset on date t)
# ────────────────────────────────────────────────────────────────

def absorption_series(rets: pd.DataFrame, window: int, k_divisor: int) -> pd.Series:
    """Rolling Kritzman absorption ratio: variance share in the top-K
    eigenvectors of the trailing-window correlation matrix. High/rising = a
    one-factor (fragile) market. Backward-looking (uses rows <= t)."""
    idx = rets.index
    out = pd.Series(index=idx, dtype=float)
    vals = rets.values
    for i in range(window, len(idx) + 1):
        w = vals[i - window:i]
        # keep assets with full coverage in this window
        col_ok = ~np.isnan(w).any(axis=0)
        m = w[:, col_ok]
        if m.shape[1] < 3:
            continue
        c = np.corrcoef(m, rowvar=False)
        if not np.all(np.isfinite(c)):
            continue
        eig = np.linalg.eigvalsh(c)
        eig = np.clip(eig, 0, None)[::-1]
        total = eig.sum()
        if total <= 0:
            continue
        k = max(1, math.ceil(m.shape[1] / k_divisor))
        out.iloc[i - 1] = float(eig[:k].sum() / total)
    return out


def avg_pairwise_corr_series(rets: pd.DataFrame, window: int) -> pd.Series:
    """Rolling mean absolute off-diagonal pairwise correlation — a scalar
    'everything moving together' gauge (co-movement precedes fragile regimes).
    Backward-looking (window ends at t)."""
    idx = rets.index
    out = pd.Series(index=idx, dtype=float)
    vals = rets.values
    for i in range(window, len(idx) + 1):
        w = vals[i - window:i]
        col_ok = ~np.isnan(w).any(axis=0)
        m = w[:, col_ok]
        if m.shape[1] < 3:
            continue
        c = np.corrcoef(m, rowvar=False)
        if not np.all(np.isfinite(c)):
            continue
        n = c.shape[0]
        off = (np.abs(c).sum() - n) / (n * (n - 1))   # mean |corr| off-diagonal
        out.iloc[i - 1] = float(off)
    return out


def _shared_spread_z(panel: pd.DataFrame, a: str, b: str) -> pd.Series:
    """z of log(a)-log(b) over Z_WINDOW; 0 if either leg absent."""
    if a not in panel or b not in panel:
        return pd.Series(0.0, index=panel.index)
    s = np.log(panel[a]) - np.log(panel[b])
    mu = s.rolling(config.Z_WINDOW, min_periods=config.Z_WINDOW).mean()
    sd = s.rolling(config.Z_WINDOW, min_periods=config.Z_WINDOW).std()
    return (s - mu) / sd.replace(0.0, np.nan)


def _market_context(panel: pd.DataFrame, rets: pd.DataFrame) -> pd.DataFrame:
    ctx = pd.DataFrame(index=panel.index)
    if "VIX" in panel:
        ctx["vix_level"] = panel["VIX"]
        ctx["vix_chg20"] = panel["VIX"].diff(20)
    else:
        ctx["vix_level"] = 0.0
        ctx["vix_chg20"] = 0.0
    ctx["absorption"] = absorption_series(
        rets, config.CORR_WINDOW, config.ABSORPTION_K_DIVISOR)
    ctx["absorption_chg5"] = ctx["absorption"].diff(5)
    # absorption regime: level vs its own trailing median (rising = fragile)
    ctx["absorption_regime"] = ctx["absorption"] - ctx["absorption"].rolling(
        config.Z_WINDOW, min_periods=60).median()
    # system co-movement
    ctx["avg_pairwise_corr"] = avg_pairwise_corr_series(rets, config.CORR_WINDOW)
    # credit stress (HY vs IG) and real-vs-nominal duration (curve) — shared
    ctx["credit_stress"] = _shared_spread_z(panel, "HYG", "LQD")
    ctx["curve_signal"] = _shared_spread_z(panel, "TIP", "TLT")
    return ctx


# ────────────────────────────────────────────────────────────────
# Relationship features to the SPY/DXY bases
# ────────────────────────────────────────────────────────────────

def _rolling_corr(a: pd.Series, b: pd.Series, w: int) -> pd.Series:
    return a.rolling(w, min_periods=w).corr(b)


def _rolling_beta(a: pd.Series, b: pd.Series, w: int) -> pd.Series:
    cov = a.rolling(w, min_periods=w).cov(b)
    var = b.rolling(w, min_periods=w).var()
    return cov / var.replace(0.0, np.nan)


# ────────────────────────────────────────────────────────────────
# Divergence spreads -> per-asset z features
# ────────────────────────────────────────────────────────────────

def _spread_zscores(panel: pd.DataFrame) -> dict[str, pd.Series]:
    """z of log(a)-log(b) over Z_WINDOW for each configured spread."""
    out: dict[str, pd.Series] = {}
    for name, (a, b) in config.SPREADS.items():
        if a not in panel or b not in panel:
            continue
        s = np.log(panel[a]) - np.log(panel[b])
        mu = s.rolling(config.Z_WINDOW, min_periods=config.Z_WINDOW).mean()
        sd = s.rolling(config.Z_WINDOW, min_periods=config.Z_WINDOW).std()
        out[name] = (s - mu) / sd.replace(0.0, np.nan)
    return out


def _asset_divergence(panel: pd.DataFrame):
    """For each asset: signed z of the first spread it appears in, and the
    max-abs z across all its spreads. Sign convention: + when the asset is the
    'a' (numerator) leg, - when it is the 'b' leg."""
    zs = _spread_zscores(panel)
    signed: dict[str, pd.Series] = {}
    absmax: dict[str, pd.Series] = {}
    for asset in config.ASSETS:
        legs = []
        first_signed = None
        for name, (a, b) in config.SPREADS.items():
            if name not in zs:
                continue
            if asset == a:
                legs.append(zs[name].abs())
                if first_signed is None:
                    first_signed = zs[name]
            elif asset == b:
                legs.append(zs[name].abs())
                if first_signed is None:
                    first_signed = -zs[name]
        if first_signed is not None:
            signed[asset] = first_signed
            absmax[asset] = pd.concat(legs, axis=1).max(axis=1)
    return signed, absmax


# ────────────────────────────────────────────────────────────────
# Assemble the long feature frame
# ────────────────────────────────────────────────────────────────

def _to_common_grid(panel: pd.DataFrame) -> pd.DataFrame:
    """Restrict to a single trading calendar so rolling windows aren't shredded
    by weekend NaNs. Use SPY's dates as the reference grid (equities/credit/
    commodities/DXY all trade it); FX/crypto extra weekend bars are dropped — we
    keep only co-traded days, the same no-LOCF ethos as the macro engine. No
    forward-fill: a genuinely missing bar stays NaN and its row is dropped later."""
    ref = "SPY" if "SPY" in panel else panel.notna().sum().idxmax()
    grid = panel[ref].dropna().index
    return panel.reindex(grid)


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Long feature frame indexed by (date, asset). Backward-looking only."""
    if panel.empty:
        return pd.DataFrame()
    panel = _to_common_grid(panel)
    rets = log_returns(panel)
    ctx = _market_context(panel, rets)
    base_rets = {b: rets[b] for b in config.BETA_BASES if b in rets}
    anchor_rets = {a: rets[a] for a in config.CORR_ANCHORS if a in rets}
    signed_div, absmax_div = _asset_divergence(panel)

    # per-force cohort mean return series (how much an asset moves with its group)
    cohort_ret: dict[str, pd.Series] = {}
    for force in set(config.FORCE.get(a, "other") for a in panel.columns):
        members = [a for a in panel.columns if config.FORCE.get(a, "other") == force]
        if members:
            cohort_ret[force] = rets[members].mean(axis=1)

    frames = []
    for asset in panel.columns:
        px = panel[asset]
        r = rets[asset]
        f = pd.DataFrame(index=panel.index)
        # own momentum / trend / vol (core — rows missing these are dropped)
        f["ret5"] = px.pct_change(5)
        f["ret20"] = px.pct_change(20)
        f["ret60"] = px.pct_change(60)
        f["px_vs_sma20"] = px / px.rolling(20, min_periods=20).mean() - 1
        f["px_vs_sma50"] = px / px.rolling(50, min_periods=30).mean() - 1
        f["px_vs_sma200"] = px / px.rolling(200, min_periods=120).mean() - 1
        f["vol20"] = r.rolling(20, min_periods=20).std()
        core = list(f.columns)
        # regression relationship to the SPY/DXY bases (+ fast/instability/drift)
        for b, br in base_rets.items():
            bl = b.lower()
            slow = _rolling_corr(r, br, config.CORR_WINDOW)
            fast = _rolling_corr(r, br, config.CORR_WINDOW_FAST)
            beta = _rolling_beta(r, br, config.CORR_WINDOW)
            f[f"corr_{bl}"] = slow
            f[f"corr_{bl}_fast"] = fast
            f[f"corr_{bl}_instab"] = (fast - slow).abs()   # relationship instability
            f[f"beta_{bl}"] = beta
            f[f"beta_{bl}_chg"] = beta.diff(config.BETA_CHG_WINDOW)  # sensitivity drift
        # correlation to the extra relationship anchors (gold/tlt/hyg etc.)
        for a, ar in anchor_rets.items():
            if a in config.BETA_BASES or a == asset:
                continue
            f[f"corr_{a.lower()}"] = _rolling_corr(r, ar, config.CORR_WINDOW)
        # lead-lag vs SPY: does the asset's PAST return correlate with SPY now?
        # (asset_r.shift(1) is backward-looking) positive => asset leads SPY.
        if "SPY" in rets:
            spy_r = rets["SPY"]
            f["leadlag_spy"] = (_rolling_corr(r.shift(1), spy_r, config.CORR_WINDOW)
                                - _rolling_corr(r, spy_r, config.CORR_WINDOW))
        # cohesion with own force cohort
        force = config.FORCE.get(asset, "other")
        if force in cohort_ret:
            f["cohort_corr"] = _rolling_corr(r, cohort_ret[force], config.CORR_WINDOW)
        # divergence participation
        f["div_z_signed"] = signed_div.get(asset, pd.Series(0.0, index=panel.index))
        f["div_z_absmax"] = absmax_div.get(asset, pd.Series(0.0, index=panel.index))
        # shared market context
        for c in ctx.columns:
            f[c] = ctx[c]
        # identity
        f["force"] = force
        f["asset"] = asset

        # aux NaN -> 0 (neutral); drop rows missing any CORE feature
        aux = [c for c in f.columns if c not in core + ["force", "asset"]]
        f[aux] = f[aux].fillna(0.0)
        f = f.dropna(subset=core)
        f.index.name = "date"
        f = f.set_index("asset", append=True)
        frames.append(f)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames).sort_index()
    # cross-sectional relative strength: z-score of ret20 across assets each date
    # (per-date, no time leakage). Captures which assets lead/lag the pack.
    g = out.groupby(level="date")["ret20"]
    out["rel_strength20"] = ((out["ret20"] - g.transform("mean"))
                             / g.transform("std").replace(0.0, np.nan)).fillna(0.0)
    # self-anchor corr cols (e.g. corr_tlt on the TLT row) are absent per-frame and
    # become NaN on concat — fill neutral so the model matrix is NaN-free.
    num_cols = [c for c in out.columns if c != "force"]
    out[num_cols] = out[num_cols].fillna(0.0)
    return out


# one-hot the categorical force for the model matrix
def feature_matrix(feat: pd.DataFrame) -> pd.DataFrame:
    """Numeric matrix: drop identity cols, one-hot 'force'."""
    df = feat.copy()
    forces = pd.get_dummies(df.pop("force"), prefix="force")
    df = pd.concat([df, forces.astype(float)], axis=1)
    return df
