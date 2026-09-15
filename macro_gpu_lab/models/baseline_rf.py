"""RandomForest baseline — the reference bar every GPU model must beat.

Mirrors copper_brain/copper_brain/model.py (RF_KWARGS, expanding walk-forward,
non-overlapping OOS, evaluate_gate) but on the pooled cross-asset panel instead
of a single series.

Trade construction (honest DSR): the model predicts p_up per (date, asset). We
rebalance every `horizon` days so forward windows never overlap; on each rebalance
date we take a cross-sectional portfolio — position each asset by sign(p_up-0.5)
and record the MEAN pos*forward_return across assets as one PnL observation. That
yields a clean, non-overlapping strategy PnL time series (one point per rebalance)
rather than pseudo-replicated per-asset trades. A model promotes only if EVERY
horizon clears the gate; else it stays SHADOW.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from ..validate import evaluate_gate, walk_forward_splits

N_TRIALS_DEFLATION = len(config.HORIZONS_DAYS)


def _dense_matrix(feat: pd.DataFrame):
    """(numeric ndarray, feature_names) from the long feature frame."""
    from ..features import feature_matrix
    fm = feature_matrix(feat)
    fm = fm.astype(float)
    return fm.values, list(fm.columns), fm.index


def oos_portfolio_pnls(feat: pd.DataFrame, fwd_ret: pd.Series, horizon: int) -> list[float]:
    """Expanding walk-forward over unique dates -> non-overlapping portfolio PnL."""
    from sklearn.ensemble import RandomForestClassifier

    X, _, idx = _dense_matrix(feat)
    dates = idx.get_level_values("date")
    fr = fwd_ret.reindex(idx).to_numpy()
    y = (fr > 0).astype(int)

    uniq = np.array(sorted(pd.unique(dates)))
    date_pos = {d: i for i, d in enumerate(uniq)}
    row_datepos = np.array([date_pos[d] for d in dates])
    n_dates = len(uniq)

    pnls: list[float] = []
    for tr_d, te_d in walk_forward_splits(n_dates, config.N_WALK_FORWARD_FOLDS, horizon):
        tr_lo, tr_hi = tr_d[0], tr_d[-1]
        te_set = set(te_d.tolist())
        tr_mask = (row_datepos >= tr_lo) & (row_datepos <= tr_hi)
        # drop train rows with NaN label
        tr_mask &= ~np.isnan(fr)
        if np.unique(y[tr_mask]).size < 2:
            continue
        m = RandomForestClassifier(**config.RF_KWARGS)
        m.fit(X[tr_mask], y[tr_mask])

        # rebalance every `horizon`-th test date -> non-overlapping forward windows
        rebal = sorted(te_set)[::horizon]
        for dp in rebal:
            rmask = (row_datepos == dp) & ~np.isnan(fr)
            if not rmask.any():
                continue
            proba = m.predict_proba(X[rmask])[:, 1]
            pos = np.where(proba > 0.5, 1.0, -1.0)
            pnls.append(float(np.mean(pos * fr[rmask])))
    return pnls


def latest_signals(feat: pd.DataFrame, labels: pd.DataFrame) -> dict:
    """Fit RF per horizon on ALL labelled rows; return latest p_up per asset.

    Advisory only — the flag-file surface. Shape: {asset: {"5d": p, "21d": p}}.
    """
    from sklearn.ensemble import RandomForestClassifier

    X, _, idx = _dense_matrix(feat)
    assets = idx.get_level_values("asset")
    dates = idx.get_level_values("date")
    out: dict[str, dict] = {}
    for h in config.HORIZONS_DAYS:
        fwd = labels[f"fwd_ret_{h}d"].reindex(idx).to_numpy()
        valid = ~np.isnan(fwd)
        y = (fwd > 0).astype(int)
        if np.unique(y[valid]).size < 2:
            continue
        m = RandomForestClassifier(**config.RF_KWARGS)
        m.fit(X[valid], y[valid])
        # latest row per asset (max date)
        for a in pd.unique(assets):
            amask = assets == a
            if not amask.any():
                continue
            last_pos = np.where(amask)[0][np.argmax(dates[amask])]
            p = float(m.predict_proba(X[last_pos:last_pos + 1])[0, 1])
            out.setdefault(a, {})[f"{h}d"] = round(p, 4)
    return out


def train_and_validate(feat: pd.DataFrame, labels: pd.DataFrame) -> dict:
    """Gate the RF across all horizons. Returns a verdict dict."""
    from datetime import datetime, timezone
    n_rows = len(feat)
    if n_rows < config.MIN_ROWS_FOR_TRAIN:
        return {"model": "rf", "status": "skipped", "promoted": False,
                "n_rows": n_rows,
                "reason": f"need >={config.MIN_ROWS_FOR_TRAIN} rows, have {n_rows}"}

    horizons: dict[str, dict] = {}
    all_pass = True
    for h in config.HORIZONS_DAYS:
        col = f"fwd_ret_{h}d"
        fwd = labels[col].reindex(feat.index)
        valid = fwd.notna()
        pnls = oos_portfolio_pnls(feat[valid], fwd[valid], h)
        gate = evaluate_gate(pnls, n_trials=N_TRIALS_DEFLATION)
        from ..backtest import confidence
        gate["bootstrap"] = confidence(pnls)
        horizons[f"{h}d"] = gate
        all_pass = all_pass and gate["passes"]

    return {
        "model": "rf",
        "status": "trained",
        "promoted": bool(all_pass),
        "n_rows": n_rows,
        "horizons": horizons,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
