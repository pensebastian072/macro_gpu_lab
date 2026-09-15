"""Stage 7 — relationship/regime SHIFT detector.

Instead of predicting each asset's direction (Stages 3-4), this predicts when the
market's *structure* is about to break: a forward realized-vol spike on SPY,
which is what accompanies absorption spikes, correlation blow-ups and lead-lag
breakdowns. It trades a defensive overlay so the signal is gate-able:

    market-wide relationship features (backward-looking)
      -> label: shift = fwd SPY vol over next h > SHIFT_VOL_MULT x trailing median
      -> RF detector, expanding walk-forward (gap = h)
      -> overlay PnL: position SPY -1 (defensive) if shift predicted else +1,
         rebalanced every h days (non-overlapping)
      -> validate.evaluate_gate (same PBO/DSR gate) + GPU bootstrap confidence
      -> promote only if EVERY horizon clears; else SHADOW (Plan rule).

Also reports detection precision/recall vs the shift base rate so a "no edge"
verdict is legible.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .. import config
from ..features import (_to_common_grid, log_returns, absorption_series,
                        avg_pairwise_corr_series, _shared_spread_z)
from ..validate import evaluate_gate, walk_forward_splits
from ..backtest import confidence

N_TRIALS_DEFLATION = len(config.HORIZONS_DAYS)


# ────────────────────────────────────────────────────────────────
# Market-wide feature frame (one row per date, backward-looking)
# ────────────────────────────────────────────────────────────────

def build_market_features(panel: pd.DataFrame) -> pd.DataFrame:
    grid = _to_common_grid(panel)
    rets = log_returns(grid)
    f = pd.DataFrame(index=grid.index)
    if "VIX" in grid:
        f["vix_level"] = grid["VIX"]
        f["vix_chg20"] = grid["VIX"].diff(20)
    absn = absorption_series(rets, config.CORR_WINDOW, config.ABSORPTION_K_DIVISOR)
    f["absorption"] = absn
    f["absorption_chg5"] = absn.diff(5)
    f["absorption_regime"] = absn - absn.rolling(config.Z_WINDOW, min_periods=60).median()
    f["avg_pairwise_corr"] = avg_pairwise_corr_series(rets, config.CORR_WINDOW)
    f["avg_corr_chg20"] = f["avg_pairwise_corr"].diff(20)
    f["credit_stress"] = _shared_spread_z(grid, "HYG", "LQD")
    f["curve_signal"] = _shared_spread_z(grid, "TIP", "TLT")
    tgt = config.SHIFT_TARGET
    if tgt in grid:
        spy = grid[tgt]
        spy_r = spy.pct_change()
        f["spy_ret20"] = spy.pct_change(20)
        f["spy_vol20"] = spy_r.rolling(20, min_periods=20).std()
        f["spy_px_vs_sma50"] = spy / spy.rolling(50, min_periods=30).mean() - 1
    # cross-sectional breadth: dispersion of 20d returns across assets
    disp = grid.pct_change(20).std(axis=1)
    f["dispersion20"] = disp
    return f.dropna()


def shift_label(panel: pd.DataFrame, horizon: int) -> pd.Series:
    """1 if SPY forward realized vol over next `horizon` days exceeds
    SHIFT_VOL_MULT x its trailing-median realized vol (backward-looking base)."""
    grid = _to_common_grid(panel)
    spy_r = grid[config.SHIFT_TARGET].pct_change()
    # forward realized vol: std over [t+1, t+h]
    fwd_vol = spy_r.rolling(horizon, min_periods=horizon).std().shift(-horizon)
    # trailing baseline: median of realized vol over the median window (<= t)
    realized = spy_r.rolling(horizon, min_periods=horizon).std()
    base = realized.rolling(config.SHIFT_VOL_MEDIAN_WINDOW,
                            min_periods=config.SHIFT_VOL_MEDIAN_WINDOW // 2).median()
    return (fwd_vol > config.SHIFT_VOL_MULT * base).astype(float).where(
        fwd_vol.notna() & base.notna())


def spy_fwd_ret(panel: pd.DataFrame, horizon: int) -> pd.Series:
    grid = _to_common_grid(panel)
    spy = grid[config.SHIFT_TARGET]
    return spy.shift(-horizon) / spy - 1.0


# ────────────────────────────────────────────────────────────────
# Walk-forward detector + defensive-overlay PnL
# ────────────────────────────────────────────────────────────────

def _oos(feat: pd.DataFrame, label: pd.Series, fwd: pd.Series, horizon: int):
    from sklearn.ensemble import RandomForestClassifier

    df = feat.join(label.rename("y")).join(fwd.rename("fwd")).dropna(subset=["y"])
    X = df[feat.columns].to_numpy()
    y = df["y"].to_numpy().astype(int)
    fr = df["fwd"].to_numpy()
    n = len(df)

    pnls: list[float] = []
    tp = fp = fn = tn = 0
    for tr, te in walk_forward_splits(n, config.N_WALK_FORWARD_FOLDS, horizon):
        if np.unique(y[tr]).size < 2:
            continue
        m = RandomForestClassifier(**config.RF_KWARGS)
        m.fit(X[tr], y[tr])
        pred = (m.predict_proba(X[te])[:, 1] > 0.5).astype(int)
        # confusion (all test rows)
        for pv, yv in zip(pred, y[te]):
            if pv and yv:
                tp += 1
            elif pv and not yv:
                fp += 1
            elif not pv and yv:
                fn += 1
            else:
                tn += 1
        # non-overlapping overlay PnL: rebalance every `horizon`-th test row
        for j in range(0, len(te), horizon):
            r = fr[te[j]]
            if np.isnan(r):
                continue
            pos = -1.0 if pred[j] else 1.0     # defensive when a shift is predicted
            pnls.append(float(pos * r))
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    base_rate = (tp + fn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) else None
    detect = {"precision": precision, "recall": recall, "base_rate": base_rate,
              "tp": tp, "fp": fp, "fn": fn, "tn": tn}
    return pnls, detect


def train_and_validate(panel: pd.DataFrame) -> dict:
    feat = build_market_features(panel)
    if len(feat) < config.MIN_ROWS_FOR_TRAIN:
        return {"model": "shift_detector", "status": "skipped", "promoted": False,
                "n_rows": len(feat)}
    horizons: dict[str, dict] = {}
    all_pass = True
    for h in config.HORIZONS_DAYS:
        label = shift_label(panel, h).reindex(feat.index)
        fwd = spy_fwd_ret(panel, h).reindex(feat.index)
        pnls, detect = _oos(feat, label, fwd, h)
        gate = evaluate_gate(pnls, n_trials=N_TRIALS_DEFLATION)
        gate["bootstrap"] = confidence(pnls)
        gate["detection"] = detect
        horizons[f"{h}d"] = gate
        all_pass = all_pass and gate["passes"]
    return {
        "model": "shift_detector",
        "status": "trained",
        "promoted": bool(all_pass),
        "n_rows": len(feat),
        "n_features": feat.shape[1],
        "horizons": horizons,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }


# ────────────────────────────────────────────────────────────────
# Reframe: SURPRISE spikes — eruptions from a CALM state
# ────────────────────────────────────────────────────────────────
# Vol persistence trivially calls spikes when vol is already high. The only
# spikes it CANNOT call are those erupting from calm — and that is exactly where
# cross-asset relationships (credit stress, rising absorption, correlation
# instability, curve) might fire before realized vol moves. So we train on all
# days but GRADE + TRADE only on calm test days; the bar to beat is the calm-state
# base rate P(spike | calm), which naive persistence scores ~0 recall on.

def _calm_and_base(panel: pd.DataFrame, horizon: int):
    """Return (calm_mask, realized_vol, base_vol) on the common grid."""
    grid = _to_common_grid(panel)
    spy_r = grid[config.SHIFT_TARGET].pct_change()
    realized = spy_r.rolling(horizon, min_periods=horizon).std()
    base = realized.rolling(config.SHIFT_VOL_MEDIAN_WINDOW,
                            min_periods=config.SHIFT_VOL_MEDIAN_WINDOW // 2).median()
    calm = (realized <= base)   # vol NOT already elevated
    return calm, realized, base


def _trail_daily_vol(panel: pd.DataFrame) -> pd.Series:
    grid = _to_common_grid(panel)
    return grid[config.SHIFT_TARGET].pct_change().rolling(
        config.TRAIL_VOL_WINDOW, min_periods=config.TRAIL_VOL_WINDOW).std()


def _oos_surprise(feat, label, spy_fwd, premium, calm, horizon):
    """Walk-forward: train RF on all rows, grade + trade only on CALM test rows.

    Computes several payoffs on the same predictions so they compare apples-to-
    apples (SPY-only — the edge is SPY-concentrated; basket pooling diluted it):
      - short_spy_overlay : pos -1 if eruption predicted else +1  (SPY directional)
      - long_vol_selective: buy a straddle ONLY on predicted-eruption days
                            payoff = |SPY move| - premium
      - long_vol_overlay  : long straddle when predicted, short straddle otherwise
    Plus a straddle_all baseline (buy every calm day) so we can see if the model's
    SELECTION beats buying vol indiscriminately.
    """
    from sklearn.ensemble import RandomForestClassifier

    df = (feat.join(label.rename("y")).join(spy_fwd.rename("fwd"))
          .join(premium.rename("prem")).join(calm.rename("calm"))
          .dropna(subset=["y", "fwd", "prem"]))
    X = df[feat.columns].to_numpy()
    y = df["y"].to_numpy().astype(int)
    fr = df["fwd"].to_numpy()
    prem = df["prem"].to_numpy()
    cm = df["calm"].to_numpy().astype(bool)
    n = len(df)

    strat = {k: [] for k in ("short_spy_overlay", "long_vol_selective",
                             "long_vol_overlay", "straddle_all")}
    tp = fp = fn = tn = 0
    for tr, te in walk_forward_splits(n, config.N_WALK_FORWARD_FOLDS, horizon):
        if np.unique(y[tr]).size < 2:
            continue
        m = RandomForestClassifier(**config.RF_KWARGS)
        m.fit(X[tr], y[tr])
        pred_all = (m.predict_proba(X[te])[:, 1] > 0.5).astype(int)
        calm_local = [i for i in range(len(te)) if cm[te[i]]]
        for i in calm_local:
            pv, yv = pred_all[i], y[te[i]]
            if pv and yv:
                tp += 1
            elif pv and not yv:
                fp += 1
            elif not pv and yv:
                fn += 1
            else:
                tn += 1
        for k in range(0, len(calm_local), horizon):
            i = calm_local[k]
            idx = te[i]
            r = fr[idx]
            straddle = abs(r) - prem[idx]     # long-vol payoff (either direction)
            pred = pred_all[i]
            strat["short_spy_overlay"].append((-1.0 if pred else 1.0) * r)
            strat["straddle_all"].append(straddle)
            if pred:
                strat["long_vol_selective"].append(straddle)
            strat["long_vol_overlay"].append(straddle if pred else -straddle)
    tot = tp + fp + fn + tn
    detect = {
        "precision": tp / (tp + fp) if (tp + fp) else None,
        "recall": tp / (tp + fn) if (tp + fn) else None,
        "base_rate_calm": (tp + fn) / tot if tot else None,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
    return strat, detect


def train_and_validate_surprise(panel: pd.DataFrame) -> dict:
    feat = build_market_features(panel)
    if len(feat) < config.MIN_ROWS_FOR_TRAIN:
        return {"model": "surprise_shift", "status": "skipped", "promoted": False,
                "n_rows": len(feat)}
    trail_vol = _trail_daily_vol(panel).reindex(feat.index)
    horizons: dict[str, dict] = {}
    any_pass = False
    for h in config.HORIZONS_DAYS:
        calm, _, _ = _calm_and_base(panel, h)
        calm = calm.reindex(feat.index)
        spike = shift_label(panel, h).reindex(feat.index)
        label = (spike * calm.astype(float)).where(calm)
        spy_fwd = spy_fwd_ret(panel, h).reindex(feat.index)
        premium = config.STRADDLE_PREMIUM_MULT * trail_vol * math.sqrt(h)
        strat, detect = _oos_surprise(feat, label, spy_fwd, premium, calm, h)

        p, br = detect["precision"], detect["base_rate_calm"]
        strategies = {}
        for name, pnls in strat.items():
            g = evaluate_gate(pnls, n_trials=len(strat) * N_TRIALS_DEFLATION)
            g["bootstrap"] = confidence(pnls)
            strategies[name] = g
            any_pass = any_pass or g["passes"]
        horizons[f"{h}d"] = {
            "detection": detect,
            "lift_vs_calm_base": round(p / br, 3) if (p and br) else None,
            "strategies": strategies,
        }
    return {
        "model": "surprise_shift",
        "status": "trained",
        "promoted": bool(any_pass),
        "n_rows": len(feat),
        "n_features": feat.shape[1],
        "horizons": horizons,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    import argparse
    import json
    from .. import data as data_mod
    ap = argparse.ArgumentParser()
    ap.add_argument("--surprise", action="store_true",
                    help="run the calm-state surprise-spike reframe")
    args = ap.parse_args()

    panel = data_mod.load_latest_panel()
    if panel is None or panel.empty:
        print("no panel — run macro_gpu_lab.data first")
        return

    if args.surprise:
        v = train_and_validate_surprise(panel)
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        p = config.SCORECARDS_DIR / f"macro_gpu_surprise_{stamp}.json"
        p.write_text(json.dumps(v, indent=2, default=str))
        print(f"\n=== surprise_shift (eruption from calm) : {v['status']} : "
              f"{'PROMOTED' if v.get('promoted') else 'SHADOW'} ===")
        for h, hd in (v.get("horizons") or {}).items():
            d = hd["detection"]
            print(f"  [{h}] detect precision={d['precision']:.3f} "
                  f"recall={d['recall']:.3f} calm_base={d['base_rate_calm']:.3f} "
                  f"LIFT={hd['lift_vs_calm_base']}x")
            for name, g in hd["strategies"].items():
                dsr = (g.get("deflated_sharpe") or {}).get("ratio")
                print(f"       {name:20s} pass={str(g['passes']):5s} "
                      f"PF={g['profit_factor']}  DSR={dsr}  n={g['n_trades']}")
        print(f"scorecard -> {p}")
        return

    v = train_and_validate(panel)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    p = config.SCORECARDS_DIR / f"macro_gpu_shift_{stamp}.json"
    p.write_text(json.dumps(v, indent=2, default=str))
    print(f"\n=== shift_detector : {v['status']} : "
          f"{'PROMOTED' if v.get('promoted') else 'SHADOW'} ===")
    for h, g in (v.get("horizons") or {}).items():
        dsr = (g.get("deflated_sharpe") or {}).get("ratio")
        d = g["detection"]
        print(f"  {h}: pass={g['passes']}  DSR={dsr}  pbo={g['pbo']}  "
              f"PF={g['profit_factor']}  n={g['n_trades']}")
        print(f"       detect precision={d['precision']} recall={d['recall']} "
              f"base_rate={d['base_rate']}")
    print(f"scorecard -> {p}")


if __name__ == "__main__":
    main()
