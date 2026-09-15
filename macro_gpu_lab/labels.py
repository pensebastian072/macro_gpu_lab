"""Forward-return labels — the only forward-looking quantity in the pipeline.

For each asset close series, the h-day forward return is close[t+h]/close[t]-1
(NaN in the last h rows — no peeking). Direction label = sign of that return.
Non-overlapping trade construction happens later in the OOS backtest (stepping h
days at a time) so PnL observations don't share forward windows — the honest-DSR
requirement mirrored from copper_brain/model.py::_oos_pnls.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def forward_returns(panel: pd.DataFrame, horizons=None) -> dict[int, pd.DataFrame]:
    """Per-horizon wide forward-return frames (index=date, cols=asset)."""
    horizons = horizons or config.HORIZONS_DAYS
    out: dict[int, pd.DataFrame] = {}
    for h in horizons:
        out[h] = panel.shift(-h) / panel - 1.0
    return out


def long_labels(panel: pd.DataFrame, horizons=None) -> pd.DataFrame:
    """Long forward-return frame indexed by (date, asset) with one column per
    horizon: fwd_ret_{h}d. Aligns with features.build_features's index."""
    horizons = horizons or config.HORIZONS_DAYS
    fr = forward_returns(panel, horizons)
    cols = {}
    for h in horizons:
        stacked = fr[h].stack()   # pandas 2.x drops NA groups; realigned on use
        stacked.index = stacked.index.set_names(["date", "asset"])
        cols[f"fwd_ret_{h}d"] = stacked
    return pd.DataFrame(cols)


def direction(fwd_ret: pd.Series) -> np.ndarray:
    """Binary up/down label (1 if forward return > 0)."""
    return (fwd_ret > 0).astype(int).to_numpy()
