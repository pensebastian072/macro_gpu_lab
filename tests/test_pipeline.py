"""Core invariants: no feature leakage, honest labels, fail-safe flag, sane gate."""
import numpy as np
import pandas as pd
import pytest

from macro_gpu_lab import config
from macro_gpu_lab.features import build_features, _to_common_grid
from macro_gpu_lab.labels import long_labels, forward_returns
from macro_gpu_lab.validate import evaluate_gate
from macro_gpu_lab import publish


def _synth_panel(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    cols = {}
    for a in ("SPY", "TLT", "GOLD", "DXY", "VIX", "QQQ"):
        r = rng.normal(0, 0.01, n)
        cols[a] = 100 * np.exp(np.cumsum(r))
    return pd.DataFrame(cols, index=idx)


def test_features_are_backward_looking():
    """A feature row at date t must not depend on any data after t: rebuilding on
    the panel truncated at t must reproduce the exact same row."""
    panel = _synth_panel()
    grid = _to_common_grid(panel)
    full = build_features(grid)
    cut = grid.index[300]
    trunc = build_features(grid.loc[:cut])
    for asset in ("TLT", "GOLD"):
        a = full.loc[(cut, asset)].drop(labels=["force"])
        b = trunc.loc[(cut, asset)].drop(labels=["force"])
        assert np.allclose(a.astype(float).values, b.astype(float).values, atol=1e-9), \
            f"leakage: {asset} feature row changed when future rows were removed"


def test_forward_returns_are_correct_and_truncated():
    panel = _synth_panel()
    grid = _to_common_grid(panel)
    fr = forward_returns(grid, [5])[5]
    px = grid["SPY"]
    t = grid.index[100]
    expected = px.shift(-5).loc[t] / px.loc[t] - 1
    assert np.isclose(fr["SPY"].loc[t], expected)
    # last h rows have no forward label
    assert fr["SPY"].iloc[-5:].isna().all()


def test_long_labels_align_to_features():
    panel = _synth_panel()
    grid = _to_common_grid(panel)
    feat = build_features(grid)
    labels = long_labels(grid).reindex(feat.index)
    assert list(labels.columns) == [f"fwd_ret_{h}d" for h in config.HORIZONS_DAYS]
    assert len(labels) == len(feat)


def test_gate_rejects_pure_noise():
    rng = np.random.default_rng(1)
    pnls = rng.normal(0, 1, 500).tolist()   # zero-mean noise
    g = evaluate_gate(pnls, n_trials=2)
    assert g["passes"] is False


def test_gate_handles_tiny_input():
    g = evaluate_gate([0.1, -0.2, 0.05], n_trials=2)
    assert g["passes"] is False   # too few for DSR/PBO -> cannot pass


def test_flag_failsafe_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "FLAG_PATH", tmp_path / "nope.json")
    s = publish.read_state()
    assert s["stale"] is True
    assert s["assets"] == {}
