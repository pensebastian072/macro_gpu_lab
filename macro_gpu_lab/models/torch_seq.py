"""GPU forward-return model — small, heavily regularized MLP on the RTX 3050.

Deliberately trained on the SAME pooled feature matrix, SAME expanding
walk-forward, and SAME non-overlapping portfolio-PnL construction as the RF
baseline (models/baseline_rf.py) so the scorecard is an honest head-to-head:
only the learner changes. Features are standardized with TRAIN-ONLY statistics
(no leakage) because a net is scale-sensitive.

Kept small on purpose — a multi-year daily panel is not a big dataset, so the net
is dropout + weight-decay + early-stopping regularized. It runs on CUDA when
available (that is where the GPU earns its keep, and more so in the GPU-parallel
validation of backtest.py) and degrades to CPU otherwise. Same gate: promote only
if every horizon clears; else SHADOW.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .. import config
from ..gpu import get_device, set_seed
from ..validate import evaluate_gate, walk_forward_splits
from .baseline_rf import _dense_matrix

N_TRIALS_DEFLATION = len(config.HORIZONS_DAYS)


def _build_mlp(in_dim: int):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(in_dim, config.TORCH_HIDDEN),
        nn.ReLU(),
        nn.Dropout(config.TORCH_DROPOUT),
        nn.Linear(config.TORCH_HIDDEN, config.TORCH_HIDDEN),
        nn.ReLU(),
        nn.Dropout(config.TORCH_DROPOUT),
        nn.Linear(config.TORCH_HIDDEN, 1),
    )


def _fit_predict(Xtr, ytr, Xte, device):
    """Train the MLP on (Xtr, ytr); return p_up for Xte. Early-stops on a
    chronological tail of the training rows."""
    import torch
    import torch.nn as nn

    set_seed(config.SEED)
    # train-only standardization
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd[sd == 0] = 1.0
    Xtr_n = (Xtr - mu) / sd
    Xte_n = (Xte - mu) / sd

    # chronological validation tail for early stopping
    n = len(Xtr_n)
    cut = max(1, int(n * 0.85))
    xt = torch.tensor(Xtr_n[:cut], dtype=torch.float32, device=device)
    yt = torch.tensor(ytr[:cut], dtype=torch.float32, device=device).unsqueeze(1)
    xv = torch.tensor(Xtr_n[cut:], dtype=torch.float32, device=device)
    yv = torch.tensor(ytr[cut:], dtype=torch.float32, device=device).unsqueeze(1)

    # class balance -> pos_weight
    pos = float(yt.sum().item())
    neg = float(yt.numel() - pos)
    pos_weight = torch.tensor([neg / pos if pos > 0 else 1.0], device=device)

    model = _build_mlp(Xtr_n.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.TORCH_LR,
                           weight_decay=config.TORCH_WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val = float("inf")
    best_state = None
    patience = 0
    for _ in range(config.TORCH_MAX_EPOCHS):
        model.train()
        opt.zero_grad()
        out = model(xt)
        loss = loss_fn(out, yt)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            if xv.shape[0] > 0:
                vloss = float(loss_fn(model(xv), yv).item())
            else:
                vloss = float(loss.item())
        if vloss < best_val - 1e-4:
            best_val = vloss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= config.TORCH_PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        xe = torch.tensor(Xte_n, dtype=torch.float32, device=device)
        proba = torch.sigmoid(model(xe)).squeeze(1).cpu().numpy()
    return proba


def oos_portfolio_pnls(feat: pd.DataFrame, fwd_ret: pd.Series, horizon: int, device=None) -> list[float]:
    device = device or get_device()
    X, _, idx = _dense_matrix(feat)
    dates = idx.get_level_values("date")
    fr = fwd_ret.reindex(idx).to_numpy()
    y = (fr > 0).astype(int).astype(float)

    uniq = np.array(sorted(pd.unique(dates)))
    date_pos = {d: i for i, d in enumerate(uniq)}
    row_datepos = np.array([date_pos[d] for d in dates])
    n_dates = len(uniq)

    pnls: list[float] = []
    for tr_d, te_d in walk_forward_splits(n_dates, config.N_WALK_FORWARD_FOLDS, horizon):
        tr_lo, tr_hi = tr_d[0], tr_d[-1]
        tr_mask = (row_datepos >= tr_lo) & (row_datepos <= tr_hi) & ~np.isnan(fr)
        if np.unique(y[tr_mask]).size < 2:
            continue
        rebal = sorted(te_d.tolist())[::horizon]
        # predict once over all rebalance rows for this fold
        te_mask = np.isin(row_datepos, rebal) & ~np.isnan(fr)
        if not te_mask.any():
            continue
        proba_all = _fit_predict(X[tr_mask], y[tr_mask], X[te_mask], device)
        te_datepos = row_datepos[te_mask]
        te_fr = fr[te_mask]
        for dp in rebal:
            sel = te_datepos == dp
            if not sel.any():
                continue
            pos = np.where(proba_all[sel] > 0.5, 1.0, -1.0)
            pnls.append(float(np.mean(pos * te_fr[sel])))
    return pnls


def train_and_validate(feat: pd.DataFrame, labels: pd.DataFrame) -> dict:
    n_rows = len(feat)
    if n_rows < config.MIN_ROWS_FOR_TRAIN:
        return {"model": "torch_mlp", "status": "skipped", "promoted": False,
                "n_rows": n_rows,
                "reason": f"need >={config.MIN_ROWS_FOR_TRAIN} rows, have {n_rows}"}
    device = get_device()
    horizons: dict[str, dict] = {}
    all_pass = True
    for h in config.HORIZONS_DAYS:
        col = f"fwd_ret_{h}d"
        fwd = labels[col].reindex(feat.index)
        valid = fwd.notna()
        pnls = oos_portfolio_pnls(feat[valid], fwd[valid], h, device=device)
        gate = evaluate_gate(pnls, n_trials=N_TRIALS_DEFLATION)
        from ..backtest import confidence
        gate["bootstrap"] = confidence(pnls)
        horizons[f"{h}d"] = gate
        all_pass = all_pass and gate["passes"]

    return {
        "model": "torch_mlp",
        "status": "trained",
        "promoted": bool(all_pass),
        "device": str(device),
        "n_rows": n_rows,
        "horizons": horizons,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
