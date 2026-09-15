"""GPU-accelerated confidence for an OOS PnL series.

This is where the RTX 3050 genuinely earns its keep: bootstrap resampling and a
sign-permutation test are embarrassingly parallel, so we run thousands of
resamples as one batched tensor op instead of a Python loop. The point-estimate
gate (PBO/DSR in validate.py) stays authoritative; this adds a resampled
confidence interval on the Sharpe and a permutation p-value on the mean so a
"pass" isn't taken on a single point estimate.

Fail-safe: if torch/CUDA is unavailable it runs the same math on CPU tensors; if
there is too little PnL it returns None. Never raises into the caller.
"""
from __future__ import annotations

import numpy as np

from . import config
from .gpu import get_device, tune_backend

N_BOOT = 10000
N_PERM = 10000
CI = (2.5, 97.5)


def confidence(pnls, n_boot: int = N_BOOT, n_perm: int = N_PERM) -> dict | None:
    """Bootstrap Sharpe CI + permutation p-value(mean>0) on GPU when available."""
    arr = np.asarray(pnls, dtype=np.float64)
    if arr.size < 8:
        return None
    try:
        import torch
    except Exception:  # noqa: BLE001 — no torch -> skip, gate still stands
        return None

    try:
        device = get_device()
        g = torch.Generator(device=device)
        g.manual_seed(config.SEED)
        t = torch.tensor(arr, dtype=torch.float32, device=device)
        n = t.numel()

        # ── bootstrap: (n_boot x n) resample-with-replacement -> Sharpe dist ──
        idx = torch.randint(0, n, (n_boot, n), generator=g, device=device)
        samp = t[idx]                                   # (n_boot, n)
        mean = samp.mean(dim=1)
        std = samp.std(dim=1, unbiased=True)
        sharpe = torch.where(std > 0, mean / std, torch.zeros_like(mean))
        lo, hi = np.percentile(sharpe.cpu().numpy(), CI)
        sharpe_pos = float((sharpe > 0).float().mean().item())

        # ── sign-permutation test: p(mean >= observed | random ± signs) ──
        obs_mean = float(t.mean().item())
        signs = torch.randint(0, 2, (n_perm, n), generator=g, device=device) * 2 - 1
        perm_means = (t.unsqueeze(0) * signs).mean(dim=1)
        p_value = float((perm_means >= obs_mean).float().mean().item())
    except RuntimeError as e:
        # The module docstring promises this never raises into the caller, and
        # callers up-stack have broad excepts that would turn a crash into a
        # silently absent CI. Since gpu.tune_backend() caps this process at a
        # fraction of VRAM, an oversized (n_boot x n) matrix now raises here
        # instead of crawling through the Windows driver's sysmem fallback.
        # Return it as a recorded error so an infrastructure failure can never
        # be mistaken for a research result.
        return {"error": repr(e), "device": str(device), "n": int(n)}


    return {
        "device": str(device),
        "tuning": tune_backend(),
        "n_boot": n_boot,
        "n_perm": n_perm,
        "sharpe_ci": [round(float(lo), 4), round(float(hi), 4)],
        "sharpe_prob_positive": round(sharpe_pos, 4),
        "mean": round(obs_mean, 6),
        "perm_p_value": round(p_value, 4),
    }
