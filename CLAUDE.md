# macro_gpu_lab — agent guide

GPU **research bench** over the HQ macro brain (torch cu121 on the RTX 3050). Builds a
multi-year daily panel for the 22-asset macro universe, engineers cross-asset
relationship features, trains RF/torch models behind the canonical overfit gate, and
writes an **advisory shadow flag-file** (`journal/flags/tv_signal.json`). Nothing here
touches a broker or a live hot path.

Current state + next steps: **`HANDOFF.md`** (read it first).

Skills: load `paper-trading-guardrails` before touching export/webhook/flag-file code,
`quant-research-gate` before any model/gate work, `win-quant-env` before
installs/git/PS/Task Scheduler.

## Hard rules

1. Research / advisory / paper only. Never wire to execution or a broker — the
   `MACRO_GPU_WEBHOOK_URL` sink is a notification webhook (Discord/Slack/JSON), NEVER
   a broker endpoint.
2. Models stay SHADOW until the gate clears (PBO < 0.5 AND **Deflated Sharpe ratio >
   1.645** on every horizon). Promotion is an env flip, never automatic. Both current
   models are honestly SHADOW (DSR < 0) — do not soften the gate to "pass" them.
   The bar was **raised from 0.0 to 1.645 on 2026-07-30** with the DSR unit fix:
   once the DSR is computed correctly, `ratio > 0` is a MEDIAN test that best-of-8 pure
   noise clears 44.8% of the time. `config.DEFLATED_SHARPE_MIN` is authoritative.
   **Nothing on this box has ever cleared it.**
3. Flag-file is fail-safe: any HQ reader is off hot path, neutral on stale/missing.
4. A correct gate usually FAILS the first model. Report that honestly.

## Verification commands

Use `.venv\Scripts\python.exe` — bare `python` is the Store stub on this box.

- CUDA smoke test: `.venv\Scripts\python.exe -m macro_gpu_lab.gpu`
- Tests: `.venv\Scripts\python.exe -m pytest`
- Build panel: `.venv\Scripts\python.exe -m macro_gpu_lab.data`
- Train once: `.venv\Scripts\python.exe -m macro_gpu_lab.train --once`
- Shift signal export: `.venv\Scripts\python.exe -m macro_gpu_lab.models.shift_export --post`

## Environment

- venv via `pwsh scripts/setup_venv.ps1` (uv-managed, torch cu121).
- Daily export job is registered **elevated** via
  `scripts\register_daily_export.ps1` (UAC relaunch to change; runs hidden via
  wscript). Webhook URL lives in gitignored `secrets\webhook_url.txt`, never in git.


### GPU numeric mode (`gpu.tune_backend()`, added 2026-09-01)

`get_device()` now enables TF32 on CUDA. Measured on this box's RTX 3050 idle at
full clock, 4096x4096 matmul, median of 3: fp32 1.43 -> TF32 2.62 TFLOPS (**1.84x**).
Only `matmul.allow_tf32` actually changes: torch already defaults
`cudnn.allow_tf32` to True, so conv/RNN layers were always running TF32 --
do **not** read this change as "the conv results moved".

| env var | default | effect |
| --- | --- | --- |
| `GPU_TF32` | on | `0` restores the previous precision setting. NOT bit-exact reproduction -- that also needs `cudnn.benchmark` pinned, `torch.use_deterministic_algorithms(True)` with `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and the same torch/CUDA/driver build. |
| `GPU_MEM_FRACTION` | 0.92 | `0` lifts the per-process VRAM cap. The cap exists so an oversized allocation raises `OutOfMemoryError` instead of silently spilling into system RAM through the Windows driver's sysmem fallback and running 10-50x slower. It caps against **total** VRAM, not free -- with ollama holding ~5.9 of 6 GB the driver still binds first. |

`confidence()` stamps the returned state under `"tuning"`, so a recorded number
says which numeric mode produced it. Keep that stamp: without it an old row and a
new one are indistinguishable in the ledger.

**Do not re-run a battery merely to compare TF32 against non-TF32.**
`registry.log_trial` appends to the ledger and `n_trials` is cumulative, so a
curiosity re-run permanently tightens the deflated-Sharpe threshold for that
family. Check precision on a scratch script, never through the harness.
