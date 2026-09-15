# HANDOFF — macro_gpu_lab + macro brain (for the Codex/Robinhood executor)

Audience: the Codex CLI agent wired to the Robinhood MCP executor, running on the
desktop. This repo is your **data + market-scooping + strategy-research base** —
not an order router. Use it to pull markets, reuse the validated research
framework, and inherit what we already learned (and disproved). HQ Trading System
is a **separate** paper stack — do not touch it or post to its pipeline.

---

## 0. TL;DR

- Repo: `C:\Users\<your-user>\macro_gpu_lab` (git, own GPU venv, RTX 3050 / torch cu121).
- It builds a multi-year daily panel for a 22-asset macro universe, engineers
  cross-asset relationship features, trains models behind a strict overfit gate,
  and emits an advisory signal file.
- **Nothing here is a validated edge.** Every strategy is SHADOW (failed the gate)
  and the one promising signal is regime-fragile (2016+ only). Treat outputs as
  research inputs, not trade instructions. See §5–§6 before trading anything.
- Live signal file: `journal/flags/tv_signal.json`. Refresh: run §4 commands.

---

## 1. What this project is

A GPU research bench over the "macro brain" idea: use cross-asset relationships
(correlations, betas, divergences, absorption, lead-lag) to find monetizable
structure. Built end-to-end with an honest validation gate so it tells the truth
about whether an edge exists. It repeatedly found that apparent edges were not
real — that discipline is the most valuable thing to inherit.

## 2. Repo map

```
macro_gpu_lab/
  config.py            universe (22 assets), horizons, gate thresholds, paths, device
  gpu.py               CUDA device select / seed / smoke test
  data.py              multi-source daily panel (Yahoo primary, Stooq fallback)
  features.py          cross-asset relationship features (backward-looking only)
  labels.py            forward-return labels (5d, 21d), non-overlapping
  validate.py          overfit gate: PBO(CSCV) + Deflated Sharpe + purged/walk-forward
  backtest.py          GPU bootstrap Sharpe-CI + permutation p-value
  publish.py           fail-safe advisory flag-file writer/reader
  train.py             orchestrator: panel -> features -> {RF, torch} -> gate -> flag
  models/
    baseline_rf.py     sklearn RF baseline (reference bar)
    torch_seq.py       GPU MLP (head-to-head vs RF)
    shift_detector.py  regime/vol-shift detector + surprise-spike reframe + payoffs
    shift_export.py    live signal -> tv_signal.json + webhook + Pine (approx)
  scripts/             venv setup + daily Task Scheduler job for the signal
  tests/               no-leakage, non-overlapping, flag fail-safe (6 tests)
  data/                panel_YYYYMMDD.parquet (+ .meta.json), panel_latest.txt
  journal/
    scorecards/        per-model gate verdicts (JSON)
    flags/             macro_gpu_state.json, tv_signal.json  <-- consume these
    runs/              scheduled-run logs
```

## 3. Data available (what to "scoop")

- **Universe (22, 21 usable):** rates TLT/TIP (US10Y drops — Yahoo serves only 1
  daily bar for yield indices ^TNX/^TYX/^FVX); dollar DXY/EURUSD/USDJPY/GBPUSD/
  USDCAD/USDCNH(=CNY=X, CNH=X is dead on Yahoo); equities SPY/QQQ/IWM/EEM/XLF;
  vol VIX; commodities GLD/USO/CPER; credit HYG/LQD; liquidity BIL; crypto BTC.
  Defined in `config.UNIVERSE` (logical name -> Yahoo + Stooq symbols).
- **Depth:** ~2,512 daily bars (10y) per asset, 2016-07 .. present.
- **Source gotchas (already handled, keep in mind):**
  - Yahoo `range=max` silently returns MONTHLY — must use `period1/period2` +
    `interval=1d` for true daily. `data.py` does this.
  - Stooq CSV is bot-blocked on this box (JS challenge) -> fails safe to
    Yahoo-only. If you need deeper history, fix Stooq access or add another source.
  - Panel is aligned to SPY's trading calendar (no LOCF; co-traded days only).
- **Access it:** `from macro_gpu_lab.data import load_latest_panel` (wide daily
  close DataFrame), or `build_panel()` to refetch. `features.build_features(panel)`
  for the relationship feature frame.

## 4. How to run

```powershell
# one-time
pwsh scripts/setup_venv.ps1          # uv venv Py3.11 + torch cu121 (--native-tls)
.venv\Scripts\python.exe -m macro_gpu_lab.gpu     # CUDA smoke test

# refresh data + models
.venv\Scripts\python.exe -m macro_gpu_lab.data                       # rebuild panel
.venv\Scripts\python.exe -m macro_gpu_lab.train --once               # direction models + gate + flag
.venv\Scripts\python.exe -m macro_gpu_lab.models.shift_detector --surprise   # calm-eruption research
.venv\Scripts\python.exe -m macro_gpu_lab.models.shift_export        # write live signal + Pine
pytest tests -q
```

Always use `.venv\Scripts\python.exe` (PATH `python` is a broken Store stub here).

## 5. Live signal to consume: `journal/flags/tv_signal.json`

Written by `shift_export`. Fail-safe (missing/stale -> neutral). Shape:

```json
{
  "model": "surprise_shift",
  "as_of": "...Z", "data_through": "YYYY-MM-DD",
  "status": "SHADOW",                       // unvalidated — do not size on this alone
  "regime_note": "edge is ~2016-2026 specific; degrades on older data",
  "horizons": {
    "5d":  {"calm": false, "p_eruption": 0.49, "eruption_predicted": false,
            "actions": {"short_spy_overlay": "LONG_SPY", "long_vol_selective": "FLAT"}},
    "21d": { ... }
  },
  "feature_importance": {"spy_vol20": 0.20, "vix_chg20": 0.15, "absorption": 0.14, ...}
}
```

- Signal is only meaningful on a **calm** day (`calm: true`) where
  `eruption_predicted: true` -> that is the "vol eruption from calm" setup.
- `long_vol_selective: BUY_STRADDLE` = the model flags a likely vol spike -> the
  Robinhood-appropriate expression is a **long-vol / options** structure, not a
  directional stock trade. `short_spy_overlay` is the crude directional version.

## 6. Strategies tried + WHAT WE LEARNED (read before trading)

| # | idea | result | verdict |
|---|------|--------|---------|
| 1 | Per-asset forward-return direction (RF + GPU MLP, 5d/21d) | DSR −4 to −13, PF~1 | **noise.** Daily direction from these features is not predictable. |
| 2 | Regime/vol-shift detector (predict SPY vol spike) | precision 0.28–0.39 | **fake skill** — a one-line "vol already high" baseline matched/beat it (vol clustering). |
| 3 | **Surprise reframe**: predict eruption from a CALM state (where persistence is blind) | 5d precision 0.230 vs 0.104 base (2.21x); 21d 0.177 vs 0.079 (2.23x), recall 0.45–0.60 | **REAL lift** over persistence — genuine cross-asset skill. |
| 4 | Monetize #3: short SPY overlay | 5d PF 1.42, 21d PF 1.99 | PF>1 but DSR<0 (not significant), regime-fragile. |
| 5 | Monetize #3: long-vol (straddle) | straddle-every-calm-day PF 0.72 (theta bleed); model-**selected** days 21d PF 2.43 (n=16) | selection beats indiscriminate vol buying, but n tiny, DSR<0. |
| 6 | Robustness: extend to 15y | 21d lift 2.23x -> 1.42x, PF 1.85 -> 1.08 | **regime-specific** to 2016-2026; fails out-of-regime. |

Net: the surprise-eruption signal (#3) is the only real finding. It genuinely
predicts vol eruptions from calm at ~2.2x base rate using cross-asset stress
(top drivers: `spy_vol20`, `vix_chg20`, `absorption`, `avg_pairwise_corr`), but
**no monetization cleared the overfit gate and the edge is regime-dependent.**

### Hard-won methodology (inherit this — it is the real asset)

1. **Baseline every "edge" against a trivial rule.** #2 looked skillful; a
   one-liner killed it. Always ask "what's the dumb version, and do I beat it?"
2. **Test out-of-regime.** #6 exposed #3 as 2016+ only. In-sample lift lies.
3. **Overfit gate is the judge, not PF.** PF>1 means nothing without Deflated
   Sharpe > 0 and PBO < 0.5 on non-overlapping, walk-forward OOS PnL. Reuse
   `validate.evaluate_gate`.
4. **Non-overlapping trades + purge/embargo** or DSR is a lie (labels overlap).
5. **No look-ahead:** all features backward-looking; a test enforces it. No LOCF;
   align to a common trading calendar.
6. **Deflate for multiple testing:** trying N strategies raises the bar — pass
   `n_trials` accordingly (we used strategies x horizons).
7. **Costs are fractions of price, not absolute.** A straddle has a premium; a
   directional trade has spread/slippage. Model them or the PnL is fiction.

## 7. If Codex wants to trade off this (Robinhood executor)

- Treat every signal here as **SHADOW / unvalidated / regime-fragile**. Paper or
  micro-size first; grade forward for weeks before trusting it.
- Prefer the **long-vol** expression for the surprise signal (it predicts vol,
  not direction). Directional short is the weaker read.
- The RF can't run in Pine; `tv_signal.json` is the authoritative signal. The
  Robinhood/Codex side should read that file (or wire `shift_export --post` to a
  webhook Codex listens on). Refresh daily (a Task Scheduler job exists:
  `scripts/register_daily_export.ps1`; the webhook post-target was left
  unfinished — point it at Codex).
- Build new strategies through the SAME gate (`validate.py`) before trusting them.
  Reuse `features.py`, `baseline_rf.py` patterns. Default shadow; earn promotion.

## 8. Macro brain (separate — reference only, DO NOT modify from here)

There are TWO macro brains; keep them straight:

- **This repo (macro_gpu_lab):** the GPU research/data base described above.
- **HQ Trading System macro brain** at `C:\Users\<your-user>\hq-trading-system\analytics\`:
  a live (shadow) cross-asset engine — `macro_engine.py` computes 60-bar
  correlations, OLS betas, z-scored divergences, partial correlations (Ledoit-
  Wolf), absorption ratio, lead-lag, and a regime label, writing
  `journal/macro/macro_state.json` every ~5 min. `macro_gate.py` maps that to a
  per-bucket ON/WAIT/OFF verdict (shadow, `MACRO_ENFORCE=no`).
  `macro_reconcile.py` grades whether it would have helped. A separate agent owns
  this — **do not edit HQ or post to its `/ingest` trade pipeline.**

Feature math in this repo mirrors that engine (same windows/formulas) but on a
multi-year DAILY panel instead of HQ's live ~150-bar intraday snapshot. If you
want the live intraday macro read, consume HQ's `macro_state.json` **read-only**;
if you want trainable history, use this repo's panel.

## 9. Guardrails (non-negotiable, carried from the quant repos)

- Research/advisory here. This repo never routes an order. If Codex+Robinhood
  executes, that responsibility and risk live entirely on the Codex side.
- Unvalidated models stay shadow; the gate promotes, not hope.
- Secrets (webhook URLs, tokens) live in `secrets/` (gitignored) or env — never
  committed, never logged in full.
- Do not wire this into HQ's execution path.

## 10. State at handoff (2026-07-05)

Stages 0–7 complete; 6 tests pass. Direction models SHADOW. Surprise-eruption
signal is the one real finding (2.2x lift, regime-fragile, unmonetized past the
gate). Live signal file + daily scheduler working; webhook post-target
intentionally left for Codex to own. Latest commits: e2e04df (scheduler),
1c0a8a3 (long-vol + export). See `git log` for the full trail.
