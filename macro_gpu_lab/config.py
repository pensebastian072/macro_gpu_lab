"""Central config — universe, horizons, paths, gate thresholds, device.

Universe mirrors hq-trading-system/analytics/macro_config.py::UNIVERSE (the same
22 tradable Yahoo proxies for each macro force). This lab pulls DAILY history for
them (multi-year) rather than the live 60m snapshot the HQ engine caches.
"""
from __future__ import annotations

import os
from pathlib import Path

# ── Universe: logical name → source symbols ─────────────────────────
# yahoo: Yahoo v8 chart symbol (%5E = ^, =X = FX spot).
# stooq: Stooq CSV symbol (lowercase; .us suffix for US equities/ETFs). None =
#        no reliable Stooq mirror → Yahoo-only for that name.
UNIVERSE = {
    # rates — NB: Yahoo serves only 1 daily bar for yield indices (^TNX/^TYX/^FVX),
    # so US10Y auto-drops; TLT (inverse long-yield) + TIP (real-rate) proxy rates.
    "US10Y":  {"yahoo": "%5ETNX",   "stooq": "^tnx"},
    "TLT":    {"yahoo": "TLT",      "stooq": "tlt.us"},
    "TIP":    {"yahoo": "TIP",      "stooq": "tip.us"},
    # dollar
    "DXY":    {"yahoo": "DX-Y.NYB", "stooq": "^dxy"},
    "EURUSD": {"yahoo": "EURUSD=X", "stooq": "eurusd"},
    "USDJPY": {"yahoo": "JPY=X",    "stooq": "usdjpy"},
    "GBPUSD": {"yahoo": "GBPUSD=X", "stooq": "gbpusd"},
    "USDCAD": {"yahoo": "CAD=X",    "stooq": "usdcad"},
    # CNH=X returns 1 daily bar on Yahoo; CNY=X (onshore yuan) has full history
    # and is an equivalent China-stress proxy.
    "USDCNH": {"yahoo": "CNY=X",    "stooq": "usdcnh"},
    # equities
    "SPY":    {"yahoo": "SPY",      "stooq": "spy.us"},
    "QQQ":    {"yahoo": "QQQ",      "stooq": "qqq.us"},
    "IWM":    {"yahoo": "IWM",      "stooq": "iwm.us"},
    "EEM":    {"yahoo": "EEM",      "stooq": "eem.us"},
    "XLF":    {"yahoo": "XLF",      "stooq": "xlf.us"},
    # volatility
    "VIX":    {"yahoo": "%5EVIX",   "stooq": "^vix"},
    # commodities
    "GOLD":   {"yahoo": "GLD",      "stooq": "gld.us"},
    "OIL":    {"yahoo": "USO",      "stooq": "uso.us"},
    "COPPER": {"yahoo": "CPER",     "stooq": "cper.us"},
    # credit
    "HYG":    {"yahoo": "HYG",      "stooq": "hyg.us"},
    "LQD":    {"yahoo": "LQD",      "stooq": "lqd.us"},
    # liquidity
    "BIL":    {"yahoo": "BIL",      "stooq": "bil.us"},
    # crypto
    "BTC":    {"yahoo": "BTC-USD",  "stooq": "btcusd"},
}

FORCE = {
    "US10Y": "rates", "TLT": "rates", "TIP": "rates",
    "DXY": "dollar", "EURUSD": "dollar", "USDJPY": "dollar",
    "GBPUSD": "dollar", "USDCAD": "dollar", "USDCNH": "china",
    "SPY": "equities", "QQQ": "equities", "IWM": "equities", "EEM": "equities", "XLF": "equities",
    "VIX": "volatility",
    "GOLD": "commodities", "OIL": "commodities", "COPPER": "commodities",
    "HYG": "credit", "LQD": "credit",
    "BIL": "liquidity",
    "BTC": "crypto",
}

ASSETS = list(UNIVERSE.keys())

# ── Data pull ───────────────────────────────────────────────────────
DAILY_YEARS = 10               # full 22-asset coverage (BTC/CPER start ~2016);
                               # a 15y test showed the surprise edge is regime-
                               # specific to 2016+ and degrades on older data.
MIN_ASSET_ROWS = 200           # drop an asset with fewer daily bars than this (junk feed)
MIN_ROWS_FOR_TRAIN = 750       # ~3y business days (copper MIN_ROWS_FOR_TRAIN bar)
BETA_BASES = ("SPY", "DXY")

# ── Feature windows (daily bars) — mirror macro_config intent ───────
CORR_WINDOW = 60
CORR_WINDOW_FAST = 20
Z_WINDOW = 120
LL_MAX_LAG = 6
LL_MARGIN = 0.10
LL_MIN_ABS = 0.25
ABSORPTION_K_DIVISOR = 5
PARTIAL_MIN_COVERAGE = 0.6
PARTIAL_MIN_ROWS = 60

# Relationship anchors each asset's rolling correlation is measured against
# (beyond the SPY/DXY regression bases): safe-haven, duration, credit.
CORR_ANCHORS = ("SPY", "DXY", "GOLD", "TLT", "HYG")
BETA_CHG_WINDOW = 20          # beta-drift lookback (sensitivity-regime change)

# Divergence spreads to z-score (mirror macro_config.SPREADS).
SPREADS = {
    "copper_gold": ("COPPER", "GOLD"),
    "qqq_tlt":     ("QQQ", "TLT"),
    "gold_dxy":    ("GOLD", "DXY"),
    "spy_hyg":     ("SPY", "HYG"),
    "btc_qqq":     ("BTC", "QQQ"),
    "tip_tlt":     ("TIP", "TLT"),
    "lqd_hyg":     ("LQD", "HYG"),
    "eem_spy":     ("EEM", "SPY"),
    "xlf_spy":     ("XLF", "SPY"),
}
Z_DIVERGENCE = 2.0

# ── Labels ──────────────────────────────────────────────────────────
HORIZONS_DAYS = [5, 21]        # forward-return horizons; BOTH must clear the gate

# ── Stage 7: relationship/regime SHIFT detector ─────────────────────
# A "shift" = forward realized-vol spike on SPY: std of daily returns over the
# next h days exceeds SHIFT_VOL_MULT x its trailing-median vol. The monetizable
# overlay goes defensive (short SPY) when a shift is predicted, else risk-on.
SHIFT_VOL_MULT = 1.5
SHIFT_VOL_MEDIAN_WINDOW = 120
SHIFT_TARGET = "SPY"
# Risk basket the defensive overlay trades. Pooling into a diversified portfolio
# (one PnL per rebalance date, averaged across the basket) lowers idiosyncratic
# variance -> higher honest Sharpe. NOT one-trade-per-asset (that would be
# pseudo-replication of correlated same-date events, faking sample size).
RISK_ASSETS = ("SPY", "QQQ", "IWM", "EEM", "HYG")
# Long-vol (ATM straddle) payoff proxy: |forward move| - premium, where premium =
# STRADDLE_PREMIUM_MULT * trailing daily vol * sqrt(horizon) (~ATM straddle cost).
# Profits from a big move either direction — the natural trade for a spike signal.
STRADDLE_PREMIUM_MULT = 0.8
TRAIL_VOL_WINDOW = 20

# ── Overfit gate thresholds (mirror copper_brain/validate.py) ───────
PBO_MAX = 0.5
# RAISED 0.0 -> 1.645 on 2026-07-30, together with the deflated_sharpe unit fix.
# Once the DSR is computed correctly, `ratio > 0` is a MEDIAN test -- it asks only
# "better than the EXPECTED max of N noise trials" -- and best-of-8 pure noise clears
# it 44.8% of the time (research_ledger/audit/dsr_audit.py). 1.645 is prob > 0.95,
# i.e. the one-sided 5% bar, which is what a promotion threshold has to mean.
#
# CORRECTED 2026-08-07. This comment previously read "Under it exactly one candidate on
# this box survives (qlib H07_vrp_spy, +2.66)". That is FALSE and no scorecard on disk
# supports it: H07_vrp_spy failed both times it was run -- ratio -17.2909 on 2026-07-13
# (pre-unit-fix) and -0.0043 on 2026-07-30 (post-fix, n_trials=20), `passes: false` in
# both. NOTHING on this box has ever cleared this bar. That the false claim sat inside
# the justification for the threshold itself is exactly why it mattered.
#
# H07 does clear at n_trials=1 (ratio 1.896), but it was one of 15 mined anomalies, and
# redeclaring it as a single trial after seeing it won is the laundering this gate
# exists to prevent. It is separately disqualified by Q-GATE-04: "long SPY when
# condition, else flat" has no benchmark leg and may be booking market beta as edge.
DEFLATED_SHARPE_MIN = 1.645
N_WALK_FORWARD_FOLDS = 6
CSCV_N_GROUPS = 10

# ── RandomForest baseline hyperparams (mirror copper RF_KWARGS) ─────
RF_KWARGS = dict(
    n_estimators=300,
    max_depth=5,
    min_samples_leaf=50,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

# ── Torch model (small + regularized — data is not big) ─────────────
SEED = 42
TORCH_HIDDEN = 48
TORCH_DROPOUT = 0.3
TORCH_WEIGHT_DECAY = 1e-3
TORCH_LR = 1e-3
TORCH_MAX_EPOCHS = 60
TORCH_PATIENCE = 8            # early-stop patience
TORCH_SEQ_LEN = 20           # lookback window fed to the GRU

# ── Promotion flag (default SHADOW — never auto-flips) ──────────────
MACRO_GPU_ENFORCE = os.environ.get("MACRO_GPU_ENFORCE", "no").strip().lower()

# ── Advisory webhook (notification sink only — never a broker) ───────
# Read from env; may embed a token, so it is NEVER logged in full or committed.
WEBHOOK_URL = os.environ.get("MACRO_GPU_WEBHOOK_URL", "").strip()

# ── Paths ───────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
JOURNAL_DIR = BASE_DIR / "journal"
MODELS_DIR = JOURNAL_DIR / "models"
SCORECARDS_DIR = JOURNAL_DIR / "scorecards"
FLAGS_DIR = JOURNAL_DIR / "flags"
FLAG_PATH = FLAGS_DIR / "macro_gpu_state.json"
FLAG_STALE_DAYS = 4

for _d in (DATA_DIR, MODELS_DIR, SCORECARDS_DIR, FLAGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
