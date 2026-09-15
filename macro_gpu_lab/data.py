"""Multi-source daily panel builder.

Assembles a multi-year DAILY close panel for the 22-asset macro universe from two
free, no-key sources:

  - Yahoo v8 chart (primary, recent + authoritative) — same endpoint/fail-safe
    contract as hq-trading-system/analytics/market_data.py.
  - Stooq daily CSV (backfill) — usually reaches far deeper history than Yahoo.

Reconciliation is return-preserving, NOT a naive concat: Yahoo is the base; Stooq
is ratio-scaled to Yahoo on their overlap and only its PRE-Yahoo tail is
prepended. Scaling by the overlap median keeps multiplicative returns intact
across the seam so we never manufacture a fake jump-return (same ethos as the
macro engine's no-LOCF rule). If both sources fail for a ticker it is dropped.

Output: data/panel_YYYYMMDD.parquet (index=date, one close column per asset) plus
a sidecar data/panel_YYYYMMDD.meta.json with per-asset source + coverage. Every
network call fails safe to None; the builder never raises into a caller.
"""
from __future__ import annotations

import io
import json
import logging
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

import pandas as pd

from . import config

logger = logging.getLogger("macro_gpu_lab.data")

_UA = {"User-Agent": "Mozilla/5.0 macro_gpu_lab/1.0"}
# NB: range=max silently downsamples to MONTHLY bars — use an explicit
# period1/period2 window with interval=1d to get true daily granularity.
YAHOO_CHART = (
    "https://query1.finance.yahoo.com/v8/finance/chart/"
    "{sym}?period1={p1}&period2={p2}&interval=1d"
)
# Stooq CSV is often bot-blocked on this box (returns a JS-challenge HTML page);
# the parser fail-safes to None so the panel degrades cleanly to Yahoo-only.
STOOQ_CSV = "https://stooq.com/q/d/l/?s={sym}&i=d"

MIN_OVERLAP_DAYS = 20   # need this many shared days to trust the ratio scale


# ────────────────────────────────────────────────────────────────
# Per-source fetchers (fail-safe -> None)
# ────────────────────────────────────────────────────────────────

def fetch_yahoo_daily(sym: str, timeout: int = 10) -> pd.Series | None:
    """Daily close series (index=UTC-naive date, name='close') or None."""
    p2 = int(time.time())
    p1 = p2 - 3600 * 24 * 365 * config.DAILY_YEARS
    url = YAHOO_CHART.format(sym=sym, p1=p1, p2=p2)
    try:
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = json.loads(r.read())
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
        logger.info(f"yahoo fetch failed {sym}: {e}")
        return None
    try:
        result = raw["chart"]["result"][0]
        ts = result["timestamp"]
        closes = result["indicators"]["quote"][0]["close"]
    except (KeyError, IndexError, TypeError):
        return None
    pairs = [(t, c) for t, c in zip(ts, closes) if c is not None and t is not None]
    if not pairs:
        return None
    idx = pd.to_datetime([t for t, _ in pairs], unit="s", utc=True).tz_localize(None).normalize()
    s = pd.Series([float(c) for _, c in pairs], index=idx, name="close")
    return s[~s.index.duplicated(keep="last")].sort_index()


def fetch_stooq_daily(sym: str, timeout: int = 10) -> pd.Series | None:
    """Daily close series from Stooq CSV, or None."""
    url = STOOQ_CSV.format(sym=sym)
    try:
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode("utf-8", "replace")
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        logger.info(f"stooq fetch failed {sym}: {e}")
        return None
    if not body or body.lstrip().lower().startswith("<") or "No data" in body:
        return None
    try:
        df = pd.read_csv(io.StringIO(body))
    except Exception:  # noqa: BLE001
        return None
    if "Date" not in df.columns or "Close" not in df.columns or df.empty:
        return None
    idx = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    s = pd.Series(pd.to_numeric(df["Close"], errors="coerce").values, index=idx, name="close")
    s = s.dropna()
    s = s[~s.index.isna()]
    if s.empty:
        return None
    return s[~s.index.duplicated(keep="last")].sort_index()


# ────────────────────────────────────────────────────────────────
# Reconcile one asset
# ────────────────────────────────────────────────────────────────

def reconcile_asset(name: str) -> tuple[pd.Series | None, dict]:
    """Return (close_series, meta) for one universe name. Series may be None."""
    syms = config.UNIVERSE[name]
    y = fetch_yahoo_daily(syms["yahoo"]) if syms.get("yahoo") else None
    st = fetch_stooq_daily(syms["stooq"]) if syms.get("stooq") else None
    meta = {"yahoo_rows": 0 if y is None else int(y.size),
            "stooq_rows": 0 if st is None else int(st.size),
            "source": None, "rows": 0, "start": None, "end": None,
            "overlap_corr": None}

    if y is None and st is None:
        return None, meta
    if y is None:
        meta.update(source="stooq")
        out = st
    elif st is None:
        meta.update(source="yahoo")
        out = y
    else:
        # Both present. Yahoo is the base; ratio-scale Stooq on the overlap and
        # prepend only its pre-Yahoo tail (return-preserving backfill).
        overlap = y.index.intersection(st.index)
        if overlap.size >= MIN_OVERLAP_DAYS:
            ratio = float((y.reindex(overlap) / st.reindex(overlap)).median())
            # health cross-check: return correlation on the overlap
            rc = (y.reindex(overlap).pct_change().corr(st.reindex(overlap).pct_change()))
            meta["overlap_corr"] = None if pd.isna(rc) else round(float(rc), 4)
            tail = st[st.index < y.index.min()] * ratio
            out = pd.concat([tail, y]).sort_index()
            out = out[~out.index.duplicated(keep="last")]
            meta.update(source="yahoo+stooq")
        else:
            meta.update(source="yahoo")
            out = y

    out = out.dropna().sort_index()
    if out.size < config.MIN_ASSET_ROWS:
        meta.update(source=None, rows=int(out.size))
        return None, meta   # junk feed (e.g. Yahoo yield indices give 1 bar)
    meta.update(rows=int(out.size),
                start=out.index.min().date().isoformat(),
                end=out.index.max().date().isoformat())
    return out, meta


# ────────────────────────────────────────────────────────────────
# Build + persist the wide panel
# ────────────────────────────────────────────────────────────────

def build_panel() -> tuple[pd.DataFrame, dict]:
    """Fetch + reconcile every asset into a wide daily close panel."""
    series: dict[str, pd.Series] = {}
    provenance: dict[str, dict] = {}
    for name in config.ASSETS:
        s, meta = reconcile_asset(name)
        provenance[name] = meta
        if s is None or s.empty:
            logger.info(f"{name}: dropped (no source)")
            continue
        series[name] = s

    if not series:
        return pd.DataFrame(), {"provenance": provenance, "dropped": config.ASSETS}

    panel = pd.DataFrame(series).sort_index()
    # trim leading all-NaN rows; keep NaNs elsewhere (feature step aligns pairwise)
    panel = panel.loc[panel.notna().any(axis=1)]
    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "n_assets": panel.shape[1],
        "n_rows": panel.shape[0],
        "date_start": panel.index.min().date().isoformat(),
        "date_end": panel.index.max().date().isoformat(),
        "dropped": [a for a in config.ASSETS if a not in series],
        "provenance": provenance,
    }
    return panel, meta


def save_panel(panel: pd.DataFrame, meta: dict) -> tuple[str, str]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    ppath = config.DATA_DIR / f"panel_{stamp}.parquet"
    mpath = config.DATA_DIR / f"panel_{stamp}.meta.json"
    panel.to_parquet(ppath)
    mpath.write_text(json.dumps(meta, indent=2))
    # convenience pointer to the newest panel
    (config.DATA_DIR / "panel_latest.txt").write_text(ppath.name)
    return str(ppath), str(mpath)


def load_latest_panel() -> pd.DataFrame | None:
    ptr = config.DATA_DIR / "panel_latest.txt"
    if ptr.exists():
        p = config.DATA_DIR / ptr.read_text().strip()
        if p.exists():
            return pd.read_parquet(p)
    panels = sorted(config.DATA_DIR.glob("panel_*.parquet"))
    return pd.read_parquet(panels[-1]) if panels else None


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    panel, meta = build_panel()
    if panel.empty:
        print("PANEL EMPTY — all sources failed")
        print(json.dumps(meta, indent=2))
        return
    ppath, mpath = save_panel(panel, meta)
    print(f"panel  -> {ppath}")
    print(f"meta   -> {mpath}")
    print(f"assets : {meta['n_assets']}/{len(config.ASSETS)}  "
          f"rows: {meta['n_rows']}  {meta['date_start']} .. {meta['date_end']}")
    if meta["dropped"]:
        print(f"dropped: {meta['dropped']}")
    ok = meta["n_rows"] >= config.MIN_ROWS_FOR_TRAIN
    print(f"min-rows gate ({config.MIN_ROWS_FOR_TRAIN}): {'OK' if ok else 'TOO FEW'}")
    # per-asset coverage
    for name, pv in meta["provenance"].items():
        if pv["source"]:
            print(f"  {name:7s} {pv['source']:12s} rows={pv['rows']:5d} "
                  f"{pv['start']}..{pv['end']} y={pv['yahoo_rows']} s={pv['stooq_rows']} "
                  f"xcorr={pv['overlap_corr']}")


if __name__ == "__main__":
    main()
