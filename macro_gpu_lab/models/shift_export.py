"""Export the surprise-shift signal for manual / TradingView use.

The RF cannot run inside Pine, so this module treats the Python model as the
source of truth: it fits the calm-state eruption detector on all history, emits
the CURRENT live signal to a fail-safe flag-file (journal/flags/tv_signal.json),
and generates a TransParent Pine v5 study that APPROXIMATES the signal from
Pine-computable cross-asset inputs (calm filter + credit/curve/vol/corr stress)
so the user can get alerts in TradingView.

IMPORTANT — honesty labels baked into the output:
  - SHADOW / unvalidated: none of these strategies clear the overfit gate
    (DSR < 0); PF > 1 is not statistical significance.
  - Regime-specific: the edge holds on ~2016-2026 and degrades on older data.
  - The Pine study is an APPROXIMATION of the RF, not the RF itself; the JSON
    flag-file is the authoritative signal.
"""
from __future__ import annotations

import hashlib
import json
import math
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .. import config
from ..features import _to_common_grid
from .shift_detector import (build_market_features, shift_label, spy_fwd_ret,
                             _calm_and_base, _trail_daily_vol)

TV_FLAG = config.FLAGS_DIR / "tv_signal.json"
LASTPOST = config.FLAGS_DIR / "tv_signal_lastpost.txt"
PINE_DIR = config.BASE_DIR / "tradingview"


def _fit_latest(panel: pd.DataFrame):
    """Fit RF per horizon on all calm-labelled rows; return latest-row signal +
    feature importances (averaged across horizons)."""
    from sklearn.ensemble import RandomForestClassifier

    feat = build_market_features(panel)
    cols = list(feat.columns)
    trail_vol = _trail_daily_vol(panel).reindex(feat.index)
    per_h = {}
    imp_acc = np.zeros(len(cols))
    imp_n = 0
    for h in config.HORIZONS_DAYS:
        calm, _, _ = _calm_and_base(panel, h)
        calm = calm.reindex(feat.index)
        spike = shift_label(panel, h).reindex(feat.index)
        label = (spike * calm.astype(float)).where(calm)
        df = feat.join(label.rename("y")).dropna(subset=["y"])
        y = df["y"].to_numpy().astype(int)
        if np.unique(y).size < 2:
            continue
        m = RandomForestClassifier(**config.RF_KWARGS)
        m.fit(df[cols].to_numpy(), y)
        imp_acc += m.feature_importances_
        imp_n += 1
        # latest row signal
        last = feat.iloc[[-1]]
        p = float(m.predict_proba(last.to_numpy())[0, 1])
        calm_now = bool(calm.iloc[-1]) if pd.notna(calm.iloc[-1]) else False
        pred = bool(p > 0.5)
        premium = (config.STRADDLE_PREMIUM_MULT
                   * float(trail_vol.iloc[-1]) * math.sqrt(h)) if pd.notna(trail_vol.iloc[-1]) else None
        per_h[f"{h}d"] = {
            "calm": calm_now,
            "p_eruption": round(p, 4),
            "eruption_predicted": pred,
            "straddle_premium": None if premium is None else round(premium, 5),
            "actions": {
                # only act on a calm day; else the signal is out-of-scope
                "short_spy_overlay": ("SHORT_SPY" if (calm_now and pred) else "LONG_SPY"),
                "long_vol_selective": ("BUY_STRADDLE" if (calm_now and pred) else "FLAT"),
            },
        }
    importances = {}
    if imp_n:
        avg = imp_acc / imp_n
        importances = {c: round(float(v), 4)
                       for c, v in sorted(zip(cols, avg), key=lambda t: -t[1])}
    return per_h, importances, feat.index[-1]


def build_signal(panel: pd.DataFrame) -> dict:
    per_h, importances, data_through = _fit_latest(panel)
    return {
        "model": "surprise_shift",
        "as_of": datetime.now(timezone.utc).isoformat(),
        "data_through": str(data_through.date()) if data_through is not None else None,
        "status": "SHADOW",              # unvalidated — DSR < 0
        "regime_note": "edge is ~2016-2026 specific; degrades on older data",
        "enforce": "no",
        "horizons": per_h,
        "feature_importance": importances,
        "disclaimer": ("advisory/paper only; PF>1 is not significance; the Pine "
                       "study approximates this RF signal, this JSON is the truth"),
    }


def publish_signal(sig: dict) -> str:
    tmp = TV_FLAG.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(sig, indent=2))
    tmp.replace(TV_FLAG)
    return str(TV_FLAG)


# ── Pine v5 study: transparent approximation of the RF signal ───────
PINE = r'''//@version=5
// macro_gpu_lab -- surprise-shift signal (TRANSPARENT APPROXIMATION)
// Fires when the market is CALM (SPY realized vol below its median) yet
// cross-asset STRESS is rising -- the setup the RF associates with a vol
// eruption from calm. This is an approximation of the Python RF model; the
// authoritative signal is journal/flags/tv_signal.json.
//
// SHADOW / UNVALIDATED: this did NOT clear the overfit gate (DSR < 0). PF>1 is
// not significance. Edge is ~2016-2026 specific. Paper/manual use only.
indicator("macro_gpu surprise-shift (approx)", overlay=true)

hz = input.int(21, "Horizon (days)")
volMedLen = input.int(120, "Vol median window")

// --- cross-asset inputs (daily) --- chosen to match the RF's top features:
// spy_vol20, vix_chg20, absorption/avg_pairwise_corr (assets bunching).
spy = request.security("AMEX:SPY", "D", close)
qqq = request.security("NASDAQ:QQQ", "D", close)
eem = request.security("AMEX:EEM", "D", close)
hyg = request.security("AMEX:HYG", "D", close)
vix = request.security("CBOE:VIX", "D", close)

// --- calm filter: SPY realized vol <= trailing median ---
ret = math.log(spy / spy[1])
rv = ta.stdev(ret, hz)
calm = rv <= ta.median(rv, volMedLen)

// --- vol ticking up from calm (vix_chg20 was the #2 feature) ---
vixRising = ta.change(vix, 20) > 0
vixLow = vix < ta.percentile_nearest_rank(vix, 252, 40)   // complacent level

// --- avg cross-asset correlation rising (absorption/co-movement proxy) ---
rq = math.log(qqq / qqq[1])
re = math.log(eem / eem[1])
rh = math.log(hyg / hyg[1])
avgCorr = (ta.correlation(ret, rq, 20) + ta.correlation(ret, re, 20)
           + ta.correlation(ret, rh, 20)) / 3.0
corrRising = ta.change(avgCorr, 5) > 0.1

// eruption-from-calm setup: calm + complacent, but vol starting to lift and
// assets starting to bunch together.
signal = calm and vixLow and vixRising and corrRising

plotshape(signal, title="surprise-shift", style=shape.triangleup,
          location=location.belowbar, color=color.orange, size=size.small)
alertcondition(signal, "macro_gpu surprise-shift",
     "Calm-state vol-eruption setup: consider long-vol / risk-off (SHADOW signal)")
'''


def write_pine() -> str:
    PINE_DIR.mkdir(parents=True, exist_ok=True)
    p = PINE_DIR / "macro_surprise_signal.pine"
    p.write_text(PINE, encoding="utf-8")
    return str(p)


# ── Advisory webhook delivery (exact RF signal, no Pine drift) ──────

def _actionable(sig: dict) -> bool:
    """True if any horizon is a calm-state eruption call (a live setup)."""
    return any(h.get("calm") and h.get("eruption_predicted")
               for h in sig.get("horizons", {}).values())


def _summary_text(sig: dict) -> str:
    lines = [f"macro_gpu surprise-shift [{sig['status']}] as of {sig['data_through']}"]
    for h, s in sig["horizons"].items():
        tag = ("SETUP" if (s["calm"] and s["eruption_predicted"])
               else ("calm" if s["calm"] else "not-calm"))
        lines.append(f"  {h}: {tag} p={s['p_eruption']} -> "
                     f"{s['actions']['long_vol_selective']} / "
                     f"{s['actions']['short_spy_overlay']}")
    lines.append("advisory/paper only; unvalidated (DSR<0); 2016+ regime")
    return "\n".join(lines)


def _payload(url: str, sig: dict, text: str) -> dict:
    """Shape the body per common webhook sinks; generic endpoints get the lot."""
    host = urllib.parse.urlparse(url).netloc.lower()
    if "discord" in host:
        return {"content": text}
    if "slack" in host or "hooks.slack" in host:
        return {"text": text}
    return {"text": text, "signal": sig}   # generic JSON endpoint


def _redact(url: str) -> str:
    p = urllib.parse.urlparse(url)
    return f"{p.scheme}://{p.netloc}/..."   # host only — never the token/path


def _actions_hash(sig: dict) -> str:
    payload = {h: s["actions"] for h, s in sig.get("horizons", {}).items()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def post_signal(sig: dict, url: str | None = None, always: bool = False,
                timeout: int = 8) -> dict:
    """POST the signal to the advisory webhook. Fail-safe: never raises; returns
    a status dict. Skips a repeat post when nothing is actionable AND the actions
    are unchanged since the last post (avoids daily FLAT spam)."""
    url = url or config.WEBHOOK_URL
    if not url:
        return {"posted": False, "reason": "no MACRO_GPU_WEBHOOK_URL set"}

    h = _actions_hash(sig)
    last = LASTPOST.read_text().strip() if LASTPOST.exists() else ""
    if not always and not _actionable(sig) and h == last:
        return {"posted": False, "reason": "unchanged + not actionable (skipped)"}

    body = json.dumps(_payload(url, sig, _summary_text(sig))).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            code = r.getcode()
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return {"posted": False, "reason": f"transport error: {type(e).__name__}",
                "url": _redact(url)}
    try:
        LASTPOST.write_text(h)
    except Exception:  # noqa: BLE001
        pass
    return {"posted": True, "http": code, "actionable": _actionable(sig),
            "url": _redact(url)}


def main() -> None:
    import argparse
    from .. import data as data_mod
    ap = argparse.ArgumentParser()
    ap.add_argument("--post", action="store_true",
                    help="POST the signal to MACRO_GPU_WEBHOOK_URL")
    ap.add_argument("--always", action="store_true",
                    help="post even if unchanged / not actionable")
    ap.add_argument("--webhook-url", default=None,
                    help="override the webhook URL (else env MACRO_GPU_WEBHOOK_URL)")
    args = ap.parse_args()

    panel = data_mod.load_latest_panel()
    if panel is None or panel.empty:
        print("no panel — run macro_gpu_lab.data first")
        return
    sig = build_signal(panel)
    fp = publish_signal(sig)
    pine = write_pine()
    print(f"signal -> {fp}")
    print(f"pine   -> {pine}")
    print(f"status : {sig['status']}  ({sig['regime_note']})")
    for h, s in sig["horizons"].items():
        print(f"  {h}: calm={s['calm']} p_eruption={s['p_eruption']} "
              f"pred={s['eruption_predicted']}  -> {s['actions']}")
    top = list(sig["feature_importance"].items())[:8]
    print("top features:", ", ".join(f"{k}={v}" for k, v in top))

    if args.post:
        res = post_signal(sig, url=args.webhook_url, always=args.always)
        print(f"webhook: {res}")


if __name__ == "__main__":
    main()
