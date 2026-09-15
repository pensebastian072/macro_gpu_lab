"""Robinhood advisory bridge — read-only aggregator for real-money context.

Aggregates this repo's own signal (tv_signal.json / macro_gpu_state.json) with
HQ Trading System's paper macro brain (macro_state.json), news veto
(veto_flag.json), gate verdicts (advisory_report.py), cell health, and the
latest daily review into one normalized JSON. Broker-agnostic by design: no
Robinhood API/creds here, no order placement, no writes outside this repo's
own journal/flags/. Whoever has live Robinhood MCP access at the time (Codex
CLI or a Claude Code session — see HANDOFF.md §7/§9 and the
robinhood-trading-mcp memory) layers real account numbers + kill-switch
enforcement on top of this output; that half deliberately does not live here
so this script stays credential-free and shareable between executors.

Fail-safe throughout: a missing/corrupt/unreachable source degrades that
section to `null` + an entry in "flags", never raises.

Run: python -m macro_gpu_lab.robinhood_advisory [--json] [--days N] [--min-trades N]
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from . import config

HQ_ROOT = Path(r"C:\Users\<your-user>\hq-trading-system")
HQ_PYTHON = HQ_ROOT / ".venv" / "Scripts" / "python.exe"
HQ_MACRO_STATE = HQ_ROOT / "journal" / "macro" / "macro_state.json"
HQ_VETO_FLAG = HQ_ROOT / "journal" / "sentiment" / "veto_flag.json"
HQ_SCORECARDS_DIR = HQ_ROOT / "journal" / "scorecards"
HQ_REVIEWS_DIR = HQ_ROOT / "journal" / "reviews"
HQ_CELL_HEALTH_LOG = HQ_ROOT / "journal" / "shadow_paper" / "cell_health_svc.log"
HQ_ADVISORY_REPORT = HQ_ROOT / "analytics" / "advisory_report.py"

TV_SIGNAL_PATH = config.FLAGS_DIR / "tv_signal.json"
OUT_PATH = config.FLAGS_DIR / "robinhood_advisory_state.json"

# macro_engine's hourly corr-alignment bug (mixed :00/:30-anchored bars silently
# starving half the correlation matrix) was fixed 2026-07-04 — HQ macro_state
# data from before this date under-covers cross-asset correlations/divergences.
ALIGNMENT_FIX_DATE = "2026-07-04"

# Ratios, not HQ's absolute paper-capital dollar figures (HQ is sized to its
# $10k paper account) — apply against the REAL Robinhood account's own
# portfolio value / today's own PnL, whatever that account's actual size is.
KILL_SWITCH_RULES_MIRRORED = {
    "daily_loss_pct_cap": 0.03,
    "max_agg_margin_pct_cap": 0.60,
    "stale_or_unknown_regime_action": "flat / no new risk",
    "no_stop_no_trade": True,
}


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as e:  # noqa: BLE001 — fail-safe, never raise
        return None, f"unavailable: {e}"


def _tail_lines(path: Path, n: int = 20):
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return lines[-n:], None
    except Exception as e:  # noqa: BLE001
        return [], f"unavailable: {e}"


def _latest_review_excerpt(head_lines: int = 40):
    try:
        files = sorted(HQ_REVIEWS_DIR.glob("*.md"))
        if not files:
            return None, "unavailable: no review files found"
        latest = files[-1]
        text = latest.read_text(encoding="utf-8", errors="replace").splitlines()
        return {"file": latest.name, "excerpt": "\n".join(text[:head_lines])}, None
    except Exception as e:  # noqa: BLE001
        return None, f"unavailable: {e}"


def _latest_scorecards():
    out = {}
    try:
        for f in sorted(HQ_SCORECARDS_DIR.glob("*.json")):
            data, err = _read_json(f)
            out[f.stem] = data if data is not None else {"error": err}
        return out, None
    except Exception as e:  # noqa: BLE001
        return {}, f"unavailable: {e}"


def _gate_verdicts(days: int, min_trades: int):
    if not HQ_ADVISORY_REPORT.exists() or not HQ_PYTHON.exists():
        return None, "unavailable: advisory_report.py or HQ venv not found on this box"
    try:
        proc = subprocess.run(
            [str(HQ_PYTHON), str(HQ_ADVISORY_REPORT),
             "--days", str(days), "--min-trades", str(min_trades), "--json"],
            cwd=str(HQ_ROOT / "analytics"),
            capture_output=True, text=True, timeout=60,
        )
        out = proc.stdout
        marker = '{\n  "days"'
        idx = out.find(marker)
        if idx == -1:
            return None, "unavailable: no JSON marker found in advisory_report output"
        return json.loads(out[idx:]), None
    except Exception as e:  # noqa: BLE001
        return None, f"unavailable: {e}"


def _is_pre_alignment_fix(data_through: str | None) -> bool | None:
    if not data_through:
        return None
    try:
        return data_through < ALIGNMENT_FIX_DATE
    except Exception:  # noqa: BLE001
        return None


def build_state(days: int = 30, min_trades: int = 30) -> dict:
    flags: list[str] = []

    tv_signal, tv_err = _read_json(TV_SIGNAL_PATH)
    if tv_err:
        flags.append(f"macro_gpu tv_signal.json {tv_err}")

    macro_gpu_state, mgs_err = _read_json(config.FLAG_PATH)
    if mgs_err:
        flags.append(f"macro_gpu_state.json {mgs_err}")

    hq_macro_regime, hqm_err = _read_json(HQ_MACRO_STATE)
    pre_fix = None
    if hqm_err:
        flags.append(f"HQ macro_state.json {hqm_err}")
    else:
        as_of = hq_macro_regime.get("generated_at") or hq_macro_regime.get("as_of")
        pre_fix = _is_pre_alignment_fix(as_of[:10] if as_of else None)
        if pre_fix:
            flags.append(
                "HQ macro_state.json predates the 2026-07-04 corr-alignment fix "
                "- correlations/divergences likely under-covered, treat as unreliable"
            )
        if hq_macro_regime.get("stale"):
            flags.append("HQ macro_state.json marked stale")

    hq_veto, veto_err = _read_json(HQ_VETO_FLAG)
    if veto_err:
        flags.append(f"HQ veto_flag.json {veto_err}")

    hq_scorecards, sc_err = _latest_scorecards()
    if sc_err:
        flags.append(f"HQ scorecards {sc_err}")

    hq_gate_verdicts, gv_err = _gate_verdicts(days, min_trades)
    if gv_err:
        flags.append(f"HQ gate verdicts {gv_err}")
    elif hq_gate_verdicts:
        for row in hq_gate_verdicts.get("gates", []):
            if row.get("verdict") is None or (
                isinstance(row.get("verdict"), str) and "NO VERDICT" in row["verdict"]
            ):
                flags.append(f"gate '{row.get('gate')}' n={row.get('n')} - NO VERDICT (insufficient data)")

    cell_health_tail, ch_err = _tail_lines(HQ_CELL_HEALTH_LOG, n=20)
    if ch_err:
        flags.append(f"HQ cell_health log {ch_err}")

    review_excerpt, rev_err = _latest_review_excerpt()
    if rev_err:
        flags.append(f"HQ latest review {rev_err}")

    if hq_veto and any(
        (a.get("recommended_action", {}).get("long_mult", 1) < 1
         or a.get("recommended_action", {}).get("short_mult", 1) < 1)
        for a in (hq_veto.get("assets") or {}).values()
    ):
        flags.append("HQ news veto has at least one asset scaled below full size")

    return {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "overall_status": "ADVISORY_ONLY - zero strategies have cleared PBO/DSR anywhere in this stack",
        "macro_gpu_signal": tv_signal,
        "macro_gpu_state": macro_gpu_state,
        "hq_macro_regime": hq_macro_regime,
        "hq_macro_regime_pre_alignment_fix": pre_fix,
        "hq_news_veto": hq_veto,
        "hq_scorecards": hq_scorecards,
        "hq_gate_verdicts": hq_gate_verdicts,
        "hq_cell_health_tail": cell_health_tail,
        "hq_latest_review_excerpt": review_excerpt,
        "kill_switch_rules_mirrored": KILL_SWITCH_RULES_MIRRORED,
        "flags": flags,
    }


def publish(state: dict) -> str:
    """Atomically write the aggregated state, mirroring publish.py's fail-safe writer."""
    tmp = OUT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str))
    tmp.replace(OUT_PATH)
    return str(OUT_PATH)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--min-trades", type=int, default=30)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    state = build_state(args.days, args.min_trades)
    path = publish(state)
    print(f"wrote {path}")
    if args.json:
        print(json.dumps(state, indent=2, default=str))
    else:
        print(state["overall_status"])
        for f in state["flags"]:
            print(f" - {f}")


if __name__ == "__main__":
    main()
