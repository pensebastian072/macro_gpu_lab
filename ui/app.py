"""Standalone read-only viewer for the macro_gpu_lab bench.

Binds 127.0.0.1 only. There is no POST route and nothing here can act.

Data source, in order of preference:
  1. journal/flags/macro_gpu_state.json   - the live flag, if you have run the bench
  2. ui/snapshot/macro_gpu_state.json     - the committed snapshot shipped with the repo

The fallback is the point: a fresh clone on any machine renders the same per-asset
leans and gate verdicts without needing market data, an API key, or a GPU. The
page states which source it is showing and as of when.

Staleness is RECOMPUTED here from `as_of` rather than trusting the `stale` field
baked into the file. That field records what was true when the flag was written,
which is not the same question as whether it is old now.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from flask import Flask, jsonify, render_template

REPO_ROOT = Path(__file__).resolve().parent.parent
LIVE_FLAG = REPO_ROOT / "journal" / "flags" / "macro_gpu_state.json"
SNAPSHOT = Path(__file__).resolve().parent / "snapshot" / "macro_gpu_state.json"

STALE_DAYS = 4

app = Flask(__name__)


def _parse_ts(value) -> datetime | None:
    try:
        ts = datetime.fromisoformat(str(value))
    except Exception:
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def _load_state() -> dict:
    """Live flag if present and parseable, else the shipped snapshot.

    Never raises. Missing or corrupt input degrades to a neutral dict rather
    than a 500 - the same fail-safe contract the flag file itself uses.
    """
    for path, source in ((LIVE_FLAG, "live"), (SNAPSHOT, "snapshot")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict):
            data["_source"] = source
            data["_source_path"] = str(path.relative_to(REPO_ROOT))
            return data
    return {
        "_source": "none",
        "_source_path": "",
        "stale": True,
        "enforce": "no",
        "promoted": {},
        "gate": {},
        "assets": {},
        "note": "no flag file and no snapshot found",
    }


def _recompute_staleness(state: dict) -> None:
    """Age the flag against the clock, not against its own stored opinion."""
    ts = _parse_ts(state.get("as_of"))
    if ts is None:
        state["stale"] = True
        state["age_days"] = None
        return
    age = (datetime.now(timezone.utc) - ts).total_seconds() / 86400.0
    state["age_days"] = round(age, 1)
    state["stale"] = age > STALE_DAYS


def _asset_rows(state: dict) -> list[dict]:
    """assets is symbol -> {force, p_up{h}, lean, conviction}; flatten for a table."""
    rows = []
    for symbol, v in (state.get("assets") or {}).items():
        if not isinstance(v, dict):
            continue
        p_up = v.get("p_up") or {}
        rows.append({
            "symbol": symbol,
            "force": v.get("force", "?"),
            "p_up_5d": p_up.get("5d"),
            "p_up_21d": p_up.get("21d"),
            "lean": v.get("lean"),
            "conviction": v.get("conviction"),
        })
    rows.sort(key=lambda r: (r["force"], r["symbol"]))
    return rows


def _gate_rows(state: dict) -> list[dict]:
    rows = []
    promoted = state.get("promoted") or {}
    for model, horizons in (state.get("gate") or {}).items():
        if not isinstance(horizons, dict):
            continue
        for horizon, passes in horizons.items():
            rows.append({
                "model": model,
                "horizon": horizon,
                "passes": bool(passes),
                "promoted": bool(promoted.get(model)),
            })
    rows.sort(key=lambda r: (r["model"], str(r["horizon"])))
    return rows


@app.route("/")
def page_index():
    return render_template("index.html")


@app.route("/api/macro_gpu")
def api_macro_gpu():
    state = _load_state()
    _recompute_staleness(state)
    promoted = state.get("promoted") or {}
    state["asset_rows"] = _asset_rows(state)
    state["gate_rows"] = _gate_rows(state)
    state["n_assets"] = len(state["asset_rows"])
    state["n_models"] = len(promoted)
    state["n_promoted"] = sum(1 for v in promoted.values() if v)
    state["forces"] = sorted({r["force"] for r in state["asset_rows"]})
    return jsonify(state)


@app.route("/health")
def health():
    return jsonify({"ok": True, "ts": datetime.now(timezone.utc).isoformat()})


def main() -> None:
    # 127.0.0.1 only. Never 0.0.0.0, never tunnelled.
    app.run(host="127.0.0.1", port=8102, debug=False)


if __name__ == "__main__":
    main()
