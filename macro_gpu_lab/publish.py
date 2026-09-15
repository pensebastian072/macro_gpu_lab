"""Publish the shadow advisory flag-file — the one artifact this lab surfaces.

Writes journal/flags/macro_gpu_state.json atomically (tmp + replace). Mirrors the
fail-safe contract of copper_brain/copper_brain/publish.py: read_state() returns a
neutral, vetoing-nothing dict if the file is missing, unparseable, or stale, and
NEVER raises — so any future HQ reader stays off the hot path and fail-safe.

Nothing here is enforced. `enforce` reflects MACRO_GPU_ENFORCE (default "no") and
every per-asset lean is advisory; promotion of a model is a separate env flip that
this build never performs automatically.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from . import config

NEUTRAL_FALLBACK = {
    "as_of": None,
    "stale": True,
    "enforce": "no",
    "promoted": {},
    "assets": {},
    "note": "fallback-neutral (flag missing/stale/corrupt)",
}


def build_state(rf_verdict: dict, torch_verdict: dict, signals: dict) -> dict:
    """Assemble the flag dict from the two model verdicts + latest RF signals.

    Per-asset lean is derived from the RF latest p_up (advisory reference bar):
    lean = +1 if mean p_up > 0.5 else -1, conviction = |mean p_up - 0.5| * 2.
    """
    def promoted(v):
        return bool(v.get("promoted")) and v.get("status") == "trained"

    assets = {}
    for a, ph in signals.items():
        ps = [p for p in ph.values() if p is not None]
        if not ps:
            continue
        mean_p = sum(ps) / len(ps)
        assets[a] = {
            "force": config.FORCE.get(a, "other"),
            "p_up": ph,
            "lean": 1 if mean_p > 0.5 else -1,
            "conviction": round(abs(mean_p - 0.5) * 2, 4),
        }

    return {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "stale": False,
        "enforce": config.MACRO_GPU_ENFORCE,     # advisory; "no" = shadow
        "promoted": {"rf": promoted(rf_verdict), "torch_mlp": promoted(torch_verdict)},
        "gate": {
            "rf": {h: g.get("passes") for h, g in (rf_verdict.get("horizons") or {}).items()},
            "torch_mlp": {h: g.get("passes") for h, g in (torch_verdict.get("horizons") or {}).items()},
        },
        "assets": assets,
        "note": "SHADOW research advisory — no model promoted enforces anything",
    }


def publish(state: dict) -> str:
    """Atomically write the flag-file."""
    tmp = config.FLAG_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(config.FLAG_PATH)
    return str(config.FLAG_PATH)


def read_state() -> dict:
    """Fail-safe reader (off hot path). Never raises; neutral on missing/stale."""
    fb = dict(NEUTRAL_FALLBACK)
    try:
        raw = json.loads(config.FLAG_PATH.read_text())
    except Exception:  # noqa: BLE001 — missing/corrupt -> neutral
        return fb
    as_of = raw.get("as_of")
    try:
        age = (datetime.now(timezone.utc)
               - datetime.fromisoformat(as_of)).total_seconds() / 86400.0
        if age > config.FLAG_STALE_DAYS:
            raw["stale"] = True
            raw["note"] = "stale (age > FLAG_STALE_DAYS)"
    except Exception:  # noqa: BLE001
        raw["stale"] = True
    return raw


# ── Copy-paste HQ reader stub (off hot path, fail-safe neutral) ─────
# def read_macro_gpu_state():
#     import json
#     from datetime import datetime, timezone
#     from pathlib import Path
#     FLAG = Path(r"C:\Users\<your-user>\macro_gpu_lab\journal\flags\macro_gpu_state.json")
#     STALE_DAYS = 4
#     fb = {"stale": True, "assets": {}, "enforce": "no"}
#     try:
#         raw = json.loads(FLAG.read_text())
#         age = (datetime.now(timezone.utc)
#                - datetime.fromisoformat(raw["as_of"])).total_seconds() / 86400.0
#         if age > STALE_DAYS:
#             raw["stale"] = True
#     except Exception:
#         return fb
#     return raw
