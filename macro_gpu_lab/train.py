"""Orchestrator: panel -> features -> labels -> {RF, torch} gate -> scorecard -> flag.

    .venv\\Scripts\\python.exe -m macro_gpu_lab.train --once
    .venv\\Scripts\\python.exe -m macro_gpu_lab.train --once --rebuild   # refetch panel first

Writes a per-model scorecard under journal/scorecards/ and the shadow advisory
flag under journal/flags/. Both models stay SHADOW unless EVERY horizon clears
the gate — a correct gate usually FAILS the first model, which is the honest
result, not a bug.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from . import config
from . import data as data_mod
from . import publish
from .features import build_features, _to_common_grid
from .labels import long_labels
from .models import baseline_rf, torch_seq


def _write_scorecard(verdict: dict) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    p = config.SCORECARDS_DIR / f"macro_gpu_{verdict['model']}_{stamp}.json"
    p.write_text(json.dumps(verdict, indent=2, default=str))
    return str(p)


def _print_verdict(v: dict) -> None:
    print(f"\n=== {v['model']} : {v['status']} : "
          f"{'PROMOTED' if v.get('promoted') else 'SHADOW'} ===")
    for h, g in (v.get("horizons") or {}).items():
        dsr = (g.get("deflated_sharpe") or {}).get("ratio")
        print(f"  {h}: pass={g['passes']}  DSR_ratio={dsr}  pbo={g['pbo']}  "
              f"PF={g['profit_factor']}  n={g['n_trades']}  {g['reasons'][0]}")


def run_once(rebuild: bool) -> dict:
    panel = None if rebuild else data_mod.load_latest_panel()
    if panel is None:
        print("building panel ...")
        panel, meta = data_mod.build_panel()
        if not panel.empty:
            data_mod.save_panel(panel, meta)
    if panel is None or panel.empty:
        print("no panel — aborting")
        return {}

    print(f"panel: {panel.shape[1]} assets x {panel.shape[0]} rows "
          f"({panel.index.min().date()} .. {panel.index.max().date()})")
    grid = _to_common_grid(panel)   # one trading calendar for features AND labels
    feat = build_features(grid)
    labels = long_labels(grid).reindex(feat.index)
    print(f"features: {len(feat)} (date,asset) rows x "
          f"{feat.shape[1] - 2} numeric cols")

    rf = baseline_rf.train_and_validate(feat, labels)
    _write_scorecard(rf)
    _print_verdict(rf)

    torch_v = torch_seq.train_and_validate(feat, labels)
    _write_scorecard(torch_v)
    _print_verdict(torch_v)

    signals = baseline_rf.latest_signals(feat, labels)
    state = publish.build_state(rf, torch_v, signals)
    fp = publish.publish(state)
    print(f"\nflag -> {fp}  (enforce={state['enforce']}, "
          f"promoted={state['promoted']})")
    return {"rf": rf, "torch_mlp": torch_v, "flag": fp}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="run one full cycle")
    ap.add_argument("--rebuild", action="store_true", help="refetch the panel first")
    args = ap.parse_args()
    run_once(rebuild=args.rebuild)


if __name__ == "__main__":
    main()
