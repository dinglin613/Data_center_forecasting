"""Pick the tuning trial with the lowest mean validation CRPS over seeds (docs/protocol.md).

    python scripts/select_configs.py --results <dir> --out <dir>/selected

Writes <out>/<task>/<model>.json = {"override", "trial", "val_crps_mean", "val_crps_std",
"n_seeds", "n_trials", "trials": [...]}. A trial counts only if every search seed finished;
ties go to the lower trial id.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--results", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--min-seeds", type=int, default=2)
a = ap.parse_args()

for task_dir in sorted(Path(a.results).iterdir()):
    if not task_dir.is_dir() or task_dir.name == "selected":
        continue
    for model_dir in sorted(task_dir.iterdir()):
        recs = defaultdict(list)
        for p in model_dir.glob("trial*_seed*.json"):
            r = json.loads(p.read_text())
            recs[r["trial"]].append(r)
        if not recs:
            continue
        rows = []
        for trial, rs in sorted(recs.items()):
            c = [r["metrics"]["val"]["stage1"]["crps_kw"] for r in rs]
            rows.append({"trial": trial, "override": rs[0]["overrides"], "seeds": sorted(r["seed"] for r in rs),
                         "val_crps": c, "val_crps_mean": float(np.mean(c)),
                         "fit_seconds": float(np.mean([r["runtime"]["fit_seconds"] for r in rs]))})
        zero_shot = rs[0]["model"]["zero_shot"]
        need = 1 if zero_shot else a.min_seeds
        ok = [r for r in rows if len(r["seeds"]) >= need]
        if not ok:
            print(f"{task_dir.name} {model_dir.name}: no complete trial")
            continue
        best = min(ok, key=lambda r: (r["val_crps_mean"], r["trial"]))
        sel = {"override": best["override"], "trial": best["trial"], "val_crps_mean": best["val_crps_mean"],
               "val_crps_std": float(np.std(best["val_crps"])), "n_seeds": len(best["seeds"]),
               "n_trials": len(rows), "n_complete": len(ok), "trials": rows}
        out = Path(a.out) / task_dir.name / f"{model_dir.name}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(sel, indent=1))
        print(f"{task_dir.name:13s} {model_dir.name:18s} trial {best['trial']:2d} val crps "
              f"{best['val_crps_mean']:.3f} ({len(ok)}/{len(rows)} complete) {best['override']}")
