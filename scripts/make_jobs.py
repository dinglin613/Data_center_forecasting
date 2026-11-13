"""Write job lists for tuning and final runs (one JSON object per line).

    python scripts/make_jobs.py tune  --models nhits deepar tft patchtst --out jobs/tune.jsonl
    python scripts/make_jobs.py zeroshot --models chronos_bolt_base timesfm --out jobs/zs.jsonl
    python scripts/make_jobs.py final --models nhits ... --selected <dir> --out jobs/final.jsonl
"""
import argparse
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from a checkout without install

from dcgridbench.data.windows import TASKS
from dcgridbench.models import registry
from dcgridbench.tuning import FINAL_SEEDS, SEARCH_SEEDS, trials_for

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["tune", "zeroshot", "final"])
ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--tasks", nargs="*", default=list(TASKS))
ap.add_argument("--selected", default=None, help="dir with <task>/<model>.json from select_configs.py")
ap.add_argument("--seeds", type=int, nargs="*", default=None)
ap.add_argument("--out", required=True)
a = ap.parse_args()

jobs = []
for task in a.tasks:
    for m in a.models:
        cfg = registry.load_config(m)
        if a.mode in ("tune", "zeroshot"):
            seeds = a.seeds or (SEARCH_SEEDS if a.mode == "tune" else (0,))
            for t in trials_for(cfg, task):
                for s in seeds:
                    jobs.append({"model": m, "task": task, "seed": s, "trial": t["trial"], "tune": True,
                                 "override": t["override"]})
        else:
            sel = Path(a.selected) / task / f"{m}.json" if a.selected else None
            seeds = a.seeds or ((0,) if cfg.get("deterministic") else FINAL_SEEDS)
            for s in seeds:
                j = {"model": m, "task": task, "seed": s}
                if sel is not None and sel.exists():
                    j["override_file"] = str(sel)
                jobs.append(j)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text("".join(json.dumps(j) + "\n" for j in jobs))
print(f"{len(jobs)} jobs -> {a.out}")
