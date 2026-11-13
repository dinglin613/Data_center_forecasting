"""Add stage 2-4 metrics (calibration variants, scenarios, reserve) to final result records.

    python scripts/score_stages.py --results results/benchmark --task mit-1h-24h [--force]

Reads each seed<k>.json with its _stream.npz (and _samples.npz for native-path models) and
writes metrics.test.calibration / stage3 / stage4 into the same JSON. CPU only.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from dcgridbench.data.windows import TASKS
from dcgridbench.leaderboard import results as R
from dcgridbench.leaderboard.stages import score_stream

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results/benchmark")
ap.add_argument("--task", nargs="*", default=list(TASKS))
ap.add_argument("--model", nargs="*", default=None)
ap.add_argument("--force", action="store_true")
a = ap.parse_args()

for tname in a.task:
    task = TASKS[tname]
    for p in sorted((Path(a.results) / tname).glob("*/seed*.json")):
        if a.model and p.parent.name not in a.model:
            continue
        rec = json.loads(p.read_text())
        if rec.get("purpose", "final") != "final" or ("stage3" in rec["metrics"]["test"] and not a.force):
            continue
        st = dict(np.load(p.with_name(p.stem + "_stream.npz")))
        native = None
        sp = p.with_name(p.stem + "_samples.npz")
        if sp.exists():
            native = np.load(sp)["scen"].astype(float)
        res = score_stream(st, task, native=native)
        rec["metrics"]["test"].update(res)
        R.validate(rec)
        R.write(rec, p)
        s4 = res["stage4"]["online_regime"]["block"]["schaake"]["0.1"]["cost"]
        print(f"{tname} {p.parent.name} {p.stem}: block cost@0.1 (online_regime+schaake) {s4:.3f}"
              + (f", native raw {res['stage4']['raw']['block']['native']['0.1']['cost']:.3f}" if native is not None else ""),
              flush=True)
