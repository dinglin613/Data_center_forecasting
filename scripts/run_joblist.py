"""Run a slice of a job list (one JSON object per line) with a few processes in parallel.

Each job is {"model", "task", "seed", optional "trial", "tune", "override", "override_file"}.
Jobs whose result JSON already exists are skipped, so a list can be resubmitted.

    python scripts/run_joblist.py --jobs jobs.jsonl --start 0 --count 8 --par 4 --out <dir> --series <parquet>
"""
import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--jobs", required=True)
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--count", type=int, default=None)
ap.add_argument("--par", type=int, default=1)
ap.add_argument("--out", required=True)
ap.add_argument("--series", required=True)
ap.add_argument("--logs", default="logs/jobs")
a = ap.parse_args()

lines = [json.loads(x) for x in Path(a.jobs).read_text().splitlines() if x.strip()]
jobs = lines[a.start: None if a.count is None else a.start + a.count]
Path(a.logs).mkdir(parents=True, exist_ok=True)


def target(j):
    stem = f"trial{j['trial']}_seed{j['seed']}" if j.get("tune") else f"seed{j['seed']}"
    return Path(a.out) / j["task"] / j.get("name", j["model"]) / f"{stem}.json"


def run(j):
    if target(j).exists():
        return f"skip {target(j)}"
    cmd = [sys.executable, "scripts/run_benchmark.py", "--model", j["model"], "--task", j["task"],
           "--seed", str(j["seed"]), "--out", a.out, "--series", a.series]
    if j.get("tune"):
        cmd += ["--tune", "--trial", str(j["trial"])]
    if j.get("override"):
        cmd += ["--override", json.dumps(j["override"])]
    if j.get("override_file"):
        cmd += ["--override-file", j["override_file"]]
    log = Path(a.logs) / (target(j).relative_to(a.out).as_posix().replace("/", "__") + ".log")
    t0 = time.time()
    with open(log, "w") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
    return f"rc={rc} {time.time() - t0:.0f}s {target(j)}"


with ThreadPoolExecutor(a.par) as ex:
    for msg in ex.map(run, jobs):
        print(msg, flush=True)
