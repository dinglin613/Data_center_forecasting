"""MIT Supercloud (TX-Gaia) -> cluster-level GPU power series.

The raw data is CC BY-NC-ND 4.0, so we never redistribute it or anything derived
from it. Users download from the public S3 bucket and run this pipeline locally.

Pipeline:
  1. ``aggregate_folder``: stream each per-job GPU file (100 ms nvidia-smi samples),
     reduce it to per-minute job power, and write one parquet per folder.
     Raw files are never stored on disk.
  2. ``build_cluster_series``: sum job power per minute, add idle power for GPUs
     no job is using, and join scheduler features.
"""
from __future__ import annotations

import io
import re
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BUCKET = "https://mit-supercloud-dataset.s3.amazonaws.com"
PREFIX = "datacenter-challenge/202201"
N_FOLDERS = 100
TOTAL_GPUS = 448          # TX-Gaia: 224 nodes x 2 V100
IDLE_GPU_W = 40.0         # rough V100 idle draw; overridable
STEP_S = 60
LOCAL_TZ = "America/New_York"  # GPU telemetry clocks are local wall time; Slurm is UTC

GPU_COLS = ["timestamp", "gpu_index", "power_draw_W", "utilization_gpu_pct"]


def list_keys(prefix: str) -> list[tuple[str, int]]:
    """List (key, size) under an S3 prefix with anonymous access."""
    out, token = [], None
    while True:
        url = f"{BUCKET}/?list-type=2&max-keys=1000&prefix={urllib.parse.quote(prefix)}"
        if token:
            url += "&continuation-token=" + urllib.parse.quote(token)
        text = _get(url).decode()
        out += [(k, int(s)) for k, s in re.findall(r"<Key>([^<]+)</Key>.*?<Size>(\d+)<", text)]
        m = re.search(r"<NextContinuationToken>([^<]+)<", text)
        if not m:
            return out
        token = m.group(1)


def _get(url: str, retries: int = 5) -> bytes:
    for i in range(retries):
        try:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            return r.content
        except requests.RequestException:
            if i == retries - 1:
                raise
            time.sleep(2 ** i)
    raise RuntimeError("unreachable")


def parse_gpu_key(key: str) -> tuple[int, str]:
    """``.../gpu/0000/<jobid>-<node>.csv`` -> (jobid, node)."""
    stem = key.rsplit("/", 1)[-1].removesuffix(".csv")
    job, node = stem.split("-", 1)
    return int(job), node


def reduce_gpu_file(raw: bytes, job_id: int, node: str, step_s: int = STEP_S) -> pd.DataFrame:
    """Per-minute job power on one node: mean power per GPU, summed over GPUs.

    Taking the per-GPU mean first keeps the result correct when GPUs log at
    slightly different rates or a sample is missing.
    """
    df = pd.read_csv(io.BytesIO(raw), usecols=GPU_COLS)
    for c in GPU_COLS:  # a few files contain non-numeric junk such as "[N/A]"
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["timestamp", "power_draw_W", "gpu_index"])
    if df.empty:
        return pd.DataFrame()
    df["t"] = (df["timestamp"] // step_s * step_s).astype(np.int64)
    per_gpu = df.groupby(["t", "gpu_index"]).agg(
        p=("power_draw_W", "mean"), u=("utilization_gpu_pct", "mean"), n=("power_draw_W", "size")
    )
    per_t = per_gpu.groupby(level="t").agg(
        power_w=("p", "sum"), util_mean=("u", "mean"), n_gpus=("p", "size"), n_samples=("n", "sum")
    ).reset_index()
    per_t["id_job"] = job_id
    per_t["node"] = node
    return per_t


def aggregate_folder(folder: int, out_dir: Path, limit: int | None = None, workers: int = 16) -> Path:
    """Stream one of the 100 GPU folders into ``out_dir/gpu_<folder>.parquet``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"gpu_{folder:04d}.parquet"
    if out.exists():
        return out
    keys = [k for k, _ in list_keys(f"{PREFIX}/gpu/{folder:04d}/") if k.endswith(".csv")]
    if limit:
        keys = keys[:limit]
    def work(k):
        job, node = parse_gpu_key(k)
        try:
            return reduce_gpu_file(_get(f"{BUCKET}/{k}"), job, node), None
        except Exception as e:  # keep going; record failures for the report
            return None, (k, repr(e))

    parts, failed = [], []
    with ThreadPoolExecutor(workers) as ex:
        for df, err in ex.map(work, keys):
            (failed.append(err) if err else parts.append(df))
    df = pd.concat([p for p in parts if not p.empty], ignore_index=True) if parts else pd.DataFrame()
    tmp = out.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.rename(out)  # atomic: a parquet exists only if the folder finished
    if failed:
        pd.DataFrame(failed, columns=["key", "error"]).to_csv(out_dir / f"failed_{folder:04d}.csv", index=False)
    return out


def local_to_utc_epoch(t: np.ndarray, tz: str = LOCAL_TZ) -> np.ndarray:
    """Telemetry epochs encode US/Eastern wall time; convert to true UTC epochs.

    Verified against Slurm start/end times: offset is 5 h before 2021-03-14 and 4 h after.
    """
    uniq, inv = np.unique(t, return_inverse=True)
    naive = pd.to_datetime(uniq, unit="s")
    utc = naive.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward").tz_convert("UTC")
    fixed = (utc.asi8 // 10**9).astype(np.int64)
    bad = utc.isna()
    fixed[bad] = uniq[bad] + 5 * 3600  # ambiguous fall-back hour: use standard-time offset
    return fixed[inv]


# ---------------------------------------------------------------- scheduler

def _tres_gpus(s) -> int:
    if not isinstance(s, str):
        return 0
    m = re.search(r"100[12]=(\d+)", s)
    return int(m.group(1)) if m else 0


def load_gpu_jobs(slurm_csv: Path) -> pd.DataFrame:
    """GPU jobs from slurm-log.csv with start/end/allocated GPUs/labels."""
    df = pd.read_csv(slurm_csv, usecols=["id_job", "time_submit", "time_start", "time_end",
                                          "tres_alloc", "nodes_alloc", "state"])
    df["gpus"] = df["tres_alloc"].map(_tres_gpus)
    df = df[(df.gpus > 0) & (df.time_start > 0) & (df.time_end > df.time_start)]
    return df.drop(columns="tres_alloc").reset_index(drop=True)


def _interval_count(start, end, weight, grid) -> np.ndarray:
    """Sum of ``weight`` over intervals [start, end) active at each grid point."""
    d = np.zeros(len(grid) + 1)
    np.add.at(d, np.searchsorted(grid, start), weight)
    np.add.at(d, np.searchsorted(grid, end), -weight)
    return np.cumsum(d)[:-1]


def scheduler_features(jobs: pd.DataFrame, grid: np.ndarray) -> pd.DataFrame:
    """Features known at time t from the scheduler (no future information)."""
    one = np.ones(len(jobs))
    f = pd.DataFrame(index=grid)
    f["gpus_alloc"] = _interval_count(jobs.time_start.values, jobs.time_end.values, jobs.gpus.values, grid)
    f["jobs_running"] = _interval_count(jobs.time_start.values, jobs.time_end.values, one, grid)
    f["gpus_queued"] = _interval_count(jobs.time_submit.values, jobs.time_start.values, jobs.gpus.values, grid)
    f["jobs_queued"] = _interval_count(jobs.time_submit.values, jobs.time_start.values, one, grid)
    return f


def build_cluster_series(agg_dir: Path, slurm_csv: Path, step_s: int = STEP_S,
                         idle_gpu_w: float = IDLE_GPU_W, total_gpus: int = TOTAL_GPUS) -> pd.DataFrame:
    """Cluster GPU power per ``step_s`` with scheduler features.

    Columns:
      power_jobs_w  measured power of GPUs in jobs with telemetry
      power_w       power_jobs_w + idle power for GPUs not in any telemetry job
                    (an explicit modelling assumption, reported in the paper)
    """
    files = sorted(Path(agg_dir).glob("gpu_*.parquet"))
    agg = pd.concat([pd.read_parquet(f, columns=["t", "power_w", "n_gpus"]) for f in files])
    agg["t"] = local_to_utc_epoch(agg["t"].to_numpy())
    agg["t"] = agg["t"] // step_s * step_s
    # mean over minutes within a step, summed over jobs/nodes
    per = agg.groupby("t").agg(power_jobs_w=("power_w", "sum"), gpus_metered=("n_gpus", "sum"))
    n_sub = step_s // STEP_S if step_s >= STEP_S else 1
    per = per / n_sub

    jobs = load_gpu_jobs(slurm_csv)
    grid = np.arange(per.index.min(), per.index.max() + step_s, step_s)
    out = scheduler_features(jobs, grid).join(per, how="left")
    out[["power_jobs_w", "gpus_metered"]] = out[["power_jobs_w", "gpus_metered"]].fillna(0.0)
    idle = np.clip(total_gpus - out["gpus_metered"], 0, None) * idle_gpu_w
    out["power_w"] = out["power_jobs_w"] + idle
    out.index = pd.to_datetime(out.index, unit="s", utc=True)
    out.index.name = "time"
    return out
