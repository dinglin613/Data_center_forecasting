import numpy as np
import pandas as pd
from aidc_gridbench.data.mit_supercloud import local_to_utc_epoch, reduce_gpu_file, scheduler_features


def test_dst_offset():
    winter = pd.Timestamp("2021-02-12 22:54").value // 10**9
    summer = pd.Timestamp("2021-06-17 15:28").value // 10**9
    out = local_to_utc_epoch(np.array([winter, summer]))
    assert list(out - [winter, summer]) == [5 * 3600, 4 * 3600]


def test_reduce_sums_gpus_per_minute():
    csv = "timestamp,gpu_index,power_draw_W,utilization_gpu_pct\n" + "\n".join(
        f"{t},{g},{100 + 50 * g},10" for t in (0.0, 30.0, 61.0) for g in (0, 1))
    df = reduce_gpu_file(csv.encode(), 1, "n")
    assert list(df.t) == [0, 60]
    assert np.allclose(df.power_w, 250.0) and list(df.n_gpus) == [2, 2]


def test_scheduler_features_interval():
    jobs = pd.DataFrame({"time_submit": [0], "time_start": [60], "time_end": [180], "gpus": [2]})
    f = scheduler_features(jobs, np.array([0, 60, 120, 180]))
    assert list(f.gpus_alloc) == [0, 2, 2, 0]
    assert list(f.gpus_queued) == [2, 0, 0, 0]
