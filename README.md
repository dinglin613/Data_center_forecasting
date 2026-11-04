# aidc-gridbench

Benchmark and reference code for **grid-oriented probabilistic load forecasting of AI data centers**,
accompanying the paper *Grid-Oriented Probabilistic Load Forecasting for AI Data Centers*.

The benchmark builds a cluster-level GPU power series from public telemetry of the MIT Supercloud
TX-Gaia cluster (448 NVIDIA V100 GPUs, Feb.–Oct. 2021) and evaluates forecasts through four stages:

1. **Probabilistic forecasting**: persistence, seasonal naive, quantile LightGBM, quantile TCN
2. **Uncertainty calibration**: causal online conformal correction, optionally regime-conditioned
3. **Temporal-dependence recovery**: independent sampling, Gaussian copula, residual bootstrap, Schaake shuffle
4. **Reserve decision**: newsvendor reserve, per step and held over the horizon

Each stage has its own metrics (CRPS and coverage; energy and variogram scores and KS distances of
peak/ramp statistics; realised reserve cost and shortfall rate), so new components can be
evaluated in place.

## Data license

The MIT Supercloud dataset is licensed under **CC BY-NC-ND 4.0**. This repository does **not**
redistribute raw or processed data. The scripts below download the public data and rebuild every
series locally. Please cite the dataset:

> S. Samsi *et al.*, "The MIT Supercloud Dataset," *Proc. IEEE HPEC*, 2021. doi:10.1109/HPEC49654.2021.9622850

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # core + tests
pip install -e ".[deep]"         # adds PyTorch for the TCN
pytest -q                        # unit tests
```

## Benchmark tasks

| Task | Resolution | Horizon | Train | Validation | Test |
|---|---|---|---|---|---|
| `mit-5min-1h` | 5 min | 1 h (12 steps) | 2021-03-01 to 06-30 | July | 2021-08-01 to 09-30 |
| `mit-15min-4h` | 15 min | 4 h (16 steps) | same | same | same |
| `mit-1h-24h` | 1 h | 24 h (24 steps) | same | same | same |

A forecast origin belongs to a split only if its whole target window lies inside that split.
Task definitions are in `aidc_gridbench/data/windows.py`.

## Reproducing the results

The raw GPU telemetry is about 2.3 TB. `mit_aggregate.py` streams it from the public S3 bucket and
keeps only per-minute job power (about 1.2 GB), so the raw files are never stored.

```bash
# 1. scheduler log and metadata (~100 MB)
mkdir -p data/raw/mit_supercloud && cd data/raw/mit_supercloud
B=https://mit-supercloud-dataset.s3.amazonaws.com/datacenter-challenge/202201
for f in slurm-log.csv labelled_jobids.csv tres-mapping.txt LICENSE README.md; do curl -fsSO $B/$f; done
cd -

# 2. stream-aggregate the 100 GPU folders (Slurm array; or loop over --folders locally)
sbatch -A <account> scripts/slurm/mit_aggregate.sbatch
#   local alternative: python scripts/mit_aggregate.py --folders 0 1 2 ... --out data/interim/mit_gpu_1min

# 3. cluster GPU power series + data-quality report
python scripts/mit_build_series.py --agg $SCRATCH/aidc_gridbench/mit_gpu_1min

# 4. forecasting, calibration, scenarios and reserve
sbatch -A <account> scripts/slurm/cpu_pipeline.sbatch     # persistence, LightGBM
sbatch -A <account> scripts/slurm/tcn_calibration.sbatch  # quantile TCN, 3 seeds (GPU)

# 5. paper figures and tables
python paper/make_figures.py
python paper/make_tables.py      # writes the table bodies into paper/main.tex
```

Notes:
- Idle power of GPUs not assigned to any job is not in the telemetry; the series adds an assumed
  40 W per idle GPU (`IDLE_GPU_W` in `aidc_gridbench/data/mit_supercloud.py`).
- GPU telemetry timestamps are US Eastern wall-clock time; the pipeline converts them to UTC to
  match the scheduler log.

## Repository layout

```
aidc_gridbench/
  data/         MIT Supercloud pipeline, task definitions and splits
  models/       baselines, quantile TCN, conformal calibrators, workload regimes
  scenarios/    scenario generators
  metrics/      probabilistic, multivariate and grid-statistic scores
  decisions/    reserve sizing and realised cost
scripts/        command-line entry points and Slurm jobs
tests/          unit tests
paper/          LaTeX source, figure and table scripts
```

## Adding a method

A forecaster implements `fit(df, train_idx, task)` and `predict(df, idx, task) -> (n, H, K)`
quantiles at `task.taus`. Register it in `aidc_gridbench/models/baselines.py::MODELS` and run
`scripts/run_calibration.py --models <name>` and `scripts/run_scenarios.py --model <name>`.

## Citation

```bibtex
@inproceedings{lin2027gridoriented,
  author    = {Lin, Ding},
  title     = {Grid-Oriented Probabilistic Load Forecasting for {AI} Data Centers},
  booktitle = {Proc. IEEE PES General Meeting},
  year      = {2027},
  note      = {under review}
}
```

## License

Code: MIT (see `LICENSE`). Data: see the MIT Supercloud dataset license (CC BY-NC-ND 4.0).
