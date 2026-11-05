# Result record schema (version 1.0)

Every benchmark run of one model on one task with one seed writes one JSON file:

```
results/benchmark/<task>/<model>/seed<k>.json         result record (this schema)
results/benchmark/<task>/<model>/seed<k>_stream.npz   raw val+test quantiles (not in git)
results/benchmark/<task>/<model>/seed<k>_samples.npz  test sample paths, native-sample models only
```

Records are produced by `scripts/run_benchmark.py` through
`dcgridbench.leaderboard.results.make_record`, which also validates them
(`results.validate`). The leaderboard is built only from these files.

All power quantities are in **kW**. Metrics are computed with `dcgridbench.metrics.scores`
on every origin of the split (stride 1), all horizons pooled.

## Top level

| Key | Type | Meaning |
|---|---|---|
| `schema_version` | str | `"1.0"` |
| `task` | str | task name, e.g. `mit-1h-24h` |
| `task_spec` | object | `resolution`, `horizon`, `stride`, `taus` (quantile levels), `unit` |
| `model` | object | see below |
| `seed` | int | random seed of this run |
| `splits` | object | per split (`train`, `val`, `test`): `period` [start, end] and `n_origins` |
| `metrics` | object | `val` and `test`, each with `stage1` and `stage2` (below) |
| `runtime` | object | `fit_seconds`, `predict_seconds` (val+test origins), `best_val_loss` (or null) |
| `predictions_file` | str or null | path of the `_stream.npz` file |
| `environment` | object | `python`, `torch`, `device` (GPU name or `cpu`), `host` |
| `code_commit` | str or null | git commit of the benchmark code |
| `created_utc` | str | ISO time stamp |
| `regime_share_test` | object | share of test origins per workload regime |
| `native_samples` | object | optional: `n_samples`, `sample_seconds`, `energy_score_kw` |

## `model`

| Key | Type | Meaning |
|---|---|---|
| `name` | str | registry name (config file stem) |
| `display_name` | str | name shown on the leaderboard |
| `family` | str | `baseline`, `cnn`, `rnn`, `mlp`, `transformer`, `foundation`, ... |
| `adapter` | str | adapter key or `module:Class` |
| `config_file` | str | YAML config used |
| `reference` | str or null | paper reference |
| `leaderboard` | bool | false for diagnostic configs that must not be ranked |
| `library` | str | pip distribution providing the model (`dcgridbench` for built-ins) |
| `library_version` | str or null | installed version |
| `hyperparameters` | object | resolved config parameters for this task |
| `zero_shot` | bool | true if no training on this benchmark |
| `native_samples` | bool | true if the model outputs sample paths |
| `exogenous` | list | scheduler features used as past-only inputs (external models) |
| `best_val_loss`, `best_step` | number or null | early-stopping checkpoint (NeuralForecast) |

## `metrics.<split>.stage1` (forecast accuracy)

| Key | Meaning |
|---|---|
| `crps_kw` | CRPS from the quantiles, 2 x mean pinball over `taus` |
| `mae_kw` | MAE of the median |
| `pinball_kw` | mean pinball loss |
| `crps_by_regime` | `crps_kw` per regime (`low`, `calm`, `active`, `volatile`) |

## `metrics.<split>.stage2` (calibration of the raw quantiles)

| Key | Meaning |
|---|---|
| `cov90` | share of targets inside [q0.05, q0.95] (nominal 0.90) |
| `cov50` | share inside [q0.25, q0.75] (nominal 0.50) |
| `width90_kw` | mean width of the 90% interval |
| `winkler90` | Winkler (interval) score of the 90% interval |
| `cov90_by_regime`, `width90_by_regime` | per regime |

Regimes are those of `dcgridbench.models.regimes.Regimes`, fitted on train and known at the origin.
Stages 3 (scenarios) and 4 (reserve decisions) will be added under the same `metrics.<split>`
objects as `stage3` and `stage4`; readers must ignore keys they do not know.

## `_stream.npz`

Same layout as `scripts/run_calibration.py`, so `scripts/run_scenarios.py` can read it:
`q` (n_val + n_test, H, K) float32 quantiles, `y` targets, `t` origin positions, `n_val`,
`reg` regime labels, plus `origins` (UTC ns) and `taus`.

## Protocol recorded by a record

* Training uses train origins only; early stopping and checkpoint selection use val only.
* The input for origin t contains rows up to and including t; no future covariates.
* Zero-shot models are not fitted; their context is at most `context_steps` rows before the origin.
