# Result record schema (version 1.1)

Every benchmark run of one model on one task with one seed writes one JSON file:

```
results/benchmark/<task>/<model>/seed<k>.json         result record (this schema)
results/benchmark/<task>/<model>/seed<k>_stream.npz   raw val+test quantiles (not in git)
results/benchmark/<task>/<model>/seed<k>_samples.npz  native test paths on the scenario subset
results/benchmark/<task>/<model>/trial<j>_seed<k>.json  tuning trial (val metrics only)
```

Version 1.1 adds `purpose`, `overrides`, `trial`, `selection` and the stage 2-4 objects
`metrics.test.calibration`, `stage3`, `stage4`, `stage34_info` (written by
`scripts/score_stages.py`). Readers must ignore keys they do not know.

Records are produced by `scripts/run_benchmark.py` through
`dcgridbench.leaderboard.results.make_record`, which also validates them
(`results.validate`). The leaderboard is built only from these files.

All power quantities are in **kW**. Metrics are computed with `dcgridbench.metrics.scores`
on every origin of the split (stride 1), all horizons pooled.

## Top level

| Key | Type | Meaning |
|---|---|---|
| `schema_version` | str | `"1.1"` |
| `purpose` | str | `final` (leaderboard run) or `tuning` (search trial; no test metrics) |
| `task` | str | task name, e.g. `mit-1h-24h` |
| `task_spec` | object | `resolution`, `horizon`, `stride`, `taus` (quantile levels), `unit` |
| `model` | object | see below |
| `seed` | int | random seed of this run |
| `splits` | object | per split (`train`, `val`, `test`): `period` [start, end] and `n_origins` |
| `metrics` | object | `val` and `test` (final runs), each with `stage1` and `stage2`; `test` also has the stage 2-4 objects below |
| `overrides` | object | parameters set by the tuning trial or the selected configuration |
| `trial` | int | tuning trial id (tuning records) |
| `selection` | object | final runs of tuned models: selected `trial`, `val_crps_mean`, `val_crps_std`, `n_seeds`, `n_trials`, `n_complete` |
| `runtime` | object | `fit_seconds`, `predict_seconds` (val+test origins), `best_val_loss` (or null) |
| `predictions_file` | str or null | path of the `_stream.npz` file |
| `environment` | object | `python`, `torch`, `device` (GPU name or `cpu`), `host` |
| `code_commit` | str or null | git commit of the benchmark code |
| `created_utc` | str | ISO time stamp |
| `regime_share_test` | object | share of test origins per workload regime |
| `native_samples` | object | optional: `n_samples`, `n_origins`, `sample_seconds` |

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
| `future_known` | list | calendar features given for the target times |
| `notes` | str | short note shown on the leaderboard (config `notes`) |
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

## `metrics.test.calibration` (stage 2, conformal variants)

One object per variant (`raw`, `static`, `regime`, `online`, `online_regime`), each holding
`stage1` and `stage2` as above for the calibrated test quantiles
(`dcgridbench.leaderboard.stages.calibration_variants`, settings of `run_calibration.py`).

## `metrics.test.stage3` (scenarios)

`stage3[<variant>][<generator>]` for variants `raw` and `online_regime` and generators
`independent`, `copula`, `bootstrap`, `scaled_bootstrap`, `schaake`, plus `native` (the
model's own paths, raw variant only):

| Key | Meaning |
|---|---|
| `energy_score` | energy score of the scenarios (kW) |
| `variogram_score` | variogram score, order 0.5 |
| `exceed_brier` | Brier score of exceeding the val 90% load quantile within the horizon |
| `pit_ks` | KS distance from uniform of the PIT of `peak`, `ramp_up`, `ramp_down`, `energy`, `dur_above`, `exceed` |

## `metrics.test.stage4` (reserve)

`stage4[<variant>]["per_step"][<c_R>]` (from quantiles) and
`stage4[<variant>]["block"][<generator>][<c_R>]` (one reserve over the horizon from
scenarios), with c_S = 1 and c_R in `0.05`, `0.1`, `0.2`; each holds `cost`, `reserve_kw`,
`shortfall_prob`, `shortfall_kw` (`dcgridbench.decisions.reserve`).

`stage34_info`: `n_origins` (fixed subset of 3000 test origins), `n_scenarios` (200),
`threshold_kw`, `cost_ratios`, `c_s`.

## `_stream.npz`

Same layout as `scripts/run_calibration.py`, so `scripts/run_scenarios.py` can read it:
`q` (n_val + n_test, H, K) float32 quantiles, `y` targets, `t` origin positions, `n_val`,
`reg` regime labels, plus `origins` (UTC ns) and `taus`.

## Protocol recorded by a record

* Training uses train origins only; early stopping and checkpoint selection use val only.
* The input for origin t contains rows up to and including t; the only future-known inputs
  are calendar features at the target times (docs/protocol.md).
* Zero-shot models are not fitted; their context is at most `context_steps` rows before the origin.
