# Evaluation and tuning protocol

This protocol applies to every model on the DCGridBench leaderboard. The code that
implements it is `dcgridbench/tuning.py`, `scripts/make_jobs.py`, `scripts/select_configs.py`,
`scripts/run_benchmark.py` and `scripts/score_stages.py`.

## Data and splits

* Series: cluster GPU power of MIT Supercloud TX-Gaia, resampled per task (mean over the
  interval, left-labelled), in kW. Tasks: `mit-5min-1h`, `mit-15min-4h`, `mit-1h-24h`
  (`dcgridbench/data/windows.py`).
* Splits: train 2021-03-01 to 06-30, validation 2021-07-01 to 07-31, test 2021-08-01 to 09-30.
* An origin belongs to a split only if its whole target window lies in that split.
  Forecasts are made at every step (stride 1) for 9 quantile levels, 0.025 to 0.975.

## What counts as leakage

| Allowed | Not allowed |
|---|---|
| Training on targets of train origins | Any target of a val or test origin in training |
| Early stopping, checkpoint and hyperparameter selection on val | Any use of test data before the final scoring |
| Input for origin t: target and scheduler values at times <= t | Scheduler or target values after t |
| Calendar values (local hour, day of week) at the target times | Other future covariates |
| History before the train start as input context | Statistics fitted on val or test (scalers, regimes) |

Scheduler features are `gpus_alloc, jobs_running, gpus_queued, jobs_queued`. A model may
leave them out when validation says so (the `scheduler` knob below). Window scalers are
computed on the input window only. Regime labels use thresholds fitted on train.
The unit tests in `tests/test_external.py` check these rules: forecasts do not change when
data after the origin are replaced, and retraining with a perturbed test period gives the
same validation forecasts.

## Trained models: random search

Models: N-HiTS, TFT, PatchTST (NeuralForecast) and DeepAR (GluonTS).

* **Budget**: 20 configurations per model and task, each trained with 2 seeds (0, 1).
  Every trained model gets the same 20 draws of the shared space (`tuning.shared_trials`,
  seeded by the task name), mapped to its own parameter names by the `search:` section
  of its config.
* **Shared space**:

| Knob | Values |
|---|---|
| scheduler features | on, off |
| calendar features | on (fixed) |
| target | level, change from the origin value |
| scaling | standard, robust (DeepAR: mean scaling on, off) |
| size | small, medium, large (per model, see configs) |
| dropout | 0, 0.1, 0.2 |
| learning rate | log-uniform in [1e-4, 3e-3] |
| lookback | 5-min: 6/24/72 h; 15-min: 24/72/168 h; 1-h: 48/168/336 h |

* **Fixed training budget** (same for all): at most 2000 optimiser steps of 256 windows,
  validation every 25 steps (DeepAR: every 50), stop after 300 steps without improvement,
  keep the weights with the best validation loss.
* **Target "change"**: NeuralForecast models shift each window by its last observed value
  (scalers `last_std`, `last_robust`); DeepAR models first differences and its sampled
  paths are summed from the origin value.
* **Selection rule**: lowest mean validation CRPS over the 2 seeds; a configuration counts
  only if both seeds finished; ties go to the lower trial id. Test data are not touched.
* **Final runs**: the selected configuration with 3 seeds (0, 1, 2), trained exactly as in
  the search. Results report mean and standard deviation over seeds.

Knobs a model does not support are ignored, and the trial still counts towards the budget:
PatchTST in NeuralForecast takes no exogenous inputs; GluonTS DeepAR has no past-only
covariates, so scheduler features are not used.

## Zero-shot models

Chronos-Bolt (base) and TimesFM 2.5 are not trained. The only choice made on validation is
the context length (Chronos: 512, 1024, 2048; TimesFM: 512, 1024, 2048, 4096 steps).
They are deterministic, so one run is reported. Both predict quantiles 0.1 to 0.9 only;
levels 0.025, 0.05, 0.95 and 0.975 are extended by scaling the outer half-spread with the
normal quantile ratio (`base.to_task_quantiles`, rule `gaussian`).

## Reference baselines

Persistence and seasonal naive (empirical train residual quantiles), quantile LightGBM with
relative features and the quantile TCN use their fixed reference settings from
`dcgridbench/models/` and are not tuned. LightGBM and TCN are run with 3 seeds.

## Scoring

* Stage 1 and 2 (raw quantiles): CRPS, MAE of the median, pinball loss, 90% and 50% coverage,
  90% width and Winkler score, overall and per workload regime.
* Stage 2 (calibration): static, regime, online and online-regime conformal corrections,
  fitted on validation and updated causally through test (14-day window, hourly refresh).
* Stage 3 (scenarios, 3000 test origins, 200 scenarios): independent, Gaussian copula,
  bootstrap, scaled bootstrap, Schaake shuffle, and the model's own paths if it has them.
  Energy and variogram scores, exceedance Brier score, KS distance of the PIT of peak,
  ramp, energy and duration statistics.
* Stage 4 (reserve): newsvendor reserve with c_S = 1 and c_R/c_S in {0.05, 0.1, 0.2},
  per step from quantiles and as one block over the horizon from scenarios.

The leaderboard headline for stages 3 and 4 is the online regime-calibrated quantiles with
the Schaake shuffle; the native paths of DeepAR are listed as an extra generator.
