# Reproducing the analysis

## Environment

Use the pinned `environment.yml` (Python 3.11.15), or install `requirements.txt`
in an equivalent Python environment. Run `scripts/check_environment.py --full`
and `python -m pytest -q` before scientific execution. The saved classifier
requires its recorded scikit-learn version; the loader checks that version,
model checksum, representation version and feature count.

The analysis notebooks have completed execution in the declared Linux scientific
environment. This review combined fresh calculations with saved feature caches,
OOF arrays and AL trajectories. It did not regenerate every GP feature and every
acquisition trajectory from raw data. Synthetic tests validate invariants; a
passing test suite is not a repetition of the thesis experiments.

## Inputs

1. Obtain the fourteen source tables listed in [data/raw/README.md](data/raw/README.md).
2. Place them in `data/raw/` and run `python scripts/verify_raw_data.py`.
3. Run notebooks from the project root in the order below. The notebooks locate
   `config.py`; all configured paths are relative to that file.

Large feature caches and cleaned per-object tables are local artifacts under
`data/processed/`. They are omitted from Git. Existing compatible caches can be
reused by copying their CSV and `.meta.json` sidecars together. Integrity checks
remain enabled. `feature_cache_compatibility.json` records specifically reviewed
historical extractor fingerprints; a later source change invalidates that bridge.

## Execution order and cost

| Stage | Notebooks | Required computation |
|---|---|---|
| Prepare data | 01–02 | Raw-data exploration, cleaning and metadata alignment |
| Build representation | 03, 03b, 03c | Statistical features and GP fits; valid caches avoid full extraction |
| Train classifier | 04 | Holdouts, CV and the operational calibrated forest |
| Compare features | 04b–04c | Cached consistency checks, or explicit ten-seed ablation refits |
| Score and query | 05–06b | Anomaly descriptors, cached AL review, or explicit new AL experiments |
| Scale to combined pool | 07–07b | Validated combined-pool caches and saved multi-seed trajectories |
| Interpret and evaluate | 08–10 | SHAP, physical diagnostics and consolidated metrics |
| Supplementary checks | Appendices A–E | Subsampling, model specification, allocation and sensitivity |

Generate Appendix A before using its tables in notebook 10. Appendix B fits
alternative model specifications. Appendices C–E review saved experiments in
their default modes; their source experiments are separate, more expensive runs.

For a complete recalculation, enable the explicit `RECOMPUTE*` switches in the
relevant notebooks. Set `RECOMPUTE_FEATURES=True` in 03/03b and request fresh
extraction in 03c; regenerate OOF calibration with the command below. For 07,
`RECOMPUTE_POOL=True` reconstructs the combined pool from individual batches.
Missing saved AL trajectories raise an error in cached mode rather than silently
launching new experiments. Inspect each notebook's settings before running all.

```bash
python scripts/regenerate_e2gp_oof_calibration.py
python scripts/run_incremental_representation_ablation.py
python scripts/run_isolated_block_ablation.py
python scripts/regenerate_fine_sweep_10seeds.py
python scripts/regenerate_subsampling_validation.py
```

The saved model can be overwritten by notebook 04 or a requested model export.
Preserve its original bytes when comparing against the published classifier.
Use the model metadata's ordered feature list to construct every prediction
matrix; a matching set of columns in a different order is insufficient.

## Reading the results correctly

- Saved ten-seed comparisons use matched evaluation splits. Paired bootstrap
  intervals describe variability across those seeds on this finite pool; they
  are not population intervals for a future Rubin survey.
- OOF calibration arrays and their manifest define the calibrated reliability
  diagnostics. Aggregated OOF MF1 and the mean of fold MF1 values are different
  summaries. Holdout variant scores must not be labelled as CV scores.
- Appendix A uses 200 samples **without replacement**, not bootstrap samples.
  It compares the used subsample with its disjoint complement; continuous
  summaries are medians. Seven nonredundant tests receive Holm correction.
  Failure to reject does not establish equivalence.
- The uncalibrated comparison forest uses sklearn's default minimum leaf size
  of one; the operational calibrated model uses three. Five-fold external CV
  and the three/five internal calibration folds are distinct settings.
- Rank fusion preserves the historical ordinal `argsort` convention. Equal
  descriptor values do not receive average ranks; exact tie ordering may vary
  between NumPy builds. Use the pinned environment for direct comparisons.
- SHAP values refer to pre-calibration forests. Brightness proxies omit full
  extinction and K corrections and are used for descriptive interpretation.

Scientific manifests accompany the OOF arrays, ablations and subsampling outputs.
They belong with those results because they identify inputs, settings and
integrity checks. Temporary transfer archives, execution-attempt logs and
editorial review reports are not distributed in the public project.
