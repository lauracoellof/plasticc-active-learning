# Scientific commands

Run commands from the repository root after activating the declared environment.
The notebooks provide the full scientific narrative; these commands regenerate
specific experiments or reusable figures.

| Command | Purpose | Inputs and cost |
|---|---|---|
| `check_environment.py --full` | Check declared versions and runtime imports | No scientific data required |
| `verify_raw_data.py` | Check the fourteen source-file hashes | Streams the local raw dataset |
| `verify_models.py` | Check saved-model integrity | Reads the model and metadata |
| `regenerate_e2gp_oof_calibration.py` | Refit out-of-fold calibration diagnostics | Training features; forest fits and CV |
| `run_incremental_representation_ablation.py` | Compare cumulative feature blocks on paired splits | Training caches; ten-seed refitting |
| `run_isolated_block_ablation.py` | Compare isolated feature blocks on paired splits | Training caches; ten-seed refitting |
| `review_saved_ablations.py` | Shared cached-result checks used by 04b/04c | Imported by the notebooks; no refitting |
| `analyze_representation_sensitivity.py` | Recompute cleaning and GP-length-scale sensitivity checks | Raw curves and feature extraction |
| `regenerate_fine_sweep_10seeds.py` | Run the full ten-seed allocation sweep | Training/pool caches; repeated AL fits, no partial input required |
| `regenerate_subsampling_validation.py` | Regenerate the finite-population diagnostic | Full-batch caches and canonical model; 200 draws |
| `extract_pool_anomaly_curves.py` | Collect raw curves for selected pool anomalies | Streams raw batches for interpretation |

`generate_*_figure*.py`, `generate_validation_figures.py` and
`generate_two_channel_architecture.py` create figures from the corresponding
inputs/results. They write under `reports/figures/`; some feature figures also
fit illustrative GPs or load the canonical classifier. See each script's
`main()` and argument parser before launching it.

Commands that regenerate experiments write their named result files. Keep a
separate copy when intentionally comparing a new configuration with the
published one. Full allocation sweeps can take substantially longer than a
cached notebook review.
