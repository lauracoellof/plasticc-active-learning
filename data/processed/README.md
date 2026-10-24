# Final scientific results

This directory contains final experiment tables and the supporting scientific
provenance needed to interpret or verify them. Raw curves, cleaned per-object
tables, predictions and large feature caches are local reproduction inputs and
are excluded from Git.

| File family | Role | Used by |
|---|---|---|
| `resultados_clasificacion_150.csv`, `resultados_test_batch4.csv` | Held-out classifier variants and operational external evaluation | 04, 06, 10 |
| `calibracion_e2gp_oof_150_cv3.*` | Out-of-fold probability arrays and their scientific manifest | 06, 10 |
| `ablacion_causal_representacion_*`, `ablacion_bloques_aislados_*` and `.meta.json` | Per-seed ablations, matched object partitions, summaries and settings | 04b–04c |
| `anomalias_*` | Individual descriptor, fusion, complementarity and transfer results | 05, 10 |
| `triaje_batch4_10seeds.csv`, `dos_canales_batch4_10seeds.csv`, `comparacion_batch4_10seeds.csv` | Final single-batch trajectories and comparison | 06, 07b, 10 |
| `estrategias_consulta_batch4_5seeds.csv` | Separate five-seed query-strategy experiment | 06 |
| `generalizacion_por_batch.csv` | Four-batch, three-system, ten-seed generalization results | 06b |
| `triaje_batch99_10seeds.csv`, `dos_canales_batch99_10seeds.csv`, `comparacion_batch99_10seeds.csv` | Final combined-pool trajectories and comparison | 07b, 10 |
| `dos_canales_batch99_fino_10s.csv` | Completed ten-seed allocation sweep | 07b, Appendix C |
| `*_budget250_10seeds.csv`, `*_budget750_10seeds.csv`, `*presupuestos*` | Trajectories, summaries and paired budget comparisons | Appendix E |
| `sensibilidad_limpieza.csv`, `sensibilidad_gp1d_ls.csv` | Cleaning and GP-length-scale sensitivity | Appendix D |
| `comparacion_especificacion_e2gp.csv` | Regenerated 150/500-tree model-specification comparison | Appendix B |
| `08_direccion_features.csv` | Descriptive pre-calibration SHAP directions | 08 |
| `validacion_submuestreo_*` and `.manifest.json` | Subsample/complement tests, Monte Carlo summary and 200 draws | Appendix A, 10 |

The manifests, matched partitions and OOF arrays are part of scientific
reproducibility, rather than temporary validation work. Keep them with their
associated results. Superseded five-seed policy outputs, unfinished sweeps,
execution-attempt logs, transfer archives and review inventories are excluded.

Historical filenames and column identifiers are explained in
[DATA_SCHEMA.md](../../DATA_SCHEMA.md). A filename's `batch99` denotes the
combined eleven-batch pool; five seeds remain valid for the separate batch-4
query-strategy comparison.
