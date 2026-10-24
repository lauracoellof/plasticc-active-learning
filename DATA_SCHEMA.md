# Stable data and experiment identifiers

The code narrative, documentation and messages are in English. Some saved CSV
columns, feature names and protocol values retain their original identifiers.
Keeping these identifiers stable preserves compatibility with the trained model,
cache manifests and experiment trajectories.

| Identifier | Meaning |
|---|---|
| `object_id` | PLAsTiCC object identifier; never a predictive feature |
| `target`, `true_target` | Training or revealed simulation class label |
| `prediccion`, `confianza` | Predicted class and maximum calibrated probability |
| `pred`, `proba_max`, `novelty` | Canonical prediction-table aliases: class, maximum probability and one minus that probability |
| `semilla` | Random seed identifying matched experiment repetitions |
| `iter`, `n_lab` | Acquisition round and number of known-class labelled objects |
| `MF1`, `WF1` | Macro- and class-frequency-weighted F1 |
| `n_anomalias` | Cumulative queried anomaly discoveries |
| `n_descubiertas` | Discovery-count column used by the two-channel trajectories |
| `pureza_Ia`, `completitud_Ia` | Precision and recall for the SNIa label |
| `recall_raras` | Mean recall over the configured rare known classes |
| `frac_desc` | Fraction of query budget allocated to discovery |
| `q_dudoso` | Queries from the intermediate-confidence region |
| `variante`, `bloque`, `particion` | Representation variant, feature block and train/test partition |
| `triaje`, `dos_canales`, `azar` | Triage, two-channel and random-sampling protocol labels |
| `fijo`, `triaje_aad`, `podada` | Fixed-weight discovery, feedback-assisted triage and pruned representation |
| `conf_anom`, `conf_dudoso` | Low- and intermediate-confidence gate thresholds |
| `etq_anom_min` | Minimum numeric label treated as an anomaly (990) |

PLAsTiCC bands use integer passbands 0–5 corresponding to `ugrizy`. Times are MJD
days and fluxes use the simulation's differential-flux convention. Known-class
codes are listed in `src/models/classification.py`; simulation 99x subtypes are
held outside the initial classifier taxonomy.

The feature representation contains 91 statistical/metadata inputs, 136 GP1D
descriptors, 53 GP2D descriptors and 24 derived proxies. The exact 304-column
order is recorded in `models/e2gp_150_calibrado.meta.json`. Zero-filled unsupported
features are a preprocessing convention, not physical zero-valued measurements.

CSV filenames also remain stable. For example, `comparacion_*` denotes a
comparison, `por_semilla` per-seed results, `resumen` a summary and `particiones`
the matched train/test object assignments. Batch 99 denotes the combined pool,
not one raw input batch. The five-seed batch-4 query-strategy comparison is a
separate experiment from the final ten-seed acquisition-policy comparisons.
