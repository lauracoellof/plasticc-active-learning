# Canonical E2-GP classifier

`e2gp_150_calibrado.joblib` is the operational class-balanced, 150-tree Random
Forest with minimum leaf size three and three-fold sigmoid calibration.

Its JSON sidecar records the ordered 304-feature input schema, class labels,
software versions and model SHA-256. `checksums.sha256` covers both artifacts.
Load it through `src.models.classification.load_e2gp_artifact()` so the integrity,
representation and scikit-learn compatibility checks remain enabled.

```bash
python scripts/verify_models.py
```

The feature matrix must follow the recorded column order. Notebook 04 can fit
and export a replacement model; retain the published artifact when comparing
new experiments with the reported baseline. Identical predictions need not
produce byte-identical serialized models.
