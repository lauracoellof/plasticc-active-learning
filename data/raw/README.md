# PLAsTiCC source data

Source: PLAsTiCC Team and PLAsTiCC Modelers, *Unblinded Data for PLAsTiCC
Classification Challenge*, [doi:10.5281/zenodo.2539456](https://doi.org/10.5281/zenodo.2539456).

Place these fourteen files in this directory:

- `training_set.csv` and `training_set_metadata.csv`: the decompressed training
  light-curve and metadata tables, using the project's stable local aliases.
- `plasticc_test_lightcurves_01.csv.gz` through `plasticc_test_lightcurves_11.csv.gz`.
- `plasticc_test_metadata.csv.gz`.

```bash
python scripts/verify_raw_data.py
```

`checksums.sha256` identifies the exact project inputs, including the decompressed
training files. Verification streams roughly 8 GB from disk; it does not load all
curves into memory. These data files are excluded from Git.
