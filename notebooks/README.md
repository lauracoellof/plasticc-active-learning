# Analysis notebooks

Use the declared scientific environment and launch Jupyter from the repository
root. Each notebook explains its inputs and scientific question. Outputs are
cleared in Git; final result tables are stored separately in `data/processed/`.

| Notebook | Scientific question |
|---|---|
| [01 · Exploration](01_exploratory_analysis.ipynb) | What are the sampling, class and flux distributions? |
| [02 · Cleaning](02_cleaning_and_filtering.ipynb) | Which observations and objects survive the quality cuts? |
| [03 · Statistics](03_statistical_features.ipynb) | What do statistical and cross-band descriptors capture? |
| [03b · GP1D](03b_gp1d_features.ipynb) | What can independent band interpolation describe? |
| [03c · GP2D](03c_gp2d_features.ipynb) | What does joint time–wavelength interpolation add? |
| [04 · Classification](04_final_classification.ipynb) | How do representations and the calibrated classifier perform? |
| [04b · Incremental ablation](04b_incremental_representation_ablation.ipynb) | What does each added feature block contribute on matched splits? |
| [04c · Isolated blocks](04c_isolated_feature_blocks.ipynb) | How informative is each block alone? |
| [05 · Anomaly scores](05_anomaly_detection_and_fusion.ipynb) | How complementary are novelty descriptors and their fusion? |
| [06 · Single batch](06_single_batch_active_learning.ipynb) | How does acquisition affect classification and discoveries? |
| [06b · Generalization](06b_batch_generalization.ipynb) | Do the policies transfer across independent batches? |
| [07 · Combined pool](07_combined_pool.ipynb) | How are features and predictions aligned at larger scale? |
| [07b · Scale comparison](07b_large_pool_comparison.ipynb) | How do the policies compare across ten matched seeds? |
| [08 · Interpretability](08_shap_interpretability.ipynb) | Which pre-calibration forest features support each class? |
| [09 · Physical diagnostics](09_physical_analysis.ipynb) | What descriptive properties distinguish simulated anomaly subtypes? |
| [10 · Evaluation](10_evaluation.ipynb) | What do the consolidated classification and discovery results support? |

The supplementary notebooks in `appendices/` cover subsampling (A), model
specification (B), budget allocation (C), representation sensitivity (D) and
budget sensitivity (E). Generate A's validated tables before consolidating them
in notebook 10.

Read [REPRODUCIBILITY.md](../REPRODUCIBILITY.md) before executing everything.
Default cached modes reuse existing scientific results. Fresh extraction,
classifier fitting, SHAP and AL experiments have different costs; saved CSVs
do not make all notebooks instantaneous.
