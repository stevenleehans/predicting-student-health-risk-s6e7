# Predicting Student Health Risk — Playground Series S6E7

Team workspace for the Kaggle competition **Predicting Student Health Risk**.

## Current benchmark

- Public leaderboard balanced accuracy: **0.95045**
- Fixed five-fold OOF balanced accuracy: **0.950636**
- Model: 44% HistGradientBoosting exact-value TE + 56% RealMLP probability blend
- Validation: blend weight learned cross-fitted on the other four folds; deployment weight fit on full OOF

The original ensemble validation figure is based on one stratified 80/20 holdout. Experiment 003 establishes the trusted evaluation framework: fixed 5-fold stratified cross-validation with reusable fold assignments and out-of-fold predictions.

## Repository contents

- `student_health_eda.ipynb` — exploratory data analysis.
- `student_health_local_ensemble_baseline.ipynb` — local-ready ensemble baseline.
- `student_health_local_ensemble_baseline_executed.ipynb` — executed baseline with validation output.
- `health-stacked-hgbc-catb-xgb-lgbm-baseline.ipynb` — original reference baseline.
- `experiments.md` — experiment registry, results, decisions, and limitations.
- `experiment_001_error_analysis.py` — error, missingness, subgroup, confidence, and outlier analysis.
- `experiment_002_two_stage_imputation.py` — two-stage stress/sleep prediction experiment.
- `experiment_003_native_nan_xgboost.py` — fixed 5-fold comparison of pre-imputation versus native-NaN XGBoost.
- `experiment_007_exact_value_te_hgbc.py` — exact-value target encoding, HGBC training, error slices, and honest blend validation.
- `experiment_008_original_generator_priors.py` — original 50k source-data priors and augmentation test.
- `experiment_009_unweighted_hgbc_prior.py` — unweighted native-NaN HGBC with fold-safe prior correction.
- `experiment_010_realmlp_mps.py` — 16-member RealMLP on Apple MPS and fold-safe HGBC blend.
- `kaggle_experiment_011_options_123_gpu.py` — public Kaggle T4 x2 package testing the exact-rule resolver, TabPFN-3, FT-Transformer-v2, and a restricted fold-safe stack.
- `kernel-metadata.json` — reproducible public Kaggle package configuration and attached competition/model sources.
- `experiment_001_artifacts/` through `experiment_010_artifacts/` — reproducible result tables and OOF outputs.

## Main findings so far

1. Numerical outliers do not explain the baseline errors.
2. Missing `stress_level` and `sleep_duration` account for most difficult rows.
3. Predicting those missing values globally did not improve balanced accuracy.
4. Native-NaN XGBoost improved 5-fold balanced accuracy from **0.93552 to 0.94646**, winning all five folds.
5. Standalone native-NaN XGBoost scored **0.94800** publicly; it improves the XGBoost component but does not beat the 0.94924 ensemble.
6. The fold-bagged native ensemble reached **0.94943**, using 80% CatBoost, 5% XGBoost, and 15% LightGBM.
7. Exact-value target encoding plus HGBC reached **0.950196 OOF / 0.95032 public**, improving the accepted public score by **+0.00089**.
8. Original-data generator priors and 50k-row source augmentation did not beat HGBC (**0.950106 / 0.949680 OOF**).
9. Unweighted HGBC with honest prior correction scored **0.950108 OOF**, so balanced training remains preferred.
10. RealMLP reached **0.950515 OOF / 0.95039 public**; its fold-safe HGBC blend reached the current best **0.950636 OOF / 0.95045 public**.
11. The exact-rule disagreement resolver was rejected at **0.948402 OOF**; the [Experiment 011 GPU package](https://www.kaggle.com/code/stevenleehans/s6e7-options-1-2-3-honest-gpu-cv-v2) is running TabPFN-3 and FT-Transformer-v2 next.

## Data setup

Download `train.csv`, `test.csv`, and `sample_submission.csv` from the Kaggle competition and place them in the repository root. Competition data is intentionally excluded from Git.

## Reproducibility

The experiments require Python and the packages imported by the notebooks/scripts, including pandas, NumPy, scikit-learn, CatBoost, XGBoost, LightGBM, and PyTorch. Experiment 007 requires scikit-learn 1.6 or newer for the configured `TargetEncoder` and HGBC options. Experiment 010 uses Apple MPS when available and otherwise falls back to CUDA or CPU.

See `experiments.md` before running a new experiment so every comparison uses the same validation contract.
