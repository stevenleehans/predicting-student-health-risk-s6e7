# Predicting Student Health Risk — Playground Series S6E7

Team workspace for the Kaggle competition **Predicting Student Health Risk**.

## Current benchmark

- Public leaderboard balanced accuracy: **0.95032**
- Fixed five-fold OOF balanced accuracy: **0.950196**
- Model: HistGradientBoosting with inner-cross-fitted exact-value target encoding
- Representation: 13 raw features plus 39 multiclass target-encoding columns

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
- `experiment_001_artifacts/` through `experiment_007_artifacts/` — reproducible result tables and OOF outputs.

## Main findings so far

1. Numerical outliers do not explain the baseline errors.
2. Missing `stress_level` and `sleep_duration` account for most difficult rows.
3. Predicting those missing values globally did not improve balanced accuracy.
4. Native-NaN XGBoost improved 5-fold balanced accuracy from **0.93552 to 0.94646**, winning all five folds.
5. Standalone native-NaN XGBoost scored **0.94800** publicly; it improves the XGBoost component but does not beat the 0.94924 ensemble.
6. The fold-bagged native ensemble reached **0.94943**, using 80% CatBoost, 5% XGBoost, and 15% LightGBM.
7. Exact-value target encoding plus HGBC reached **0.950196 OOF / 0.95032 public**, improving the accepted public score by **+0.00089**.

## Data setup

Download `train.csv`, `test.csv`, and `sample_submission.csv` from the Kaggle competition and place them in the repository root. Competition data is intentionally excluded from Git.

## Reproducibility

The experiments require Python and the packages imported by the notebooks/scripts, including pandas, NumPy, scikit-learn, CatBoost, XGBoost, and LightGBM. Experiment 007 requires scikit-learn 1.6 or newer for the configured `TargetEncoder` and HGBC options.

See `experiments.md` before running a new experiment so every comparison uses the same validation contract.
