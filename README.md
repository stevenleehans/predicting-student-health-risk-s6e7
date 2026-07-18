# Predicting Student Health Risk — Playground Series S6E7

Team workspace for the Kaggle competition **Predicting Student Health Risk**.

## Current benchmark

- Public leaderboard balanced accuracy: **0.94924**
- Local validation balanced accuracy: **0.95014**
- Model: CatBoost/XGBoost/LightGBM probability ensemble
- Blend: 80% CatBoost, 10% XGBoost, 10% LightGBM

The current validation figure is based on one stratified 80/20 holdout. The next modeling framework will use fixed 5-fold stratified cross-validation with reusable folds and out-of-fold predictions.

## Repository contents

- `student_health_eda.ipynb` — exploratory data analysis.
- `student_health_local_ensemble_baseline.ipynb` — local-ready ensemble baseline.
- `student_health_local_ensemble_baseline_executed.ipynb` — executed baseline with validation output.
- `health-stacked-hgbc-catb-xgb-lgbm-baseline.ipynb` — original reference baseline.
- `experiments.md` — experiment registry, results, decisions, and limitations.
- `experiment_001_error_analysis.py` — error, missingness, subgroup, confidence, and outlier analysis.
- `experiment_002_two_stage_imputation.py` — two-stage stress/sleep prediction experiment.
- `experiment_001_artifacts/` and `experiment_002_artifacts/` — compact result tables.

## Main findings so far

1. Numerical outliers do not explain the baseline errors.
2. Missing `stress_level` and `sleep_duration` account for most difficult rows.
3. Predicting those missing values globally did not improve balanced accuracy.
4. XGBoost in the current baseline uses pre-imputation plus missing indicators; native NaN routing is the next high-priority test.

## Data setup

Download `train.csv`, `test.csv`, and `sample_submission.csv` from the Kaggle competition and place them in the repository root. Competition data is intentionally excluded from Git.

## Reproducibility

The experiments require Python and the packages imported by the notebooks/scripts, including pandas, NumPy, scikit-learn, CatBoost, XGBoost, and LightGBM.

See `experiments.md` before running a new experiment so every comparison uses the same validation contract.
