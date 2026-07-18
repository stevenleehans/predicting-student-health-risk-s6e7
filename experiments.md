# Experiment Log

## Baseline reference

- Competition: Playground Series S6E7 — Predicting Student Health Risk
- Submission ID: `54803701`
- Submission file: `submission_local_ensemble.csv`
- Public leaderboard balanced accuracy: **0.94924**
- Local validation balanced accuracy: **0.95014**
- Validation scheme: one stratified 80/20 split, `random_state=42`
- Full training data used: 690,088 rows
- Ensemble: CatBoost + XGBoost + LightGBM
- Validation-selected blend weights: CatBoost 0.80, XGBoost 0.10, LightGBM 0.10

---

## Experiment 001 — Where does the baseline make errors?

### Date

2026-07-18

### Status

Completed.

### Question

Where do the current ensemble's errors live? In particular:

1. Are errors concentrated in extreme numerical outliers?
2. Are errors concentrated in rows with missing information?
3. Which true classes, predicted classes, and categorical subgroups are hardest?
4. Are the errors mostly low-confidence boundary cases or confident systematic mistakes?

### Hypothesis

The initial hypothesis was that unusual health measurements or sparse rows might account for a disproportionate share of the errors. Numerical outliers and missingness were tested separately so that they would not be conflated.

### Experimental design

- Reproduced the submitted ensemble on the exact baseline validation split.
- Training rows: 552,070.
- Validation rows: 138,018.
- Target labels were encoded as `unhealthy=0`, `at-risk=1`, and `fit=2`.
- Primary metric: balanced accuracy.
- CatBoost used raw categorical features and balanced class weights.
- XGBoost and LightGBM used median numerical imputation, missing indicators, and ordinal categorical encoding.
- Ensemble weights were selected on a 0.1-spaced validation grid, matching the baseline procedure.
- An error means `predicted class != true class`.
- A numerical outlier was defined using bounds calculated only from the training fold:
  - lower bound = Q1 - 3 × IQR
  - upper bound = Q3 + 3 × IQR
- Missingness was measured across the 13 original predictors, not the engineered features.
- Categorical subgroup rates are descriptive. Small groups must not be interpreted without checking their row count.

### Reproduction check

The experiment reproduced the baseline exactly:

| Model | Validation balanced accuracy |
|---|---:|
| CatBoost | 0.949915 |
| XGBoost | 0.932188 |
| LightGBM | 0.937371 |
| 0.8 / 0.1 / 0.1 blend | **0.950137** |

Overall validation accuracy was 0.940964. There were **8,148 errors among 138,018 rows**.

### Confusion matrix

Rows are true classes and columns are predicted classes.

| True class | Predicted unhealthy | Predicted at-risk | Predicted fit | Recall |
|---|---:|---:|---:|---:|
| unhealthy | 11,124 | 373 | 48 | 0.9635 |
| at-risk | 4,723 | 111,194 | 2,595 | 0.9383 |
| fit | 49 | 360 | 7,552 | 0.9486 |

The error pairs were:

| True → predicted | Errors | Share of all errors |
|---|---:|---:|
| at-risk → unhealthy | 4,723 | 58.0% |
| at-risk → fit | 2,595 | 31.8% |
| unhealthy → at-risk | 373 | 4.6% |
| fit → at-risk | 360 | 4.4% |
| fit → unhealthy | 49 | 0.6% |
| unhealthy → fit | 48 | 0.6% |

**Interpretation:** 7,318 of 8,148 errors (89.8%) have a true label of `at-risk`. Because `at-risk` is also the majority class, its row-level error rate is the more useful comparison: 6.17%, versus 3.65% for `unhealthy` and 5.14% for `fit`. The model is not simply collapsing minority classes into the majority class; it is frequently pushing ambiguous majority-class rows toward either extreme class.

### Outlier analysis

Conservative 3×IQR bounds found only **2 validation rows with any numerical outlier**, both in `water_intake`. Neither row was misclassified.

| Has numerical outlier | Rows | Errors | Error rate |
|---|---:|---:|---:|
| No | 138,016 | 8,148 | 5.90% |
| Yes | 2 | 0 | 0.00% |

The numerical ranges are highly bounded. No validation outliers were detected for sleep duration, heart rate, BMI, calorie expenditure, step count, or exercise duration under this definition.

**Conclusion on outliers:** numerical outliers are not the current failure mode. Removing or clipping extreme rows is very unlikely to improve this baseline. This does not prove that every numerical value is realistic; it shows that conventional extreme-value anomalies do not explain the model errors.

### Missingness analysis

Error rate rises sharply with the number of missing predictors:

| Missing predictors | Rows | Errors | Error rate |
|---:|---:|---:|---:|
| 0 | 69,986 | 780 | **1.11%** |
| 1 | 49,501 | 4,267 | **8.62%** |
| 2 | 15,510 | 2,452 | **15.81%** |
| 3 | 2,691 | 561 | **20.85%** |
| 4 | 310 | 82 | **26.45%** |
| 5 | 17 | 6 | **35.29%** |

Rows with no missing predictors are handled extremely well. The largest individual missing-feature effects are:

| Missing feature | Rows | Errors | Error rate | Share of all errors |
|---|---:|---:|---:|---:|
| stress_level | 16,393 | 4,062 | **24.78%** | **49.85%** |
| sleep_duration | 15,297 | 3,522 | **23.02%** | **43.23%** |
| physical_activity_level | 7,368 | 683 | 9.27% | 8.38% |
| sleep_quality | 11,598 | 682 | 5.88% | 8.37% |
| calorie_expenditure | 10,731 | 637 | 5.94% | 7.82% |

These shares overlap because a row can have multiple missing values.

The most common missingness patterns reinforce the result:

| Pattern | Rows | Errors | Error rate |
|---|---:|---:|---:|
| No original feature missing | 69,986 | 780 | 1.11% |
| Only stress_level missing | 9,448 | 2,092 | 22.14% |
| Only sleep_duration missing | 8,632 | 1,754 | 20.32% |
| stress_level and sleep_duration missing | 1,179 | 510 | **43.26%** |

The combination of missing `stress_level` and missing `sleep_duration` is particularly damaging.

### Categorical subgroup analysis

Selected groups with at least 500 validation rows:

| Feature/group | Rows | Errors | Error rate |
|---|---:|---:|---:|
| stress_level = missing | 16,393 | 4,062 | **24.78%** |
| physical_activity_level = missing | 7,368 | 683 | 9.27% |
| physical_activity_level = active | 42,463 | 3,855 | 9.08% |
| stress_level = high | 35,590 | 2,733 | 7.68% |
| sleep_quality = poor | 42,406 | 2,873 | 6.78% |
| stress_level = low | 33,596 | 1,055 | 3.14% |
| stress_level = medium | 52,439 | 298 | **0.57%** |

`stress_level` is the clearest structural feature. Medium stress is nearly deterministic for the current model, while missing stress is extremely difficult. Missing stress alone accounts for almost half of all validation errors.

### Confidence analysis

- Mean confidence on correct predictions: 0.9369.
- Mean confidence on errors: 0.7742.
- Errors with confidence at least 0.90: 1,419.
- Therefore, most mistakes are lower-confidence cases, but a meaningful 17.4% of errors are high-confidence systematic mistakes.

| Maximum predicted probability | Rows | Errors | Error rate |
|---|---:|---:|---:|
| ≤ 0.50 | 404 | 240 | 59.41% |
| 0.50–0.60 | 1,963 | 950 | 48.40% |
| 0.60–0.70 | 1,877 | 929 | 49.49% |
| 0.70–0.80 | 4,388 | 1,907 | 43.46% |
| 0.80–0.90 | 14,181 | 2,703 | 19.06% |
| 0.90–1.00 | 115,205 | 1,419 | 1.23% |

### Main conclusion

**The error does not live in numerical outliers. It lives primarily in missing `stress_level` and missing `sleep_duration`, especially for true `at-risk` rows that the model pushes toward `unhealthy` or `fit`.**

The clean, fully observed subset already has only a 1.11% error rate. Generic model-capacity increases are therefore unlikely to be the highest-value next move. The next experiment should target the missing-feature mechanism directly.

### Recommended Experiment 002

Test missingness-aware specialist models:

1. Train the normal model on all rows.
2. Train separate specialist models for `stress_level` missing, `sleep_duration` missing, and both missing.
3. Add explicit missingness-pattern features and interactions.
4. Evaluate each specialist only on its routed validation subset.
5. Compare against the current ensemble using the identical split, with special attention to `at-risk → unhealthy` and `at-risk → fit` errors.

An alternative or complementary test is to predict the missing stress/sleep variable from the remaining features, then provide both the imputed value and an explicit missing flag to the health-condition model. This must be done fold-safely to avoid leakage.

### Artifacts

- `experiment_001_error_analysis.py` — reproducible experiment.
- `experiment_001_artifacts/summary.json` — headline metrics and blend weights.
- `experiment_001_artifacts/validation_predictions.csv` — row-level labels, probabilities, confidence, missingness, and outlier flags.
- `experiment_001_artifacts/confusion_matrix.csv` — class confusion counts.
- `experiment_001_artifacts/class_error.csv` — error rate by true class.
- `experiment_001_artifacts/missingness_error.csv` — error rate by missing-feature count.
- `experiment_001_artifacts/outlier_error.csv` — error rate for outlier versus non-outlier rows.
- `experiment_001_artifacts/outlier_bounds.csv` — training-fold-only outlier bounds.
- `experiment_001_artifacts/categorical_subgroup_error.csv` — subgroup error tables.
- `experiment_001_artifacts/numeric_error_profile.csv` — numerical profiles for correct and incorrect rows.
- `experiment_001_artifacts/high_confidence_errors.csv` — 1,000 most confident mistakes.

### Limitations

- This is one stratified holdout split, not out-of-fold analysis. The close local/public scores support the split, but subgroup results should eventually be confirmed across folds.
- The blend weights were selected on the same validation split used for diagnosis, so its score is mildly optimistic locally.
- IQR outliers address extreme values, not semantic or medically implausible combinations.
- Subgroup associations identify where errors occur; they do not prove causality.

---

## Experiment 002 — Predict missing stress and sleep before predicting health

### Date

2026-07-18

### Status

Completed. The idea was **not accepted for the next submission** because it reduced overall balanced accuracy.

### Question

Are `stress_level` and `sleep_duration` important enough to justify a two-stage system, and can predicted values for those missing features improve health-condition classification?

### Decision rule

1. Extract feature importance from all three fitted baseline models.
2. If stress and sleep rank highly, train auxiliary models using only the baseline training fold.
3. Predict missing stress first.
4. Predict missing sleep second, allowing the sleep model to use observed or predicted stress.
5. Retrain the dominant CatBoost health model with the filled values plus explicit missingness flags.
6. Compare against the original CatBoost on the identical validation rows.

The final comparison is CatBoost-to-CatBoost, not ensemble-to-CatBoost. This isolates the effect of the two-stage feature treatment before paying the cost of rebuilding the full ensemble.

### Feature-importance method

The reported values are each model's native importance normalized to sum to one:

- CatBoost: native feature importance.
- XGBoost: native tree feature importance.
- LightGBM: native tree feature importance.

These rankings describe how each fitted model used its inputs. They are not causal effects, and raw importance magnitude should not be compared directly across model families. LightGBM's importance is also more distributed across continuous variables, while XGBoost assigns substantial importance to explicit missing indicators.

### Feature-importance results

#### CatBoost — all features

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | stress_level | 0.2537 |
| 2 | sleep_duration | 0.1419 |
| 3 | physical_activity_level | 0.1019 |
| 4 | sleep_distance_8h | 0.0984 |
| 5 | bmi | 0.0913 |
| 6 | bmi_distance_normal | 0.0508 |
| 7 | exercise_duration | 0.0339 |
| 8 | smoking_alcohol | 0.0291 |
| 9 | water_intake | 0.0261 |
| 10 | heart_rate | 0.0260 |
| 11 | calories_per_step | 0.0256 |
| 12 | step_count | 0.0250 |
| 13 | sleep_quality | 0.0236 |
| 14 | calorie_expenditure | 0.0182 |
| 15 | steps_per_exercise_min | 0.0180 |
| 16 | missing_count | 0.0135 |
| 17 | water_per_calorie | 0.0130 |
| 18 | gender | 0.0096 |
| 19 | diet_type | 0.0004 |

Stress and sleep-related features account for almost half of CatBoost's normalized native importance when `stress_level`, `sleep_duration`, and `sleep_distance_8h` are considered together.

#### XGBoost — top features

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | stress_level | 0.2413 |
| 2 | physical_activity_level | 0.2044 |
| 3 | missingindicator_sleep_distance_8h | 0.1219 |
| 4 | missingindicator_sleep_duration | 0.1111 |
| 5 | sleep_duration | 0.1085 |
| 6 | missing_count | 0.0360 |
| 7 | sleep_distance_8h | 0.0293 |
| 8 | missingindicator_water_per_calorie | 0.0177 |
| 9 | missingindicator_steps_per_exercise_min | 0.0104 |
| 10 | smoking_alcohol | 0.0098 |

XGBoost strongly confirms the importance of stress, sleep, and the fact that sleep is missing. The sleep-duration value and its two related missing indicators together receive 0.3415 importance.

#### LightGBM — all original/engineered features plus leading indicators

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | heart_rate | 0.0902 |
| 2 | exercise_duration | 0.0787 |
| 3 | water_intake | 0.0773 |
| 4 | bmi | 0.0754 |
| 5 | bmi_distance_normal | 0.0740 |
| 6 | calorie_expenditure | 0.0731 |
| 7 | sleep_duration | 0.0709 |
| 8 | steps_per_exercise_min | 0.0687 |
| 9 | step_count | 0.0671 |
| 10 | sleep_distance_8h | 0.0630 |
| 11 | water_per_calorie | 0.0606 |
| 12 | calories_per_step | 0.0605 |
| 13 | missing_count | 0.0261 |
| 14 | stress_level | 0.0206 |
| 15 | smoking_alcohol | 0.0179 |
| 16 | sleep_quality | 0.0164 |
| 17 | diet_type | 0.0138 |
| 18 | gender | 0.0134 |
| 19 | physical_activity_level | 0.0133 |
| 20 | missingindicator_sleep_duration | 0.0041 |

LightGBM ranks sleep duration seventh and stress fourteenth. Its built-in importance is more evenly distributed than the other models, so this does not negate the strong CatBoost/XGBoost result.

The complete, untruncated importance table—including every missing-indicator feature—is stored in `experiment_002_artifacts/feature_importance_all_models.csv`.

### Two-stage implementation

#### Stage A — stress prediction

- Model: CatBoost multiclass classifier.
- Target: `stress_level`.
- Inputs: every original predictor except stress level.
- Training data: only baseline-training-fold rows with observed stress.
- Evaluation: a separate stratified 15% auxiliary holdout drawn only from the baseline training fold.
- Result: **45.88% accuracy**.
- Missing stress was predicted for both health-training and health-validation rows.
- Original observed stress values were retained.
- `stress_level_was_missing` was added as an explicit flag.

#### Stage B — sleep-duration prediction

- Model: CatBoost regressor.
- Target: `sleep_duration`.
- Inputs: every original predictor except sleep duration, plus the stress-missing flag. Stress could be observed or predicted by Stage A.
- Training data: only baseline-training-fold rows with observed sleep duration.
- Evaluation: a separate 15% auxiliary holdout from the baseline training fold.
- MAE: **0.8710 hours**.
- RMSE: **1.1023 hours**.
- Missing sleep was predicted for health-training and health-validation rows.
- Original observed sleep values were retained.
- `sleep_duration_was_missing` was added as an explicit flag.
- Sleep-derived features were recalculated from the filled sleep value.

#### Stage C — health prediction

- Model: CatBoost with the same principal hyperparameters as the baseline CatBoost.
- Inputs: two-stage-filled data, explicit stress/sleep missing flags, and recalculated engineered features.
- Comparator: original baseline CatBoost on the exact same split.

All auxiliary models were trained strictly inside the baseline training fold. Validation stress, sleep, and health labels were not used to train the auxiliary models.

### Main result

| Model | Balanced accuracy | Accuracy | Errors |
|---|---:|---:|---:|
| Baseline CatBoost | **0.949915** | 0.938146 | 8,537 |
| Two-stage CatBoost | 0.949559 | **0.938704** | **8,460** |
| Difference | **-0.000357** | +0.000558 | -77 |

The two-stage model made 77 fewer total errors but had worse balanced accuracy. It improved majority-weighted accuracy while slightly damaging the average recall across classes. Because the competition metric is balanced accuracy, this is a failed overall experiment.

### Missingness-subset results

| Subset | Rows | Baseline balanced accuracy | Two-stage balanced accuracy | Difference | Baseline errors | Two-stage errors |
|---|---:|---:|---:|---:|---:|---:|
| All rows | 138,018 | **0.949915** | 0.949559 | -0.000357 | 8,537 | 8,460 |
| Stress missing | 16,393 | **0.859721** | 0.859056 | -0.000665 | 4,278 | 4,242 |
| Sleep missing | 15,297 | **0.859208** | 0.857327 | -0.001881 | 3,762 | 3,651 |
| Both missing | 1,797 | 0.663601 | **0.671831** | **+0.008230** | 915 | 841 |
| Neither missing | 108,125 | 0.971550 | **0.971565** | +0.000015 | 1,412 | 1,408 |

The only meaningful balanced-accuracy gain occurs when **both** stress and sleep are missing: +0.00823, with 74 fewer errors across 1,797 rows. However, routing only that subset to the specialist would have a limited whole-validation effect and requires a separate routing experiment before use.

### Interpretation

The feature-importance premise was correct, but the imputation premise was not strong enough:

1. Stress and sleep are highly predictive of health condition.
2. Their missing values are not highly predictable from the remaining fields.
3. A predicted value is not equivalent to an observed value. The auxiliary noise can distort class boundaries.
4. CatBoost already handles missing values and missingness patterns directly. Replacing missingness with a noisy point estimate can remove useful uncertainty information even when an explicit flag is retained.
5. The reduced error count alongside reduced balanced accuracy shows why ordinary accuracy must not guide this competition.

### Decision

**Do not replace the baseline with the global two-stage model and do not submit it.**

The idea remains promising only as a narrowly routed specialist for rows where both stress and sleep are missing. That should be tested using probabilities rather than hard imputations and evaluated across multiple folds before submission.

### Recommended next test

Experiment 003 should compare three treatments only on the `stress missing AND sleep missing` route:

1. Baseline ensemble probabilities.
2. Two-stage specialist probabilities.
3. A learned or cross-validated probability blend between the two.

The router must leave every other row on the original ensemble. This isolates the +0.00823 subgroup signal and avoids the degradation observed for stress-only and sleep-only rows.

### Artifacts

- `experiment_002_two_stage_imputation.py` — reproducible two-stage test.
- `experiment_002_artifacts/summary.json` — headline auxiliary and health-model results.
- `experiment_002_artifacts/feature_importance_all_models.csv` — complete importance ranking for all three baseline models.
- `experiment_002_artifacts/feature_importance_top25.csv` — compact ranking table.
- `experiment_002_artifacts/two_stage_comparison.csv` — overall and missingness-subset comparisons.
- `experiment_002_artifacts/two_stage_health_feature_importance.csv` — importance from the final two-stage health model.

### Limitations

- The test used a single validation split.
- Model-native importance can be biased toward features that offer many split opportunities and toward correlated feature families.
- The auxiliary models generate point estimates, which discard predictive uncertainty.
- The final test retrained CatBoost only. It intentionally did not rebuild XGBoost, LightGBM, or the full ensemble after the CatBoost-level test failed overall.
- The auxiliary validation scores measure prediction on naturally observed values; missing values could follow a different, unobservable mechanism.

---

## Experiment 003 — Native-NaN XGBoost under trusted 5-fold CV

### Date

2026-07-18

### Status

Completed, accepted as the new XGBoost representation, trained on the full dataset, and submitted as a standalone model.

### Question

Does preserving all numerical and categorical NaNs for XGBoost's learned default split directions outperform the baseline representation of median/mode imputation plus explicit numerical missing indicators?

### Motivation

Experiment 001 showed that missingness—not numerical outliers—is the dominant error mechanism. A Kaggle discussion reported a large gain from letting XGBoost route NaNs natively. Our original XGBoost did not do this: numerical columns were median-imputed with indicators and categorical columns were mode-imputed then ordinal-encoded.

### Trusted CV contract established here

- Splitter: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
- Metric: balanced accuracy.
- Every row receives exactly one out-of-fold prediction.
- Fold assignments are saved and must be reused in subsequent experiments.
- Report mean, sample standard deviation, OOF score, per-class recall, and fold-level deltas.
- Model comparisons use identical folds and target encoding.
- Target encoding: `unhealthy=0`, `at-risk=1`, `fit=2`.

This replaces the single 80/20 holdout as the trusted experiment framework. The earlier holdout remains useful as historical baseline evidence but is no longer sufficient for accepting small changes.

### Controlled configurations

Both configurations used the same feature engineering, class weights, model hyperparameters, and fold rows.

#### Pre-imputed XGBoost

- Numerical values: median imputation with missing indicators.
- Categorical values: most-frequent imputation followed by ordinal encoding.
- Unknown categorical values: `-1`.

#### Native-NaN XGBoost

- Numerical NaNs remained NaN.
- Categorical columns used pandas categorical dtype.
- Categorical NaNs remained NaN.
- Category dictionaries were learned from the fitting fold only.
- Unknown validation categories became NaN and followed learned missing paths.
- `enable_categorical=True` and `missing=np.nan`.

### Shared model settings

- XGBoost 3.2.0.
- `n_estimators=1000`.
- `learning_rate=0.05`.
- `max_depth=8`.
- `min_child_weight=15`.
- `subsample=0.85`.
- `colsample_bytree=0.8`.
- `reg_lambda=5`.
- `tree_method='hist'`.
- Fold-specific balanced sample weights.
- `random_state=42`.

### Device and parallelism

- Device: CPU. XGBoost on this Apple Silicon environment does not provide an Apple MPS backend.
- Parallelism: `n_jobs=-1` within each fold.
- Folds ran sequentially to avoid CPU oversubscription and memory contention between simultaneous 690k-row fits.
- Total runtime: approximately 450 seconds.

### Fold results

| Fold | Pre-imputed | Native NaN | Native gain |
|---:|---:|---:|---:|
| 0 | 0.934884 | **0.947658** | +0.012774 |
| 1 | 0.937602 | **0.948132** | +0.010529 |
| 2 | 0.936568 | **0.946293** | +0.009725 |
| 3 | 0.934687 | **0.946027** | +0.011340 |
| 4 | 0.933834 | **0.944176** | +0.010342 |

Native NaNs won all five folds. The paired gain was **+0.010942 ± 0.001176**, with a minimum fold gain of +0.009725.

### Aggregate results

| Configuration | CV balanced accuracy | CV standard deviation | OOF accuracy | OOF errors |
|---|---:|---:|---:|---:|
| Pre-imputed | 0.935515 | 0.001531 | 0.936244 | 43,997 |
| Native NaN | **0.946457** | 0.001554 | **0.947415** | **36,288** |
| Difference | **+0.010942** | — | +0.011171 | **-7,709** |

Because every fold has nearly identical size and stratification, the aggregate OOF balanced accuracy equals the fold mean to the displayed precision.

### Per-class OOF recall

| Configuration | Unhealthy recall | At-risk recall | Fit recall |
|---|---:|---:|---:|
| Pre-imputed | 0.942035 | 0.936214 | 0.928297 |
| Native NaN | **0.950021** | **0.947538** | **0.941813** |
| Difference | +0.007986 | +0.011324 | +0.013516 |

Native handling improves every class, with the largest recall improvement for `fit`. This is not a majority-class-only accuracy gain.

### Confusion reduction

- Pre-imputed OOF errors: 43,997.
- Native-NaN OOF errors: 36,288.
- Errors removed: **7,709**, or 17.5% of the pre-imputed error count.
- True `at-risk` rows predicted as `unhealthy` fell from 23,514 to 19,404.
- True `at-risk` rows predicted as `fit` fell from 14,283 to 11,683.

### Decision

**Accept native missing-value routing for all future XGBoost experiments.**

The gain is large, appears on every fold, improves all three class recalls, and directly matches the error mechanism found in Experiment 001. Pre-imputation is now deprecated for XGBoost in this project.

This standalone native XGBoost CV score is still below the historical one-split ensemble score of 0.95014. Those numbers are not directly comparable because the ensemble figure used one holdout. The ensemble must be rebuilt using these fixed folds before deciding whether native XGBoost replaces or blends with CatBoost.

### Full-data build and Kaggle submission

- Full training rows: 690,088.
- Test rows: 295,753.
- Training device: multicore CPU, `n_jobs=-1`.
- Full-model training and prediction time: approximately 61 seconds.
- Submission file: `submission_native_nan_xgboost.csv`.
- Kaggle submission ID: `54804927`.
- Public leaderboard balanced accuracy: **0.94800**.
- Five-fold CV mean: **0.94646 ± 0.00155**.
- Public-minus-CV difference: **+0.00154**, approximately one CV standard deviation.
- Previous ensemble public score: **0.94924**.
- Native XGBoost versus previous ensemble: **-0.00124**.

The public result supports the CV framework: the leaderboard score is close to the expected CV range. It also confirms that native XGBoost alone does not replace the original ensemble. Its value is as an improved XGBoost component for the next cross-validated blend.

The full-data prediction distribution was 82.03% `at-risk`, 10.85% `unhealthy`, and 7.11% `fit`.

### Recommended next experiment

Experiment 004 should test class-prior probability correction on the saved native-NaN OOF probabilities, then compare it with sample weighting. This is inexpensive because probability correction can first be evaluated without refitting. Any correction parameters must be learned fold-safely or through nested/cross-fitted logic before final acceptance.

After that, rebuild CatBoost and the ensemble using the same saved five folds. Multi-seed bagging should remain a later variance-reduction step, not the next representation experiment.

### Artifacts

- `experiment_003_native_nan_xgboost.py` — reproducible five-fold experiment.
- `experiment_003_train_submit.py` — full-data native-XGBoost training and submission build.
- `experiment_003_artifacts/fold_assignments.csv` — fixed reusable fold assignment for every training ID.
- `experiment_003_artifacts/fold_scores.csv` — fold metrics and per-class recall.
- `experiment_003_artifacts/oof_predictions.npz` — targets, fold IDs, and OOF probabilities for both representations.
- `experiment_003_artifacts/summary.csv` — aggregate CV and OOF metrics.
- `experiment_003_artifacts/metadata.json` — versions, parameters, device, parallelism, runtime, and confusion matrices.
- `experiment_003_artifacts/full_train_metadata.json` — full-build parameters, runtime, and prediction distribution.
- `experiment_003_artifacts/full_model_feature_importance.csv` — feature importance from the full native-XGBoost model.

### Limitations

- Both models use class weights. Prior correction has not yet been tested.
- Hyperparameters were inherited from the baseline rather than tuned under five-fold CV.
- The comparison isolates missing-value representation; it does not establish the best XGBoost configuration.
- The current public leaderboard reference is from the old ensemble, not this standalone native-NaN XGBoost.

---

## Experiment 004 — Fold-bagged native ensemble

### Date

2026-07-18

### Status

Completed, submitted, and accepted as the current best public submission.

### Question

Does combining native-missing CatBoost, native-NaN XGBoost, and native-NaN LightGBM on the trusted five folds improve over the individual models and the previous ensemble?

### Missing-value decision

Global imputation was not reused. Experiment 002 showed that predicted stress/sleep values reduced overall balanced accuracy, and Experiment 003 showed a large, stable gain from native XGBoost missing routing.

- CatBoost: numerical NaNs remained native; categorical missingness used a dedicated `<MISSING>` category because CatBoost requires categorical inputs as text/category values.
- XGBoost: numerical and categorical NaNs remained native, with pandas categorical dtype and `enable_categorical=True`.
- LightGBM: numerical NaNs remained native; categorical inputs used pandas categorical dtype so missing categories used native missing codes.

Missingness was therefore preserved as signal in every library. No median or most-frequent replacement was applied.

### Validation and bagging

- Reused the exact five fold assignments from Experiment 003.
- Every model generated aligned OOF probabilities.
- Each fold model also predicted the test data.
- Test probabilities were averaged across five fold models for each library.
- The final submission therefore includes fold bagging in addition to model blending.
- Models used all CPU threads within a fold; folds ran sequentially to avoid oversubscription.
- Apple MPS was unavailable for CatBoost, XGBoost, and LightGBM.
- Total runtime: approximately 2,387 seconds (39.8 minutes).

### Fold scores

| Fold | CatBoost | XGBoost | LightGBM |
|---:|---:|---:|---:|
| 0 | 0.949564 | 0.947658 | 0.948871 |
| 1 | 0.951198 | 0.948132 | 0.949099 |
| 2 | 0.948966 | 0.946293 | 0.947162 |
| 3 | 0.949015 | 0.946027 | 0.947531 |
| 4 | 0.947715 | 0.944176 | 0.945068 |

CatBoost led every fold. The relative fold difficulty was consistent across all three libraries, supporting the stability of the fixed split.

### OOF results

| Model | OOF balanced accuracy | OOF accuracy | OOF errors | Unhealthy recall | At-risk recall | Fit recall |
|---|---:|---:|---:|---:|---:|---:|
| CatBoost | 0.949292 | 0.938435 | 42,485 | 0.963239 | 0.935286 | 0.949351 |
| XGBoost | 0.946457 | **0.947415** | **36,288** | 0.950021 | **0.947538** | 0.941813 |
| LightGBM | 0.947546 | 0.946200 | 37,127 | 0.953624 | 0.945668 | 0.943346 |
| Blend | **0.949439** | 0.940351 | 41,163 | 0.961645 | 0.937698 | 0.948974 |

Balanced accuracy and ordinary accuracy prefer different models. XGBoost makes fewer total errors because it is stronger on the majority `at-risk` class, while class-balanced CatBoost has higher macro recall. The competition metric correctly favors CatBoost and the blend.

### Blend search

A coarse convex grid with 0.05 increments was searched against the aligned OOF probabilities. Fine-grained weights were intentionally avoided because the expected gain is small and tiny weight differences would be difficult to trust.

Best weights:

- CatBoost: **0.80**.
- XGBoost: **0.05**.
- LightGBM: **0.15**.

The blend improved OOF balanced accuracy from CatBoost's 0.949292 to **0.949439**, a gain of **+0.000147**. This is small, so leaderboard confirmation was important.

### Kaggle result

- Submission file: `submission_native_ensemble.csv`.
- Submission ID: `54805913`.
- Public leaderboard balanced accuracy: **0.94943**.
- Previous best public score: 0.94924.
- Public improvement: **+0.00019**.
- Standalone native-XGBoost public score: 0.94800.

The small OOF gain transferred almost exactly to the public leaderboard. This is strong evidence that the fixed five-fold CV is directionally trustworthy even for small changes, although private-leaderboard uncertainty remains.

### Decision

**Accept the 0.80 CatBoost / 0.05 XGBoost / 0.15 LightGBM fold-bagged native ensemble as the current best model.**

Continue to preserve missingness natively. Do not return to global median/mode imputation or hard two-stage imputation.

### Remaining high-value levers

1. **OOF class-prior/probability correction.** Balanced sample weights can distort calibration. Test class-wise probability multipliers on saved OOF probabilities, then cross-fit or use a sufficiently coarse/stable correction before submission.
2. **Missing-pattern specialist routing.** Experiment 002 improved the both-stress-and-sleep-missing subset. Test a routed probability blend only for that subset, leaving all other rows on the accepted ensemble.
3. **Native CatBoost/LightGBM tuning.** Tune one model at a time on fixed folds—particularly class weighting, depth/leaves, regularization, and early-stopping behavior.
4. **Original/source dataset augmentation.** Playground data is synthetic; identify and validate the original 50k health dataset without leaking validation labels. Distribution alignment and deduplication are mandatory.
5. **Missingness interactions.** Add compact pattern features or targeted interactions without replacing NaNs. Ablate them under the fixed folds.
6. **Multi-seed bagging.** Use only after the representation and probability correction are stable. It is a variance-reduction lever, not the main signal lever.

### Recommended next experiment

Experiment 005 should test class-wise probability multipliers on the saved OOF blend and individual-model probabilities. Start with a coarse, constrained search and measure fold-by-fold changes. Because optimization and evaluation on the same OOF labels can overfit, acceptance should require either cross-fitted multiplier selection or stability across independently optimized folds.

### Artifacts

- `experiment_004_native_ensemble.py` — complete fold training, native missing handling, OOF blend search, fold-bagged test prediction, and submission build.
- `experiment_004_artifacts/fold_scores.csv` — fold-level model scores.
- `experiment_004_artifacts/summary.csv` — OOF model and blend metrics.
- `experiment_004_artifacts/oof_predictions.npz` — aligned OOF probabilities and targets.
- `experiment_004_artifacts/metadata.json` — weights, runtime, device, prediction distribution, submission ID, and public score.

### Limitations

- Blend weights were selected and evaluated on the same complete OOF labels, so the +0.000147 local gain may be mildly optimistic.
- Only one seed was used per fold. Fold bagging reduces variance, but this is not multi-seed bagging.
- Test probabilities come from fold models rather than models retrained on 100% of the data. This is deliberate bagging but changes the training size per model.
- The public leaderboard is only a subset of the hidden test labels; the private score may reorder small gains.

---

## Experiment 005 — Cross-fitted class probability correction

### Date

2026-07-18

### Status

Completed, submitted, and rejected. Experiment 004 remains the accepted model.

### Question

Can constrained class-wise probability multipliers improve balanced accuracy by correcting the native ensemble's lower recall on `at-risk` rows?

### Method

- Used the saved Experiment 004 OOF blend probabilities.
- Fixed `at-risk` as the reference multiplier at 1.0.
- Searched conservative multipliers from 0.75 to 1.10 for `unhealthy` and `fit`.
- Used coordinate search rather than a large unrestricted optimizer.
- For every held-out fold, learned multipliers using only the other four folds and then evaluated on the untouched fold.
- Saved fold-bagged test probabilities from the deterministic Experiment 004 rebuild so future probability experiments do not require retraining.

### Cross-fitted results

| Fold | Base | Corrected | Delta | Unhealthy multiplier | Fit multiplier |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.949907 | 0.949930 | +0.000023 | 1.0950 | 0.9725 |
| 1 | 0.951460 | 0.951260 | -0.000200 | 1.0950 | 0.9900 |
| 2 | 0.949100 | 0.949121 | +0.000022 | 1.0925 | 0.9900 |
| 3 | 0.949053 | 0.949159 | +0.000107 | 1.0950 | 0.9900 |
| 4 | 0.947673 | 0.948025 | +0.000351 | 1.0900 | 0.9975 |

- Base OOF balanced accuracy: 0.949439.
- Cross-fitted corrected OOF: **0.949499**.
- Cross-fitted gain: **+0.000060**.
- Fold delta standard deviation: 0.000198.
- Improved folds: 4 of 5.
- Full-OOF optimized score: 0.949550; this is optimistic because the same labels selected and evaluated the multipliers.

Deployment multipliers selected on complete OOF predictions:

- `unhealthy`: 1.095.
- `at-risk`: 1.000.
- `fit`: 0.990.

Only 409 of 295,753 test labels changed.

### Kaggle result

- Submission file: `submission_probability_corrected_ensemble.csv`.
- Submission ID: `54806732`.
- Corrected public score: **0.94884**.
- Accepted Experiment 004 score: **0.94943**.
- Public difference: **-0.00059**.

### Decision

**Reject probability correction and retain the uncorrected Experiment 004 ensemble.**

The cross-fitted local improvement was much smaller than its fold-to-fold variability. Although four folds improved, the +0.000060 mean gain was not strong enough to survive the public leaderboard subset. This is not evidence that the CV is broadly untrustworthy: Experiment 003's large native-NaN gain and Experiment 004's ensemble gain transferred. It shows that changes far below approximately one fold-delta standard deviation should not be accepted based on sign alone.

Future acceptance criteria should require both:

1. Positive mean paired fold gain.
2. A gain materially larger than its paired fold standard deviation or a clear improvement on every fold.

### Impact on the 0.951 target

Probability correction is not the lever that will move the project from 0.94943 to 0.951. Reaching 0.951 requires a larger source of new signal rather than boundary adjustments affecting only hundreds of rows.

### Recommended next lever

Test missing-pattern specialist routing using already identified hard subsets, beginning with rows where both `stress_level` and `sleep_duration` are missing. Unlike global correction, this targets a subgroup where Experiment 002 found a +0.00823 balanced-accuracy improvement. The router must be evaluated cross-fitted and must leave all other rows on Experiment 004 probabilities.

### Artifacts

- `experiment_005_probability_correction.py` — cross-fitted multiplier optimization and corrected submission build.
- `experiment_005_artifacts/crossfit_fold_results.csv` — fold-level base/corrected comparison and learned multipliers.
- `experiment_005_artifacts/optimization_history.csv` — fold-specific coordinate-search history.
- `experiment_005_artifacts/full_optimization_history.csv` — complete-OOF search history.
- `experiment_005_artifacts/crossfit_predictions.npz` — cross-fitted predictions and deployment multipliers.
- `experiment_005_artifacts/summary.json` — local metrics, test changes, Kaggle score, and decision.
- `experiment_004_artifacts/test_probabilities.npz` — reusable fold-bagged test probabilities for all component models and the accepted blend.

---

## Experiment 006 — Both-stress-and-sleep-missing specialist router

### Date

2026-07-18

### Status

Completed and rejected. Not submitted to Kaggle because cross-fitted OOF performance decreased.

### Question

Can specialist models improve the hard subset where both `stress_level` and `sleep_duration` are missing, while leaving every other row on the accepted Experiment 004 ensemble?

### Motivation

Experiment 002 showed a +0.00823 balanced-accuracy improvement on this subset in one 80/20 holdout using a global two-stage model. Experiment 006 tested whether a targeted version of that signal survives the trusted five-fold framework.

### Method

- Route: `stress_level is NaN AND sleep_duration is NaN`.
- Routed training rows: 9,013.
- Routed test rows: 3,888.
- Reused Experiment 003 fold assignments and Experiment 004 base probabilities.
- Trained CatBoost, XGBoost, and LightGBM specialists only on matching routed rows from the fitting folds.
- Removed stress and sleep themselves from the specialist feature set because both are constant-missing inside the route.
- Preserved all other missing values natively.
- Each held-out fold's specialist weights and routing alpha were learned using routed rows from the other four folds.
- The search was allowed to choose alpha 0, which means retaining the base ensemble.

### Standalone specialist results

| Fold | Base ensemble | Specialist CatBoost | Specialist XGBoost | Specialist LightGBM |
|---:|---:|---:|---:|---:|
| 0 | 0.659442 | 0.642460 | 0.487194 | 0.457840 |
| 1 | 0.671178 | 0.640584 | 0.495767 | 0.479638 |
| 2 | 0.644670 | **0.653306** | 0.471824 | 0.448342 |
| 3 | 0.617510 | **0.619635** | 0.466183 | 0.452297 |
| 4 | 0.664190 | **0.664283** | 0.499033 | 0.472719 |

CatBoost was the only viable specialist, but it was unstable: it improved three folds slightly and damaged folds 0–1, especially fold 1. XGBoost and LightGBM strongly overfit the small specialist population.

### Cross-fitted router results

| Fold | Route rows | Alpha | Base route BA | Routed route BA | Route delta | Global delta |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1,867 | 0.1 | 0.659442 | 0.657319 | -0.002123 | -0.000029 |
| 1 | 1,791 | 0.8 | 0.671178 | 0.638070 | -0.033108 | -0.000451 |
| 2 | 1,751 | 0.1 | 0.644670 | 0.647223 | +0.002553 | +0.000032 |
| 3 | 1,823 | 0.1 | 0.617510 | 0.621371 | +0.003860 | +0.000049 |
| 4 | 1,781 | 0.1 | 0.664190 | 0.664400 | +0.000210 | +0.000002 |

Aggregate results:

- Base global OOF balanced accuracy: **0.949439**.
- Cross-fitted routed global OOF: **0.949360**.
- Global difference: **-0.000079**.
- Base routed-subset balanced accuracy: **0.650675**.
- Cross-fitted routed-subset balanced accuracy: **0.644657**.
- Routed-subset difference: **-0.006018**.
- Global folds improved: 3 of 5.
- Global fold-delta standard deviation: 0.000210.

### Decision

**Reject the specialist router and do not submit it. Experiment 004 remains best.**

The apparent subset improvement from Experiment 002 was a single-holdout result that did not generalize across the fixed folds. In addition, Experiment 002's model was a global two-stage model rather than a specialist trained only on 9,013 routed rows, so the experiments are not identical. Experiment 006 is the stronger decision basis because its routing parameters were cross-fitted and its failure is visible across multiple folds.

The deployment search would have used only 10% specialist CatBoost and changed 38 test labels, but cross-fitted evidence is negative. No Kaggle submission was spent on it.

### Implication for the 0.951 target

Neither global probability correction nor narrow both-missing routing supplies enough stable signal. The project needs a larger modeling or data lever. The next experiment should test source/original dataset augmentation or a materially different target-generation model, not another tiny boundary adjustment.

### Artifacts

- `experiment_006_missing_pattern_specialist.py` — native specialist training, cross-fitted routing, and optional submission build.
- `experiment_006_artifacts/specialist_fold_scores.csv` — standalone specialist fold scores.
- `experiment_006_artifacts/crossfit_router_results.csv` — fold-specific routing parameters and effects.
- `experiment_006_artifacts/specialist_predictions.npz` — specialist OOF/test probabilities and cross-fitted routed predictions.
- `experiment_006_artifacts/summary.json` — aggregate results and rejection decision.

---

## Experiment 007 — Exact-value target encoding + HistGradientBoosting

### Date

2026-07-18

### Status

Completed, submitted, and accepted as the new project leader. The standalone HGBC-TE candidate was selected for Kaggle submission; the blend candidate was rejected by cross-fitted validation.

### Question

Does inner-cross-fitted target encoding of every exact feature value, paired with `HistGradientBoostingClassifier`, provide a large and stable improvement over the accepted Experiment 004 ensemble?

### Research basis

Kaggle MCP research identified exact-value numeric target encoding as the only repeatedly reported representation change with a nested-CV gain near +0.0009 and matching leaderboard transfer. The main references were:

- `gdataranger/s6e7-v0-7-histgradientboosting-target-encoding` — OOF 0.9502, nested gain +0.0009, public LB 0.95036.
- `redamountassir/ps-s6e7-hgbc-baseline-lb-0-95034-cv-0-95026` — CV 0.95026, public LB 0.95034.
- `yaoguang516/s6e7-per-value-te-hgbc-0-9505-single-model` — independent per-value TE + HGBC implementation.

### Method

- Reused the fixed five folds from Experiment 003.
- Used the 13 original features: seven numerical and six categorical.
- Preserved numerical NaNs in the raw HGBC view.
- Represented categorical missingness with a dedicated `<MISSING>` category.
- Created a second feature view where every raw feature, including each exact numerical value, is a string category.
- Within each outer fitting fold, used `TargetEncoder(cv=5, target_type='multiclass', smooth='auto')` to create 39 leakage-controlled target-encoding columns.
- Concatenated the 39 encoded columns with the 13 raw columns.
- Trained HGBC with the published operating point: learning rate 0.0627037, 300 maximum iterations, 33 leaves, 298 minimum samples per leaf, 237 bins, 0.820265 maximum-feature fraction, balanced class weights, and early stopping.
- Used multicore CPU because scikit-learn HGBC has no Apple MPS backend. OpenMP used the machine's 12 logical CPUs; folds were sequential to control memory.
- Compared the standalone model against Experiment 004 and tested an honest cross-fitted probability blend. Each held-out fold's blend weight was selected using only the other four folds.

### Fold results

| Fold | Rows | HGBC-TE balanced accuracy | Iterations |
|---:|---:|---:|---:|
| 0 | 138,018 | 0.950714 | 100 |
| 1 | 138,018 | 0.951810 | 87 |
| 2 | 138,018 | 0.949515 | 98 |
| 3 | 138,017 | 0.949881 | 93 |
| 4 | 138,017 | 0.949061 | 110 |

### Aggregate results

| Candidate | OOF balanced accuracy | Accuracy | Errors | Unhealthy recall | At-risk recall | Fit recall |
|---|---:|---:|---:|---:|---:|---:|
| Experiment 004 ensemble | 0.949439 | 0.940351 | 41,163 | 0.961645 | 0.937698 | 0.948974 |
| **HGBC exact-value TE** | **0.950196** | **0.940429** | **41,109** | **0.963464** | 0.937573 | **0.949552** |
| Cross-fitted blend | 0.950151 | 0.940408 | 41,124 | 0.963239 | 0.937563 | 0.949652 |

Standalone HGBC-TE improved OOF balanced accuracy by **+0.000757** over Experiment 004. It removed 54 additional errors and improved unhealthy and fit recall while losing only 0.000125 at-risk recall.

### Honest blend check

| Fold | HGBC weight learned on other folds | Base fold BA | HGBC fold BA | Blend fold BA |
|---:|---:|---:|---:|---:|
| 0 | 0.84 | 0.949907 | 0.950714 | 0.950607 |
| 1 | 0.93 | 0.951460 | 0.951810 | **0.951931** |
| 2 | 0.93 | 0.949100 | 0.949515 | **0.949710** |
| 3 | 0.59 | 0.949053 | **0.949881** | 0.949746 |
| 4 | 0.52 | 0.947673 | **0.949061** | 0.948762 |

The full-OOF weight search preferred 84% HGBC and reported 0.950250 on the same data used to choose the weight. The honest cross-fitted blend scored only 0.950151, below standalone HGBC-TE's 0.950196. Therefore the apparent full-OOF blend gain is treated as weight-selection optimism.

### Where the improvement lives

| Candidate | Sleep-missing BA | Sleep-missing error | Sleep-present BA | Sleep-present error |
|---|---:|---:|---:|---:|
| Experiment 004 ensemble | 0.858892 | 23.30% | 0.960638 | 3.82% |
| **HGBC exact-value TE** | **0.861939** | 23.42% | **0.961113** | **3.80%** |
| Cross-fitted blend | 0.861841 | 23.40% | 0.961075 | 3.80% |

The new representation improves balanced accuracy in both segments. On sleep-missing rows it improves class-balanced recall even though raw error rate increases slightly, which is consistent with trading majority-class accuracy for minority recall under balanced accuracy.

### Decision

**Accept standalone HGBC exact-value target encoding as the new local leader. Reject the blend candidate.**

The +0.000757 OOF gain exceeds the project's 0.0005 threshold for a meaningful experiment and closely matches the independently reported +0.0009 gain. The result supports the research conclusion that representation, not another boundary adjustment or conventional ensemble member, was the binding constraint.

### Kaggle result

- Submission file: `submission_experiment_007_hgbc_te.csv`.
- Kaggle submission ID: `54807562`.
- Public leaderboard balanced accuracy: **0.95032**.
- Previous accepted public score: **0.94943**.
- Public improvement: **+0.00089**.
- OOF-to-public difference: **+0.000124**.

The public result validates the fixed-fold improvement and establishes Experiment 007 as the new accepted baseline. It does not yet reach the 0.951 target. The next ranked independent lever is RealMLP on Apple MPS, evaluated on the same folds; any later stack must use cross-fitted OOF probabilities.

### Artifacts

- `experiment_007_exact_value_te_hgbc.py` — fixed-fold training, exact-value target encoding, error slices, honest blend check, and submission generation.
- `experiment_007_artifacts/fold_scores.csv` — fold-level HGBC scores and early-stopping iterations.
- `experiment_007_artifacts/summary.csv` — accepted baseline, HGBC-TE, and cross-fitted blend metrics.
- `experiment_007_artifacts/crossfit_blend_fold_scores.csv` — fold-safe learned weights and evaluation results.
- `experiment_007_artifacts/sleep_missing_error_slices.csv` — error and balanced-accuracy comparison by sleep-missingness.
- `experiment_007_artifacts/oof_predictions.npz` — baseline, HGBC-TE, and cross-fitted OOF probabilities.
- `experiment_007_artifacts/test_probabilities.npz` — standalone and deployment-blend test probabilities.
- `experiment_007_artifacts/metadata.json` — full configuration, versions, runtime, and aggregate deltas.
- `submission_experiment_007_hgbc_te.csv` — selected standalone submission candidate.
- `submission_experiment_007_blend.csv` — rejected blend candidate retained locally for reproducibility.
