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
