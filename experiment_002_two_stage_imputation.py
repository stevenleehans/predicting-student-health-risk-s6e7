from pathlib import Path
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    mean_absolute_error,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split

# Reproduce Experiment 001 and reuse its fitted models, preprocessing, split,
# and predictions. This ensures the importance and comparison use the same run.
import experiment_001_error_analysis as base


OUT = Path("experiment_002_artifacts")
OUT.mkdir(exist_ok=True)
RANDOM_STATE = base.RANDOM_STATE


def normalized(values):
    values = np.asarray(values, dtype=float)
    total = values.sum()
    return values / total if total else values


# Model-native importances. These are useful ranking diagnostics but are not
# directly comparable estimands across algorithms.
cat_imp = pd.DataFrame({
    "model": "CatBoost",
    "feature": base.features,
    "importance": normalized(base.cat_model.feature_importances_),
})

encoded_names = base.preprocessor.get_feature_names_out().tolist()
xgb_imp = pd.DataFrame({
    "model": "XGBoost",
    "feature": encoded_names,
    "importance": normalized(base.xgb_model.feature_importances_),
})
lgb_imp = pd.DataFrame({
    "model": "LightGBM",
    "feature": encoded_names,
    "importance": normalized(base.lgb_model.feature_importances_),
})

importance = pd.concat([cat_imp, xgb_imp, lgb_imp], ignore_index=True)
importance["rank"] = importance.groupby("model")["importance"].rank(
    method="min", ascending=False
).astype(int)
importance = importance.sort_values(["model", "rank", "feature"])
importance.to_csv(OUT / "feature_importance_all_models.csv", index=False)

importance_top = importance.groupby("model", group_keys=False).head(25)
importance_top.to_csv(OUT / "feature_importance_top25.csv", index=False)


# Work from raw data so engineered sleep features are recalculated after the
# auxiliary predictions. Preserve explicit flags for the original missingness.
raw = pd.read_csv("train.csv")
train_idx, valid_idx = base.train_idx, base.valid_idx
raw_train = raw.loc[train_idx].copy()
raw_valid = raw.loc[valid_idx].copy()
raw_features = [c for c in raw.columns if c not in [base.ID_COL, base.TARGET]]


def prepare_catboost(df, columns):
    out = df[columns].copy()
    cat_cols = out.select_dtypes(exclude=np.number).columns.tolist()
    for col in cat_cols:
        out[col] = out[col].fillna("<MISSING>").astype(str)
    return out, cat_cols


# Stage A: predict stress_level where it is missing.
stress_features = [c for c in raw_features if c != "stress_level"]
stress_observed = raw_train["stress_level"].notna()
stress_fit_idx, stress_eval_idx = train_test_split(
    raw_train.index[stress_observed], test_size=0.15,
    stratify=raw_train.loc[stress_observed, "stress_level"],
    random_state=RANDOM_STATE,
)
stress_X_fit, stress_cat = prepare_catboost(raw_train.loc[stress_fit_idx], stress_features)
stress_X_eval, _ = prepare_catboost(raw_train.loc[stress_eval_idx], stress_features)
stress_model = CatBoostClassifier(
    iterations=700, learning_rate=0.06, depth=8,
    loss_function="MultiClass", eval_metric="Accuracy",
    random_seed=RANDOM_STATE, verbose=100, allow_writing_files=False,
)
stress_model.fit(
    stress_X_fit, raw_train.loc[stress_fit_idx, "stress_level"],
    cat_features=stress_cat,
    eval_set=(stress_X_eval, raw_train.loc[stress_eval_idx, "stress_level"]),
    early_stopping_rounds=80,
)
stress_eval_pred = stress_model.predict(stress_X_eval).reshape(-1)
stress_accuracy = accuracy_score(raw_train.loc[stress_eval_idx, "stress_level"], stress_eval_pred)


def fill_stress(df):
    out = df.copy()
    out["stress_level_was_missing"] = out["stress_level"].isna().astype("int8")
    mask = out["stress_level"].isna()
    if mask.any():
        X, _ = prepare_catboost(out.loc[mask], stress_features)
        out.loc[mask, "stress_level"] = stress_model.predict(X).reshape(-1)
    return out


train_stage = fill_stress(raw_train)
valid_stage = fill_stress(raw_valid)


# Stage B: predict sleep_duration where it is missing, using the observed or
# Stage-A-predicted stress_level.
sleep_features = [
    c for c in raw_features + ["stress_level_was_missing"]
    if c != "sleep_duration"
]
sleep_observed = train_stage["sleep_duration"].notna()
sleep_fit_idx, sleep_eval_idx = train_test_split(
    train_stage.index[sleep_observed], test_size=0.15,
    random_state=RANDOM_STATE,
)
sleep_X_fit, sleep_cat = prepare_catboost(train_stage.loc[sleep_fit_idx], sleep_features)
sleep_X_eval, _ = prepare_catboost(train_stage.loc[sleep_eval_idx], sleep_features)
sleep_model = CatBoostRegressor(
    iterations=700, learning_rate=0.06, depth=8,
    loss_function="RMSE", eval_metric="RMSE",
    random_seed=RANDOM_STATE, verbose=100, allow_writing_files=False,
)
sleep_model.fit(
    sleep_X_fit, train_stage.loc[sleep_fit_idx, "sleep_duration"],
    cat_features=sleep_cat,
    eval_set=(sleep_X_eval, train_stage.loc[sleep_eval_idx, "sleep_duration"]),
    early_stopping_rounds=80,
)
sleep_eval_pred = sleep_model.predict(sleep_X_eval)
sleep_mae = mean_absolute_error(train_stage.loc[sleep_eval_idx, "sleep_duration"], sleep_eval_pred)
sleep_rmse = root_mean_squared_error(train_stage.loc[sleep_eval_idx, "sleep_duration"], sleep_eval_pred)


def fill_sleep(df):
    out = df.copy()
    out["sleep_duration_was_missing"] = out["sleep_duration"].isna().astype("int8")
    mask = out["sleep_duration"].isna()
    if mask.any():
        X, _ = prepare_catboost(out.loc[mask], sleep_features)
        out.loc[mask, "sleep_duration"] = sleep_model.predict(X)
    return out


train_imputed = fill_sleep(train_stage)
valid_imputed = fill_sleep(valid_stage)


def add_health_features(df):
    df = df.copy()
    eps = 1e-6
    original = [c for c in raw_features]
    df["missing_count"] = raw.loc[df.index, original].isna().sum(axis=1).astype("int8")
    df["bmi_distance_normal"] = (df["bmi"] - 22.0).abs()
    df["sleep_distance_8h"] = (df["sleep_duration"] - 8.0).abs()
    df["steps_per_exercise_min"] = df["step_count"] / (df["exercise_duration"] + eps)
    df["calories_per_step"] = df["calorie_expenditure"] / (df["step_count"] + eps)
    df["water_per_calorie"] = df["water_intake"] / (df["calorie_expenditure"] + eps)
    return df.replace([np.inf, -np.inf], np.nan)


train_health = add_health_features(train_imputed)
valid_health = add_health_features(valid_imputed)
health_features = [c for c in train_health.columns if c not in [base.ID_COL, base.TARGET]]
health_cat = train_health[health_features].select_dtypes(exclude=np.number).columns.tolist()
for col in health_cat:
    train_health[col] = train_health[col].fillna("<MISSING>").astype(str)
    valid_health[col] = valid_health[col].fillna("<MISSING>").astype(str)

health_model = CatBoostClassifier(
    iterations=1200, learning_rate=0.05, depth=8,
    loss_function="MultiClass", eval_metric="MultiClass",
    auto_class_weights="Balanced", random_seed=RANDOM_STATE,
    verbose=100, allow_writing_files=False,
)
health_model.fit(
    train_health[health_features], base.y_train,
    cat_features=health_cat,
    eval_set=(valid_health[health_features], base.y_valid),
    early_stopping_rounds=100,
)
two_stage_prob = health_model.predict_proba(valid_health[health_features])
two_stage_pred = two_stage_prob.argmax(axis=1)
two_stage_score = balanced_accuracy_score(base.y_valid, two_stage_pred)

result_rows = []
baseline_pred = base.cat_prob.argmax(axis=1)
subsets = {
    "all": np.ones(len(raw_valid), dtype=bool),
    "stress_missing": raw_valid["stress_level"].isna().to_numpy(),
    "sleep_missing": raw_valid["sleep_duration"].isna().to_numpy(),
    "both_missing": (
        raw_valid["stress_level"].isna() & raw_valid["sleep_duration"].isna()
    ).to_numpy(),
    "neither_missing": (
        raw_valid["stress_level"].notna() & raw_valid["sleep_duration"].notna()
    ).to_numpy(),
}
for subset, mask in subsets.items():
    for model_name, pred in [("baseline_catboost", baseline_pred), ("two_stage_catboost", two_stage_pred)]:
        result_rows.append({
            "subset": subset,
            "model": model_name,
            "rows": int(mask.sum()),
            "balanced_accuracy": float(balanced_accuracy_score(base.y_valid.to_numpy()[mask], pred[mask])),
            "accuracy": float((base.y_valid.to_numpy()[mask] == pred[mask]).mean()),
            "errors": int((base.y_valid.to_numpy()[mask] != pred[mask]).sum()),
        })
comparison = pd.DataFrame(result_rows)
comparison.to_csv(OUT / "two_stage_comparison.csv", index=False)

health_importance = pd.DataFrame({
    "feature": health_features,
    "importance": normalized(health_model.feature_importances_),
}).sort_values("importance", ascending=False)
health_importance["rank"] = np.arange(1, len(health_importance) + 1)
health_importance.to_csv(OUT / "two_stage_health_feature_importance.csv", index=False)

summary = {
    "baseline_catboost_balanced_accuracy": float(base.balanced_accuracy_score(base.y_valid, baseline_pred)),
    "two_stage_catboost_balanced_accuracy": float(two_stage_score),
    "delta": float(two_stage_score - base.balanced_accuracy_score(base.y_valid, baseline_pred)),
    "stress_aux_accuracy": float(stress_accuracy),
    "sleep_aux_mae": float(sleep_mae),
    "sleep_aux_rmse": float(sleep_rmse),
    "importance_method": "model-native normalized importance",
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print("\nTOP FEATURE IMPORTANCES")
print(importance_top.to_string(index=False))
print("\nTWO-STAGE COMPARISON")
print(comparison.to_string(index=False))
print("\nSUMMARY")
print(json.dumps(summary, indent=2))
