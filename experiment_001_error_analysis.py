from pathlib import Path
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier


RANDOM_STATE = 42
TARGET, ID_COL = "health_condition", "id"
LABELS = ["unhealthy", "at-risk", "fit"]
LABEL_TO_INT = {label: i for i, label in enumerate(LABELS)}
INT_TO_LABEL = {i: label for label, i in LABEL_TO_INT.items()}
OUT = Path("experiment_001_artifacts")
OUT.mkdir(exist_ok=True)


def add_features(df):
    df = df.copy()
    eps = 1e-6
    base = [c for c in df.columns if c not in [ID_COL, TARGET]]
    df["missing_count"] = df[base].isna().sum(axis=1).astype("int8")
    df["bmi_distance_normal"] = (df["bmi"] - 22.0).abs()
    df["sleep_distance_8h"] = (df["sleep_duration"] - 8.0).abs()
    df["steps_per_exercise_min"] = df["step_count"] / (df["exercise_duration"] + eps)
    df["calories_per_step"] = df["calorie_expenditure"] / (df["step_count"] + eps)
    df["water_per_calorie"] = df["water_intake"] / (df["calorie_expenditure"] + eps)
    return df.replace([np.inf, -np.inf], np.nan)


train_raw = pd.read_csv("train.csv")
train = add_features(train_raw)
features = [c for c in train.columns if c not in [ID_COL, TARGET]]
cat_cols = train[features].select_dtypes(exclude=np.number).columns.tolist()
num_cols = [c for c in features if c not in cat_cols]

train_idx, valid_idx = train_test_split(
    np.arange(len(train)), test_size=0.20,
    stratify=train[TARGET], random_state=RANDOM_STATE,
)
X_train = train.loc[train_idx, features].copy()
X_valid = train.loc[valid_idx, features].copy()
y_train = train.loc[train_idx, TARGET].map(LABEL_TO_INT)
y_valid = train.loc[valid_idx, TARGET].map(LABEL_TO_INT)


def catboost_frame(df):
    out = df.copy()
    for col in cat_cols:
        out[col] = out[col].fillna("<MISSING>").astype(str)
    return out


cat_model = CatBoostClassifier(
    iterations=1200, learning_rate=0.05, depth=8,
    loss_function="MultiClass", eval_metric="MultiClass",
    auto_class_weights="Balanced", random_seed=RANDOM_STATE,
    verbose=100, allow_writing_files=False,
)
X_train_cat, X_valid_cat = catboost_frame(X_train), catboost_frame(X_valid)
cat_model.fit(
    X_train_cat, y_train, cat_features=cat_cols,
    eval_set=(X_valid_cat, y_valid), early_stopping_rounds=100,
)
cat_prob = cat_model.predict_proba(X_valid_cat)

preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median", add_indicator=True), num_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1,
            encoded_missing_value=-1, dtype=np.float32,
        )),
    ]), cat_cols),
], verbose_feature_names_out=False)
X_train_enc = preprocessor.fit_transform(X_train)
X_valid_enc = preprocessor.transform(X_valid)
class_weights = (1 / y_train.value_counts(normalize=True)).to_dict()
sample_weight = y_train.map(class_weights).to_numpy()

xgb_model = XGBClassifier(
    objective="multi:softprob", num_class=3, eval_metric="mlogloss",
    n_estimators=1000, learning_rate=0.05, max_depth=8,
    min_child_weight=15, subsample=0.85, colsample_bytree=0.8,
    reg_lambda=5, tree_method="hist", random_state=RANDOM_STATE, n_jobs=-1,
)
xgb_model.fit(
    X_train_enc, y_train, sample_weight=sample_weight,
    eval_set=[(X_valid_enc, y_valid)], verbose=100,
)
xgb_prob = xgb_model.predict_proba(X_valid_enc)

lgb_model = LGBMClassifier(
    objective="multiclass", num_class=3, n_estimators=1200,
    learning_rate=0.04, num_leaves=63, max_depth=-1,
    min_child_samples=50, subsample=0.85, colsample_bytree=0.8,
    reg_lambda=5, class_weight="balanced",
    random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1,
)
lgb_model.fit(X_train_enc, y_train)
lgb_prob = lgb_model.predict_proba(X_valid_enc)

probs = [cat_prob, xgb_prob, lgb_prob]
names = ["CatBoost", "XGBoost", "LightGBM"]
grid = np.arange(0, 1.01, 0.1)
candidates = []
for w_cat in grid:
    for w_xgb in grid:
        w_lgb = 1 - w_cat - w_xgb
        if w_lgb < -1e-9:
            continue
        weights = np.array([w_cat, w_xgb, max(0, w_lgb)])
        blend = sum(w * p for w, p in zip(weights, probs))
        score = balanced_accuracy_score(y_valid, blend.argmax(axis=1))
        candidates.append((score, *weights))
best_score, *best_weights = max(candidates)
best_weights = np.array(best_weights)
blend_prob = sum(w * p for w, p in zip(best_weights, probs))
pred = blend_prob.argmax(axis=1)

analysis = train_raw.loc[valid_idx].copy().reset_index(drop=True)
analysis["true_label"] = y_valid.map(INT_TO_LABEL).reset_index(drop=True)
analysis["pred_label"] = pd.Series(pred).map(INT_TO_LABEL)
analysis["is_error"] = analysis["true_label"] != analysis["pred_label"]
analysis["confidence"] = blend_prob.max(axis=1)
sorted_prob = np.sort(blend_prob, axis=1)
analysis["margin"] = sorted_prob[:, -1] - sorted_prob[:, -2]
analysis["missing_count"] = analysis[features[:13]].isna().sum(axis=1)
for i, label in enumerate(LABELS):
    analysis[f"prob_{label}"] = blend_prob[:, i]

# Outliers are defined from the training-fold distribution only. A conservative
# 3*IQR fence avoids labelling ordinary skewed observations as extreme.
raw_num_cols = train_raw.drop(columns=[ID_COL, TARGET]).select_dtypes(include=np.number).columns.tolist()
outlier_cols = []
outlier_bounds = []
for col in raw_num_cols:
    q1, q3 = X_train[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
    flag_col = f"outlier__{col}"
    analysis[flag_col] = analysis[col].notna() & ((analysis[col] < lo) | (analysis[col] > hi))
    outlier_cols.append(flag_col)
    outlier_bounds.append({"feature": col, "lower": lo, "upper": hi,
                           "outlier_n": int(analysis[flag_col].sum())})
analysis["outlier_count"] = analysis[outlier_cols].sum(axis=1)
analysis["has_outlier"] = analysis["outlier_count"] > 0

analysis.to_csv(OUT / "validation_predictions.csv", index=False)

conf = pd.DataFrame(
    confusion_matrix(analysis["true_label"], analysis["pred_label"], labels=LABELS),
    index=pd.Index(LABELS, name="true"), columns=pd.Index(LABELS, name="pred"),
)
conf.to_csv(OUT / "confusion_matrix.csv")

class_error = analysis.groupby("true_label", observed=True).agg(
    rows=("is_error", "size"), errors=("is_error", "sum"),
    error_rate=("is_error", "mean"), mean_confidence=("confidence", "mean"),
).reindex(LABELS)
class_error.to_csv(OUT / "class_error.csv")

missing_error = analysis.groupby("missing_count", observed=True).agg(
    rows=("is_error", "size"), errors=("is_error", "sum"),
    error_rate=("is_error", "mean"), mean_confidence=("confidence", "mean"),
).reset_index()
missing_error.to_csv(OUT / "missingness_error.csv", index=False)

outlier_error = analysis.groupby("has_outlier", observed=True).agg(
    rows=("is_error", "size"), errors=("is_error", "sum"),
    error_rate=("is_error", "mean"), mean_confidence=("confidence", "mean"),
).reset_index()
outlier_error.to_csv(OUT / "outlier_error.csv", index=False)
pd.DataFrame(outlier_bounds).to_csv(OUT / "outlier_bounds.csv", index=False)

subgroups = []
for col in cat_cols:
    temp = analysis.assign(_group=analysis[col].fillna("<MISSING>"))
    table = temp.groupby("_group", observed=True).agg(
        rows=("is_error", "size"), errors=("is_error", "sum"),
        error_rate=("is_error", "mean"), mean_confidence=("confidence", "mean"),
    ).reset_index().rename(columns={"_group": "group"})
    table.insert(0, "feature", col)
    subgroups.append(table)
subgroup_error = pd.concat(subgroups, ignore_index=True)
subgroup_error.to_csv(OUT / "categorical_subgroup_error.csv", index=False)

num_error = []
for col in raw_num_cols:
    for status, part in analysis.groupby("is_error"):
        num_error.append({
            "feature": col, "is_error": bool(status), "rows": len(part),
            "mean": part[col].mean(), "median": part[col].median(),
            "missing_rate": part[col].isna().mean(),
        })
pd.DataFrame(num_error).to_csv(OUT / "numeric_error_profile.csv", index=False)

high_conf_errors = analysis.loc[analysis.is_error].sort_values(
    ["confidence", "margin"], ascending=False
).head(1000)
high_conf_errors.to_csv(OUT / "high_confidence_errors.csv", index=False)

summary = {
    "random_state": RANDOM_STATE,
    "train_rows": int(len(train_idx)),
    "validation_rows": int(len(valid_idx)),
    "balanced_accuracy": float(best_score),
    "accuracy": float((~analysis.is_error).mean()),
    "error_count": int(analysis.is_error.sum()),
    "weights": {name: float(w) for name, w in zip(names, best_weights)},
    "single_model_balanced_accuracy": {
        name: float(balanced_accuracy_score(y_valid, p.argmax(axis=1)))
        for name, p in zip(names, probs)
    },
    "mean_confidence_correct": float(analysis.loc[~analysis.is_error, "confidence"].mean()),
    "mean_confidence_error": float(analysis.loc[analysis.is_error, "confidence"].mean()),
    "high_confidence_errors_ge_0_90": int((analysis.is_error & (analysis.confidence >= 0.90)).sum()),
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
