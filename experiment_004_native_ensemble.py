from pathlib import Path
import gc
import json
import platform
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import balanced_accuracy_score, recall_score
from xgboost import XGBClassifier


RANDOM_STATE = 42
N_SPLITS = 5
TARGET, ID_COL = "health_condition", "id"
LABELS = ["unhealthy", "at-risk", "fit"]
LABEL_TO_INT = {label: i for i, label in enumerate(LABELS)}
INT_TO_LABEL = {i: label for label, i in LABEL_TO_INT.items()}
OUT = Path("experiment_004_artifacts")
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


def categorical_frames(X_fit, X_eval, X_test, cat_cols):
    fit, eval_, test_ = X_fit.copy(), X_eval.copy(), X_test.copy()
    for col in cat_cols:
        levels = pd.Index(fit[col].dropna().unique())
        fit[col] = pd.Categorical(fit[col], categories=levels)
        eval_[col] = pd.Categorical(eval_[col], categories=levels)
        test_[col] = pd.Categorical(test_[col], categories=levels)
    return fit, eval_, test_


train = add_features(pd.read_csv("train.csv"))
test = add_features(pd.read_csv("test.csv"))
sample = pd.read_csv("sample_submission.csv")
features = [c for c in test.columns if c != ID_COL]
cat_cols = train[features].select_dtypes(exclude=np.number).columns.tolist()
y = train[TARGET].map(LABEL_TO_INT).to_numpy()

fold_map = pd.read_csv("experiment_003_artifacts/fold_assignments.csv")
assert fold_map[ID_COL].equals(train[ID_COL])
fold_id = fold_map["fold"].to_numpy()

xgb_saved = np.load("experiment_003_artifacts/oof_predictions.npz")
assert np.array_equal(xgb_saved["y"], y)
assert np.array_equal(xgb_saved["fold"], fold_id)
oof = {
    "CatBoost": np.zeros((len(train), len(LABELS)), dtype=np.float32),
    "XGBoost": xgb_saved["native_nan"].astype(np.float32),
    "LightGBM": np.zeros((len(train), len(LABELS)), dtype=np.float32),
}
test_prob = {
    name: np.zeros((len(test), len(LABELS)), dtype=np.float64)
    for name in oof
}
fold_rows = []
start = time.time()

for fold in range(N_SPLITS):
    fit_idx = np.flatnonzero(fold_id != fold)
    eval_idx = np.flatnonzero(fold_id == fold)
    X_fit = train.iloc[fit_idx][features]
    X_eval = train.iloc[eval_idx][features]
    y_fit, y_eval = y[fit_idx], y[eval_idx]
    counts = np.bincount(y_fit, minlength=len(LABELS))
    class_weight = len(y_fit) / (len(LABELS) * counts)
    sample_weight = class_weight[y_fit]

    # CatBoost preserves numeric NaNs natively. Categorical missingness is a
    # dedicated category because CatBoost requires categorical values as text.
    cat_fit, cat_eval, cat_test = X_fit.copy(), X_eval.copy(), test[features].copy()
    for col in cat_cols:
        cat_fit[col] = cat_fit[col].fillna("<MISSING>").astype(str)
        cat_eval[col] = cat_eval[col].fillna("<MISSING>").astype(str)
        cat_test[col] = cat_test[col].fillna("<MISSING>").astype(str)
    cat_model = CatBoostClassifier(
        iterations=1200,
        learning_rate=0.05,
        depth=8,
        loss_function="MultiClass",
        eval_metric="MultiClass",
        auto_class_weights="Balanced",
        random_seed=RANDOM_STATE,
        verbose=100,
        allow_writing_files=False,
        thread_count=-1,
    )
    cat_model.fit(
        cat_fit, y_fit, cat_features=cat_cols,
        eval_set=(cat_eval, y_eval), early_stopping_rounds=100,
    )
    cat_eval_prob = cat_model.predict_proba(cat_eval)
    oof["CatBoost"][eval_idx] = cat_eval_prob
    test_prob["CatBoost"] += cat_model.predict_proba(cat_test) / N_SPLITS
    cat_score = balanced_accuracy_score(y_eval, cat_eval_prob.argmax(axis=1))
    del cat_fit, cat_eval, cat_test, cat_model, cat_eval_prob
    gc.collect()

    # XGBoost: native numeric/categorical NaNs and native categorical splits.
    native_fit, native_eval, native_test = categorical_frames(
        X_fit, X_eval, test[features], cat_cols
    )
    xgb_model = XGBClassifier(
        objective="multi:softprob", num_class=3, eval_metric="mlogloss",
        n_estimators=1000, learning_rate=0.05, max_depth=8,
        min_child_weight=15, subsample=0.85, colsample_bytree=0.8,
        reg_lambda=5, tree_method="hist", device="cpu",
        enable_categorical=True, missing=np.nan,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    xgb_model.fit(
        native_fit, y_fit, sample_weight=sample_weight,
        eval_set=[(native_eval, y_eval)], verbose=False,
    )
    # OOF was already produced by Experiment 003 with these exact settings.
    xgb_score = balanced_accuracy_score(
        y_eval, oof["XGBoost"][eval_idx].argmax(axis=1)
    )
    test_prob["XGBoost"] += xgb_model.predict_proba(native_test) / N_SPLITS
    del xgb_model
    gc.collect()

    # LightGBM: categorical dtype and native NaNs; no median/mode imputation.
    lgb_model = LGBMClassifier(
        objective="multiclass", num_class=3, n_estimators=1200,
        learning_rate=0.04, num_leaves=63, max_depth=-1,
        min_child_samples=50, subsample=0.85, colsample_bytree=0.8,
        reg_lambda=5, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1,
    )
    lgb_model.fit(native_fit, y_fit, categorical_feature=cat_cols)
    lgb_eval_prob = lgb_model.predict_proba(native_eval)
    oof["LightGBM"][eval_idx] = lgb_eval_prob
    test_prob["LightGBM"] += lgb_model.predict_proba(native_test) / N_SPLITS
    lgb_score = balanced_accuracy_score(y_eval, lgb_eval_prob.argmax(axis=1))
    del native_fit, native_eval, native_test, lgb_model, lgb_eval_prob
    gc.collect()

    for name, score in [
        ("CatBoost", cat_score), ("XGBoost", xgb_score), ("LightGBM", lgb_score)
    ]:
        fold_rows.append({"fold": fold, "model": name, "balanced_accuracy": score})
    print(
        f"Fold {fold}: CatBoost={cat_score:.6f} XGBoost={xgb_score:.6f} "
        f"LightGBM={lgb_score:.6f}", flush=True,
    )

fold_scores = pd.DataFrame(fold_rows)
fold_scores.to_csv(OUT / "fold_scores.csv", index=False)

names = ["CatBoost", "XGBoost", "LightGBM"]
model_rows = []
for name in names:
    pred = oof[name].argmax(axis=1)
    recalls = recall_score(y, pred, labels=np.arange(len(LABELS)), average=None)
    model_rows.append({
        "model": name,
        "oof_balanced_accuracy": balanced_accuracy_score(y, pred),
        "oof_accuracy": (y == pred).mean(),
        "oof_errors": int((y != pred).sum()),
        **{f"recall_{label}": recalls[i] for i, label in enumerate(LABELS)},
    })

# Coarse convex blend search. The 0.05 grid is intentionally low-dimensional
# and avoids pretending that tiny weight changes are reliable.
candidates = []
grid = np.arange(0, 1.001, 0.05)
for w_cat in grid:
    for w_xgb in grid:
        w_lgb = 1 - w_cat - w_xgb
        if w_lgb < -1e-9:
            continue
        weights = np.array([w_cat, w_xgb, max(0.0, w_lgb)])
        prob = sum(w * oof[name] for w, name in zip(weights, names))
        score = balanced_accuracy_score(y, prob.argmax(axis=1))
        candidates.append((score, *weights))
best_score, *best_weights = max(candidates)
best_weights = np.array(best_weights)

blend_oof = sum(w * oof[name] for w, name in zip(best_weights, names))
blend_pred = blend_oof.argmax(axis=1)
blend_recalls = recall_score(y, blend_pred, labels=np.arange(len(LABELS)), average=None)
model_rows.append({
    "model": "Blend",
    "oof_balanced_accuracy": best_score,
    "oof_accuracy": (y == blend_pred).mean(),
    "oof_errors": int((y != blend_pred).sum()),
    **{f"recall_{label}": blend_recalls[i] for i, label in enumerate(LABELS)},
})
summary = pd.DataFrame(model_rows)
summary.to_csv(OUT / "summary.csv", index=False)

blend_test = sum(w * test_prob[name] for w, name in zip(best_weights, names))
submission = sample.copy()
submission[TARGET] = pd.Series(blend_test.argmax(axis=1)).map(INT_TO_LABEL)
assert submission[ID_COL].equals(test[ID_COL].reset_index(drop=True))
assert submission[TARGET].isin(LABELS).all()
assert not submission.isna().any().any()
submission.to_csv("submission_native_ensemble.csv", index=False)

np.savez_compressed(
    OUT / "oof_predictions.npz", y=y.astype(np.int8), fold=fold_id,
    catboost=oof["CatBoost"], xgboost=oof["XGBoost"],
    lightgbm=oof["LightGBM"], blend=blend_oof.astype(np.float32),
)
np.savez_compressed(
    OUT / "test_probabilities.npz",
    catboost=test_prob["CatBoost"].astype(np.float32),
    xgboost=test_prob["XGBoost"].astype(np.float32),
    lightgbm=test_prob["LightGBM"].astype(np.float32),
    blend=blend_test.astype(np.float32),
)
metadata = {
    "weights": dict(zip(names, best_weights.tolist())),
    "best_oof_balanced_accuracy": float(best_score),
    "prediction_counts": submission[TARGET].value_counts().to_dict(),
    "prediction_shares": submission[TARGET].value_counts(normalize=True).to_dict(),
    "device": "multicore CPU; no Apple MPS backend for these libraries",
    "parallelism": "all models use all CPU threads within each sequential fold",
    "elapsed_seconds": time.time() - start,
    "platform": platform.platform(),
}
(OUT / "metadata.json").write_text(json.dumps(metadata, indent=2))
print("\nSUMMARY")
print(summary.to_string(index=False))
print("\nMETADATA")
print(json.dumps(metadata, indent=2))
