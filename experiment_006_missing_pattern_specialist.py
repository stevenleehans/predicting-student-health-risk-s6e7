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
OUT = Path("experiment_006_artifacts")
OUT.mkdir(exist_ok=True)


def add_features(df):
    df = df.copy()
    eps = 1e-6
    base = [c for c in df.columns if c not in [ID_COL, TARGET]]
    df["missing_count"] = df[base].isna().sum(axis=1).astype("int8")
    df["bmi_distance_normal"] = (df["bmi"] - 22.0).abs()
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


def ba(y, pred):
    return balanced_accuracy_score(y, pred)


train = add_features(pd.read_csv("train.csv"))
test = add_features(pd.read_csv("test.csv"))
sample = pd.read_csv("sample_submission.csv")
y = train[TARGET].map(LABEL_TO_INT).to_numpy()
fold_map = pd.read_csv("experiment_003_artifacts/fold_assignments.csv")
assert fold_map[ID_COL].equals(train[ID_COL])
fold_id = fold_map["fold"].to_numpy()

base_oof_data = np.load("experiment_004_artifacts/oof_predictions.npz")
base_test_data = np.load("experiment_004_artifacts/test_probabilities.npz")
base_oof = base_oof_data["blend"].astype(np.float32)
base_test = base_test_data["blend"].astype(np.float32)
assert np.array_equal(base_oof_data["y"], y)

route_train = train["stress_level"].isna() & train["sleep_duration"].isna()
route_test = test["stress_level"].isna() & test["sleep_duration"].isna()

# Both routing variables and their direct sleep derivative are constant-missing
# inside this specialist population, so exclude them. All other NaNs remain.
drop_features = {ID_COL, TARGET, "stress_level", "sleep_duration"}
features = [c for c in test.columns if c not in drop_features]
cat_cols = train[features].select_dtypes(exclude=np.number).columns.tolist()

special_oof = {
    name: np.full((len(train), len(LABELS)), np.nan, dtype=np.float32)
    for name in ["CatBoost", "XGBoost", "LightGBM"]
}
special_test = {
    name: np.zeros((int(route_test.sum()), len(LABELS)), dtype=np.float64)
    for name in special_oof
}
fold_rows = []
start = time.time()

for fold in range(N_SPLITS):
    fit_mask = (fold_id != fold) & route_train.to_numpy()
    eval_mask = (fold_id == fold) & route_train.to_numpy()
    fit_idx, eval_idx = np.flatnonzero(fit_mask), np.flatnonzero(eval_mask)
    X_fit = train.iloc[fit_idx][features]
    X_eval = train.iloc[eval_idx][features]
    X_test = test.loc[route_test, features]
    y_fit, y_eval = y[fit_idx], y[eval_idx]
    counts = np.bincount(y_fit, minlength=len(LABELS))
    class_weight = len(y_fit) / (len(LABELS) * counts)
    sample_weight = class_weight[y_fit]

    cat_fit, cat_eval, cat_test = X_fit.copy(), X_eval.copy(), X_test.copy()
    for col in cat_cols:
        cat_fit[col] = cat_fit[col].fillna("<MISSING>").astype(str)
        cat_eval[col] = cat_eval[col].fillna("<MISSING>").astype(str)
        cat_test[col] = cat_test[col].fillna("<MISSING>").astype(str)
    cat_model = CatBoostClassifier(
        iterations=800, learning_rate=0.04, depth=7,
        loss_function="MultiClass", eval_metric="MultiClass",
        auto_class_weights="Balanced", random_seed=RANDOM_STATE,
        verbose=False, allow_writing_files=False, thread_count=-1,
    )
    cat_model.fit(
        cat_fit, y_fit, cat_features=cat_cols,
        eval_set=(cat_eval, y_eval), early_stopping_rounds=80,
    )
    prob = cat_model.predict_proba(cat_eval)
    special_oof["CatBoost"][eval_idx] = prob
    special_test["CatBoost"] += cat_model.predict_proba(cat_test) / N_SPLITS
    del cat_fit, cat_eval, cat_test, cat_model, prob
    gc.collect()

    native_fit, native_eval, native_test = categorical_frames(
        X_fit, X_eval, X_test, cat_cols
    )
    xgb_model = XGBClassifier(
        objective="multi:softprob", num_class=3, eval_metric="mlogloss",
        n_estimators=700, learning_rate=0.04, max_depth=7,
        min_child_weight=10, subsample=0.9, colsample_bytree=0.9,
        reg_lambda=5, tree_method="hist", device="cpu",
        enable_categorical=True, missing=np.nan,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    xgb_model.fit(native_fit, y_fit, sample_weight=sample_weight, verbose=False)
    prob = xgb_model.predict_proba(native_eval)
    special_oof["XGBoost"][eval_idx] = prob
    special_test["XGBoost"] += xgb_model.predict_proba(native_test) / N_SPLITS
    del xgb_model, prob
    gc.collect()

    lgb_model = LGBMClassifier(
        objective="multiclass", num_class=3, n_estimators=700,
        learning_rate=0.04, num_leaves=31, min_child_samples=30,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=5,
        class_weight="balanced", random_state=RANDOM_STATE,
        n_jobs=-1, verbosity=-1,
    )
    lgb_model.fit(native_fit, y_fit, categorical_feature=cat_cols)
    prob = lgb_model.predict_proba(native_eval)
    special_oof["LightGBM"][eval_idx] = prob
    special_test["LightGBM"] += lgb_model.predict_proba(native_test) / N_SPLITS
    del native_fit, native_eval, native_test, lgb_model, prob
    gc.collect()

    for name in special_oof:
        score = ba(y_eval, special_oof[name][eval_idx].argmax(axis=1))
        fold_rows.append({
            "fold": fold, "model": name,
            "train_rows": len(fit_idx), "eval_rows": len(eval_idx),
            "route_balanced_accuracy": score,
        })
    base_score = ba(y_eval, base_oof[eval_idx].argmax(axis=1))
    print(f"Fold {fold}: base={base_score:.6f} " + " ".join(
        f"{name}={fold_rows[-3+i]['route_balanced_accuracy']:.6f}"
        for i, name in enumerate(special_oof)
    ), flush=True)

fold_scores = pd.DataFrame(fold_rows)
fold_scores.to_csv(OUT / "specialist_fold_scores.csv", index=False)
route_idx = np.flatnonzero(route_train.to_numpy())
assert all(np.isfinite(special_oof[name][route_idx]).all() for name in special_oof)


def candidate_grid(indices):
    names = ["CatBoost", "XGBoost", "LightGBM"]
    results = []
    grid = np.arange(0, 1.001, 0.1)
    for w_cat in grid:
        for w_xgb in grid:
            w_lgb = 1 - w_cat - w_xgb
            if w_lgb < -1e-9:
                continue
            weights = np.array([w_cat, w_xgb, max(0.0, w_lgb)])
            specialist = sum(w * special_oof[name][indices] for w, name in zip(weights, names))
            for alpha in grid:
                routed = (1 - alpha) * base_oof[indices] + alpha * specialist
                score = ba(y[indices], routed.argmax(axis=1))
                results.append((score, alpha, *weights))
    return max(results)


# Cross-fit router parameters: select on routed rows in four folds, score on the fifth.
crossfit_routed_prob = base_oof.copy()
router_rows = []
for heldout_fold in range(N_SPLITS):
    select_idx = np.flatnonzero(route_train.to_numpy() & (fold_id != heldout_fold))
    eval_idx = np.flatnonzero(route_train.to_numpy() & (fold_id == heldout_fold))
    _, alpha, w_cat, w_xgb, w_lgb = candidate_grid(select_idx)
    weights = np.array([w_cat, w_xgb, w_lgb])
    specialist = sum(
        w * special_oof[name][eval_idx]
        for w, name in zip(weights, special_oof)
    )
    crossfit_routed_prob[eval_idx] = (
        (1 - alpha) * base_oof[eval_idx] + alpha * specialist
    )
    base_route_score = ba(y[eval_idx], base_oof[eval_idx].argmax(axis=1))
    routed_score = ba(y[eval_idx], crossfit_routed_prob[eval_idx].argmax(axis=1))
    base_global_pred = base_oof[np.flatnonzero(fold_id == heldout_fold)].argmax(axis=1)
    routed_global_pred = crossfit_routed_prob[np.flatnonzero(fold_id == heldout_fold)].argmax(axis=1)
    y_fold = y[np.flatnonzero(fold_id == heldout_fold)]
    router_rows.append({
        "fold": heldout_fold, "route_rows": len(eval_idx),
        "alpha": alpha, "catboost_weight": w_cat,
        "xgboost_weight": w_xgb, "lightgbm_weight": w_lgb,
        "base_route_balanced_accuracy": base_route_score,
        "routed_route_balanced_accuracy": routed_score,
        "route_delta": routed_score - base_route_score,
        "base_global_balanced_accuracy": ba(y_fold, base_global_pred),
        "routed_global_balanced_accuracy": ba(y_fold, routed_global_pred),
        "global_delta": ba(y_fold, routed_global_pred) - ba(y_fold, base_global_pred),
    })

router_results = pd.DataFrame(router_rows)
router_results.to_csv(OUT / "crossfit_router_results.csv", index=False)
base_global_score = ba(y, base_oof.argmax(axis=1))
crossfit_global_score = ba(y, crossfit_routed_prob.argmax(axis=1))
base_route_score = ba(y[route_idx], base_oof[route_idx].argmax(axis=1))
crossfit_route_score = ba(y[route_idx], crossfit_routed_prob[route_idx].argmax(axis=1))

# Deployment router selected on all routed OOF rows.
_, alpha, w_cat, w_xgb, w_lgb = candidate_grid(route_idx)
deployment_weights = np.array([w_cat, w_xgb, w_lgb])
special_test_blend = sum(
    w * special_test[name] for w, name in zip(deployment_weights, special_test)
)
routed_test_prob = base_test.copy()
routed_test_prob[route_test.to_numpy()] = (
    (1 - alpha) * base_test[route_test.to_numpy()] + alpha * special_test_blend
)
test_pred = routed_test_prob.argmax(axis=1)
submission = sample.copy()
submission[TARGET] = pd.Series(test_pred).map(INT_TO_LABEL)
assert submission[TARGET].notna().all()
submission.to_csv("submission_missing_pattern_specialist.csv", index=False)

summary = {
    "route": "stress_level is NaN AND sleep_duration is NaN",
    "train_route_rows": int(route_train.sum()),
    "test_route_rows": int(route_test.sum()),
    "base_global_oof_balanced_accuracy": base_global_score,
    "crossfit_routed_global_oof_balanced_accuracy": crossfit_global_score,
    "crossfit_global_delta": crossfit_global_score - base_global_score,
    "base_route_oof_balanced_accuracy": base_route_score,
    "crossfit_routed_route_oof_balanced_accuracy": crossfit_route_score,
    "crossfit_route_delta": crossfit_route_score - base_route_score,
    "folds_global_improved": int((router_results.global_delta > 0).sum()),
    "global_delta_mean": float(router_results.global_delta.mean()),
    "global_delta_std": float(router_results.global_delta.std(ddof=1)),
    "deployment_alpha": float(alpha),
    "deployment_specialist_weights": dict(zip(special_oof, deployment_weights.tolist())),
    "test_labels_changed": int((base_test.argmax(axis=1) != test_pred).sum()),
    "prediction_counts": submission[TARGET].value_counts().to_dict(),
    "elapsed_seconds": time.time() - start,
    "device": "multicore CPU; no Apple MPS backend for these libraries",
    "platform": platform.platform(),
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
np.savez_compressed(
    OUT / "specialist_predictions.npz",
    y=y.astype(np.int8), fold=fold_id,
    route_train=route_train.to_numpy(), route_test=route_test.to_numpy(),
    catboost_oof=special_oof["CatBoost"],
    xgboost_oof=special_oof["XGBoost"],
    lightgbm_oof=special_oof["LightGBM"],
    crossfit_routed_oof=crossfit_routed_prob,
    routed_test=routed_test_prob.astype(np.float32),
)
print("\nROUTER RESULTS")
print(router_results.to_string(index=False))
print("\nSUMMARY")
print(json.dumps(summary, indent=2))
