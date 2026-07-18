"""Experiment 011 local: exact deterministic generator features in HGBC.

Uses the fixed Experiment 003 folds, native numerical NaNs, and Experiment
007's exact-value target encoding.  The only model change is a small set of
explicit features encoding the recovered source rule and its boundaries.
"""

from pathlib import Path
import gc
import json
import os
import platform
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.preprocessing import TargetEncoder


SEED = 42
N_FOLDS = 5
TARGET = "health_condition"
ID_COL = "id"
LABELS = ["unhealthy", "at-risk", "fit"]
LABEL_TO_INT = {label: i for i, label in enumerate(LABELS)}
NUMERIC_COLS = [
    "sleep_duration", "heart_rate", "bmi", "calorie_expenditure",
    "step_count", "exercise_duration", "water_intake",
]
CATEGORICAL_COLS = [
    "diet_type", "stress_level", "sleep_quality",
    "physical_activity_level", "smoking_alcohol", "gender",
]
BASE_FEATURES = NUMERIC_COLS + CATEGORICAL_COLS
RULE_NUMERIC = [
    "rule_complete", "rule_unhealthy", "rule_fit",
    "sleep_below_6", "sleep_above_7", "sleep_distance_6", "sleep_distance_7",
]
RULE_CATEGORICAL = ["exact_rule_label", "generator_cell", "key_missing_pattern"]
MODEL_FEATURES = BASE_FEATURES + RULE_NUMERIC + RULE_CATEGORICAL
OUT = Path("experiment_011_local_artifacts")
OUT.mkdir(exist_ok=True)

HGBC_CONFIG = {
    "learning_rate": 0.0627037115235577,
    "max_iter": 300,
    "max_leaf_nodes": 33,
    "min_samples_leaf": 298,
    "l2_regularization": 0.028912644384523085,
    "max_bins": 237,
    "max_features": 0.820265066682815,
    "early_stopping": True,
    "categorical_features": "from_dtype",
    "class_weight": "balanced",
    "random_state": SEED,
}


def add_exact_rule_features(frame):
    result = frame.copy()
    complete = result[[
        "sleep_duration", "stress_level", "physical_activity_level"
    ]].notna().all(axis=1)
    high = result["stress_level"].eq("high").fillna(False)
    low = result["stress_level"].eq("low").fillna(False)
    active = result["physical_activity_level"].eq("active").fillna(False)
    unhealthy = complete & result["sleep_duration"].lt(6) & high
    fit = complete & result["sleep_duration"].ge(7) & low & active

    rule_label = pd.Series("at-risk", index=result.index, dtype="string")
    rule_label.loc[unhealthy] = "unhealthy"
    rule_label.loc[fit] = "fit"
    rule_label.loc[~complete] = "<MISSING>"
    result["exact_rule_label"] = rule_label
    result["rule_complete"] = complete.astype(np.float32)
    result["rule_unhealthy"] = unhealthy.astype(np.float32)
    result["rule_fit"] = fit.astype(np.float32)
    result["sleep_below_6"] = (6.0 - result["sleep_duration"]).clip(lower=0).astype(np.float32)
    result["sleep_above_7"] = (result["sleep_duration"] - 7.0).clip(lower=0).astype(np.float32)
    result["sleep_distance_6"] = (result["sleep_duration"] - 6.0).abs().astype(np.float32)
    result["sleep_distance_7"] = (result["sleep_duration"] - 7.0).abs().astype(np.float32)

    sleep_band = pd.cut(
        result["sleep_duration"], [-np.inf, 6.0, 7.0, np.inf],
        right=False, labels=["lt6", "6to7", "ge7"],
    ).astype("string").fillna("<MISSING>")
    stress = result["stress_level"].astype("string").fillna("<MISSING>")
    activity = result["physical_activity_level"].astype("string").fillna("<MISSING>")
    result["generator_cell"] = sleep_band + "|" + stress + "|" + activity
    result["key_missing_pattern"] = (
        result["sleep_duration"].isna().astype(str)
        + result["stress_level"].isna().astype(str)
        + result["physical_activity_level"].isna().astype(str)
    )
    return result


def make_model_views(train, test):
    views = [train[MODEL_FEATURES].copy(), test[MODEL_FEATURES].copy()]
    for col in NUMERIC_COLS + RULE_NUMERIC:
        for view in views:
            view[col] = view[col].astype(np.float32)
    for col in CATEGORICAL_COLS + RULE_CATEGORICAL:
        values = [view[col].astype("string").fillna("<MISSING>") for view in views]
        categories = sorted(set(values[0].unique()) | set(values[1].unique()))
        for view, series in zip(views, values):
            view[col] = pd.Categorical(series, categories=categories)
    return views


def exact_te_view(frame):
    return pd.DataFrame({
        col: frame[col].astype("string").fillna("<MISSING>")
        for col in BASE_FEATURES
    })


def compose(raw, encoded, names):
    return pd.concat([
        raw.reset_index(drop=True),
        pd.DataFrame(np.asarray(encoded, dtype=np.float32), columns=names),
    ], axis=1)


def metric_row(name, y, probability):
    pred = probability.argmax(1)
    recalls = recall_score(y, pred, labels=np.arange(3), average=None)
    return {
        "candidate": name,
        "oof_balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "accuracy": float(np.mean(y == pred)),
        "errors": int(np.sum(y != pred)),
        **{f"recall_{label}": float(recalls[i]) for i, label in enumerate(LABELS)},
    }


def best_weight(y, left, right):
    candidates = []
    for right_weight in np.arange(0.0, 1.001, 0.01):
        probability = (1.0 - right_weight) * left + right_weight * right
        score = balanced_accuracy_score(y, probability.argmax(1))
        candidates.append((float(score), float(right_weight)))
    return max(candidates, key=lambda item: (item[0], -item[1]))


def crossfit_blend(y, fold_id, left, right):
    output = np.zeros_like(left)
    rows = []
    for fold in range(N_FOLDS):
        meta = fold_id != fold
        valid = ~meta
        meta_score, weight = best_weight(y[meta], left[meta], right[meta])
        output[valid] = (1.0 - weight) * left[valid] + weight * right[valid]
        rows.append({
            "fold": fold,
            "right_weight": weight,
            "meta_balanced_accuracy": meta_score,
            "heldout_balanced_accuracy": float(
                balanced_accuracy_score(y[valid], output[valid].argmax(1))
            ),
        })
    return output, rows


def main():
    start = time.time()
    train_raw = pd.read_csv("train.csv")
    test_raw = pd.read_csv("test.csv")
    train = add_exact_rule_features(train_raw)
    test = add_exact_rule_features(test_raw)
    train_model, test_model = make_model_views(train, test)
    train_te, test_te = exact_te_view(train), exact_te_view(test)
    te_names = [f"te_{col}_{klass}" for col in BASE_FEATURES for klass in range(3)]

    y = train[TARGET].map(LABEL_TO_INT).to_numpy(np.int8)
    fold_map = pd.read_csv("experiment_003_artifacts/fold_assignments.csv")
    assert fold_map[ID_COL].equals(train[ID_COL])
    fold_id = fold_map["fold"].to_numpy(np.int8)
    base = np.load("experiment_010_artifacts/oof_predictions.npz")
    assert np.array_equal(base["y"], y) and np.array_equal(base["fold"], fold_id)
    old_hgbc = base["hgbc"].astype(np.float32)
    realmlp = base["realmlp"].astype(np.float32)
    trusted_blend = base["crossfit_blend"].astype(np.float32)

    oof = np.zeros((len(train), 3), np.float32)
    fold_rows = []
    for fold in range(N_FOLDS):
        fold_start = time.time()
        fit_idx = np.flatnonzero(fold_id != fold)
        valid_idx = np.flatnonzero(fold_id == fold)
        encoder = TargetEncoder(
            cv=5, smooth="auto", target_type="multiclass", shuffle=True,
            random_state=SEED + fold,
        )
        te_fit = encoder.fit_transform(train_te.iloc[fit_idx], y[fit_idx])
        te_valid = encoder.transform(train_te.iloc[valid_idx])
        x_fit = compose(train_model.iloc[fit_idx], te_fit, te_names)
        x_valid = compose(train_model.iloc[valid_idx], te_valid, te_names)
        model = HistGradientBoostingClassifier(**HGBC_CONFIG)
        model.fit(x_fit, y[fit_idx])
        oof[valid_idx] = model.predict_proba(x_valid).astype(np.float32)
        score = balanced_accuracy_score(y[valid_idx], oof[valid_idx].argmax(1))
        fold_rows.append({
            "fold": fold, "balanced_accuracy": float(score),
            "n_iter": int(model.n_iter_),
            "runtime_seconds": time.time() - fold_start,
        })
        print(
            f"Fold {fold}: rule-HGBC={score:.6f} n_iter={model.n_iter_} "
            f"runtime={time.time() - fold_start:.1f}s",
            flush=True,
        )
        np.savez_compressed(
            OUT / "checkpoint.npz", completed_fold=fold, y=y, fold=fold_id,
            exact_rule_hgbc=oof,
        )
        del encoder, te_fit, te_valid, x_fit, x_valid, model
        gc.collect()

    rule_real_blend, blend_rows = crossfit_blend(y, fold_id, oof, realmlp)
    old_rule_blend, old_rule_rows = crossfit_blend(y, fold_id, trusted_blend, oof)
    summary = pd.DataFrame([
        metric_row("experiment_010_trusted_blend", y, trusted_blend),
        metric_row("experiment_007_hgbc", y, old_hgbc),
        metric_row("exact_rule_hgbc", y, oof),
        metric_row("exact_rule_hgbc_realmlp_crossfit", y, rule_real_blend),
        metric_row("experiment010_plus_exact_rule_hgbc_crossfit", y, old_rule_blend),
    ]).sort_values("oof_balanced_accuracy", ascending=False)

    missing = train[[
        "sleep_duration", "stress_level", "physical_activity_level"
    ]].isna().astype(int).astype(str).agg("".join, axis=1)
    slices = []
    for name, probability in [
        ("experiment_010_trusted_blend", trusted_blend),
        ("exact_rule_hgbc", oof),
        ("exact_rule_hgbc_realmlp_crossfit", rule_real_blend),
    ]:
        prediction = probability.argmax(1)
        for pattern in sorted(missing.unique()):
            mask = missing.eq(pattern).to_numpy()
            slices.append({
                "candidate": name, "missing_pattern": pattern,
                "rows": int(mask.sum()),
                "balanced_accuracy": float(balanced_accuracy_score(y[mask], prediction[mask])),
                "error_rate": float(np.mean(y[mask] != prediction[mask])),
            })

    pd.DataFrame(fold_rows).to_csv(OUT / "fold_scores.csv", index=False)
    pd.DataFrame(blend_rows).to_csv(OUT / "rule_realmlp_blend_folds.csv", index=False)
    pd.DataFrame(old_rule_rows).to_csv(OUT / "trusted_rule_blend_folds.csv", index=False)
    summary.to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(slices).to_csv(OUT / "missing_pattern_slices.csv", index=False)
    np.savez_compressed(
        OUT / "oof_predictions.npz", y=y, fold=fold_id,
        trusted_blend=trusted_blend, old_hgbc=old_hgbc,
        exact_rule_hgbc=oof, rule_realmlp_blend=rule_real_blend,
        trusted_rule_blend=old_rule_blend,
    )
    metadata = {
        "experiment": 11,
        "scope": "local fixed-fold CV only; no Kaggle execution or leaderboard tuning",
        "folds": "Experiment 003 fixed five folds",
        "features": MODEL_FEATURES + te_names,
        "hgbc_config": HGBC_CONFIG,
        "device": "multicore CPU; sklearn HGBC has no MPS backend",
        "parallelism": f"native OpenMP across {os.cpu_count()} logical CPUs",
        "runtime_seconds": time.time() - start,
        "platform": platform.platform(),
        "versions": {
            "numpy": np.__version__, "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
        },
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print("\nSUMMARY")
    print(summary.to_string(index=False))
    print("\nMETADATA")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
