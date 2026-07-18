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


RANDOM_STATE = 42
N_SPLITS = 5
TARGET = "health_condition"
ID_COL = "id"
LABELS = ["unhealthy", "at-risk", "fit"]
LABEL_TO_INT = {label: i for i, label in enumerate(LABELS)}
INT_TO_LABEL = {i: label for label, i in LABEL_TO_INT.items()}

NUMERIC_COLS = [
    "sleep_duration",
    "heart_rate",
    "bmi",
    "calorie_expenditure",
    "step_count",
    "exercise_duration",
    "water_intake",
]
CATEGORICAL_COLS = [
    "diet_type",
    "stress_level",
    "sleep_quality",
    "physical_activity_level",
    "smoking_alcohol",
    "gender",
]
BASE_FEATURES = NUMERIC_COLS + CATEGORICAL_COLS

OUT = Path("experiment_007_artifacts")
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
    "random_state": RANDOM_STATE,
}


def make_model_views(train, test):
    train_model = train[BASE_FEATURES].copy()
    test_model = test[BASE_FEATURES].copy()

    for col in NUMERIC_COLS:
        train_model[col] = train_model[col].astype(np.float32)
        test_model[col] = test_model[col].astype(np.float32)

    for col in CATEGORICAL_COLS:
        train_values = train_model[col].astype("string").fillna("<MISSING>")
        test_values = test_model[col].astype("string").fillna("<MISSING>")
        categories = sorted(set(train_values.unique()) | set(test_values.unique()))
        train_model[col] = pd.Categorical(train_values, categories=categories)
        test_model[col] = pd.Categorical(test_values, categories=categories)

    return train_model, test_model


def make_exact_value_te_view(df):
    view = pd.DataFrame(index=df.index)
    for col in BASE_FEATURES:
        # String conversion preserves every repeated exact numeric value as its
        # own category. Missing values remain a distinct, explicit level.
        view[col] = df[col].astype("string").fillna("<MISSING>")
    return view


def compose_model_frame(raw_frame, te_values, te_names):
    raw = raw_frame.reset_index(drop=True)
    te = pd.DataFrame(
        np.asarray(te_values, dtype=np.float32),
        columns=te_names,
    )
    return pd.concat([raw, te], axis=1)


def metrics_row(name, y_true, probabilities):
    pred = probabilities.argmax(axis=1)
    recalls = recall_score(
        y_true,
        pred,
        labels=np.arange(len(LABELS)),
        average=None,
    )
    return {
        "model": name,
        "oof_balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "oof_accuracy": float((y_true == pred).mean()),
        "oof_errors": int((y_true != pred).sum()),
        **{f"recall_{label}": float(recalls[i]) for i, label in enumerate(LABELS)},
    }


def best_hgbc_weight(y_true, base_prob, hgbc_prob, step=0.01):
    rows = []
    for weight in np.arange(0.0, 1.0 + step / 2, step):
        probability = (1.0 - weight) * base_prob + weight * hgbc_prob
        score = balanced_accuracy_score(y_true, probability.argmax(axis=1))
        rows.append((float(score), float(weight)))
    # Prefer the simpler baseline when scores tie exactly.
    return max(rows, key=lambda item: (item[0], -item[1]))


start = time.time()
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sample = pd.read_csv("sample_submission.csv")

assert set(BASE_FEATURES).issubset(train.columns)
assert set(BASE_FEATURES).issubset(test.columns)
assert list(sample.columns) == [ID_COL, TARGET]

y = train[TARGET].map(LABEL_TO_INT).to_numpy(dtype=np.int8)
assert not np.isnan(y).any()

fold_map = pd.read_csv("experiment_003_artifacts/fold_assignments.csv")
assert fold_map[ID_COL].equals(train[ID_COL])
fold_id = fold_map["fold"].to_numpy(dtype=np.int8)
assert set(np.unique(fold_id)) == set(range(N_SPLITS))

base_oof_saved = np.load("experiment_004_artifacts/oof_predictions.npz")
base_test_saved = np.load("experiment_004_artifacts/test_probabilities.npz")
assert np.array_equal(base_oof_saved["y"], y)
assert np.array_equal(base_oof_saved["fold"], fold_id)
base_oof = base_oof_saved["blend"].astype(np.float32)
base_test = base_test_saved["blend"].astype(np.float32)

train_model, test_model = make_model_views(train, test)
train_te = make_exact_value_te_view(train)
test_te = make_exact_value_te_view(test)
te_names = [f"te_{column}_{klass}" for column in BASE_FEATURES for klass in range(len(LABELS))]

oof_hgbc = np.zeros((len(train), len(LABELS)), dtype=np.float32)
test_hgbc = np.zeros((len(test), len(LABELS)), dtype=np.float64)
fold_rows = []

for fold in range(N_SPLITS):
    fit_idx = np.flatnonzero(fold_id != fold)
    eval_idx = np.flatnonzero(fold_id == fold)
    y_fit = y[fit_idx]
    y_eval = y[eval_idx]

    encoder = TargetEncoder(
        cv=5,
        smooth="auto",
        target_type="multiclass",
        shuffle=True,
        random_state=RANDOM_STATE + fold,
    )
    te_fit = encoder.fit_transform(train_te.iloc[fit_idx], y_fit)
    te_eval = encoder.transform(train_te.iloc[eval_idx])
    te_test = encoder.transform(test_te)

    X_fit = compose_model_frame(train_model.iloc[fit_idx], te_fit, te_names)
    X_eval = compose_model_frame(train_model.iloc[eval_idx], te_eval, te_names)
    X_test = compose_model_frame(test_model, te_test, te_names)

    model = HistGradientBoostingClassifier(**HGBC_CONFIG)
    model.fit(X_fit, y_fit)
    eval_prob = model.predict_proba(X_eval).astype(np.float32)
    oof_hgbc[eval_idx] = eval_prob
    test_hgbc += model.predict_proba(X_test) / N_SPLITS

    fold_score = balanced_accuracy_score(y_eval, eval_prob.argmax(axis=1))
    fold_rows.append({
        "fold": fold,
        "rows": len(eval_idx),
        "balanced_accuracy": float(fold_score),
        "n_iter": int(model.n_iter_),
    })
    print(
        f"Fold {fold}: HGBC-TE={fold_score:.6f} n_iter={model.n_iter_}",
        flush=True,
    )

    np.savez_compressed(
        OUT / "checkpoint_predictions.npz",
        completed_fold=np.array([fold], dtype=np.int8),
        y=y,
        fold=fold_id,
        hgbc_te=oof_hgbc,
        test_hgbc_te=test_hgbc.astype(np.float32),
    )

    del encoder, te_fit, te_eval, te_test
    del X_fit, X_eval, X_test, model, eval_prob
    gc.collect()

fold_scores = pd.DataFrame(fold_rows)
fold_scores.to_csv(OUT / "fold_scores.csv", index=False)

# Fit blend weights on four completed OOF folds and evaluate on the fifth.
crossfit_blend = np.zeros_like(oof_hgbc)
blend_rows = []
for fold in range(N_SPLITS):
    train_meta_idx = np.flatnonzero(fold_id != fold)
    eval_meta_idx = np.flatnonzero(fold_id == fold)
    train_score, weight = best_hgbc_weight(
        y[train_meta_idx],
        base_oof[train_meta_idx],
        oof_hgbc[train_meta_idx],
    )
    probability = (
        (1.0 - weight) * base_oof[eval_meta_idx]
        + weight * oof_hgbc[eval_meta_idx]
    )
    crossfit_blend[eval_meta_idx] = probability
    blend_rows.append({
        "fold": fold,
        "hgbc_weight_learned_on_other_folds": weight,
        "meta_train_balanced_accuracy": train_score,
        "base_eval_balanced_accuracy": balanced_accuracy_score(
            y[eval_meta_idx], base_oof[eval_meta_idx].argmax(axis=1)
        ),
        "hgbc_eval_balanced_accuracy": balanced_accuracy_score(
            y[eval_meta_idx], oof_hgbc[eval_meta_idx].argmax(axis=1)
        ),
        "blend_eval_balanced_accuracy": balanced_accuracy_score(
            y[eval_meta_idx], probability.argmax(axis=1)
        ),
    })

blend_fold_scores = pd.DataFrame(blend_rows)
blend_fold_scores.to_csv(OUT / "crossfit_blend_fold_scores.csv", index=False)

full_blend_score, full_hgbc_weight = best_hgbc_weight(y, base_oof, oof_hgbc)
deployment_blend_test = (
    (1.0 - full_hgbc_weight) * base_test
    + full_hgbc_weight * test_hgbc.astype(np.float32)
)

summary = pd.DataFrame([
    metrics_row("experiment_004_base", y, base_oof),
    metrics_row("hgbc_exact_value_te", y, oof_hgbc),
    metrics_row("crossfit_blend", y, crossfit_blend),
])
summary.to_csv(OUT / "summary.csv", index=False)

sleep_missing = train["sleep_duration"].isna().to_numpy()
error_rows = []
for name, probabilities in [
    ("experiment_004_base", base_oof),
    ("hgbc_exact_value_te", oof_hgbc),
    ("crossfit_blend", crossfit_blend),
]:
    pred = probabilities.argmax(axis=1)
    for segment, mask in [
        ("sleep_missing", sleep_missing),
        ("sleep_present", ~sleep_missing),
    ]:
        error_rows.append({
            "model": name,
            "segment": segment,
            "rows": int(mask.sum()),
            "error_rate": float((pred[mask] != y[mask]).mean()),
            "balanced_accuracy": float(balanced_accuracy_score(y[mask], pred[mask])),
        })
error_slices = pd.DataFrame(error_rows)
error_slices.to_csv(OUT / "sleep_missing_error_slices.csv", index=False)

np.savez_compressed(
    OUT / "oof_predictions.npz",
    y=y,
    fold=fold_id,
    base=base_oof,
    hgbc_te=oof_hgbc,
    crossfit_blend=crossfit_blend,
)
np.savez_compressed(
    OUT / "test_probabilities.npz",
    base=base_test,
    hgbc_te=test_hgbc.astype(np.float32),
    deployment_blend=deployment_blend_test.astype(np.float32),
)

for filename, probability in [
    ("submission_experiment_007_hgbc_te.csv", test_hgbc),
    ("submission_experiment_007_blend.csv", deployment_blend_test),
]:
    submission = sample.copy()
    submission[TARGET] = pd.Series(probability.argmax(axis=1)).map(INT_TO_LABEL)
    assert submission[ID_COL].equals(test[ID_COL].reset_index(drop=True))
    assert submission[TARGET].isin(LABELS).all()
    assert not submission.isna().any().any()
    submission.to_csv(filename, index=False)

base_score = float(summary.loc[summary.model == "experiment_004_base", "oof_balanced_accuracy"].iloc[0])
hgbc_score = float(summary.loc[summary.model == "hgbc_exact_value_te", "oof_balanced_accuracy"].iloc[0])
crossfit_score = float(summary.loc[summary.model == "crossfit_blend", "oof_balanced_accuracy"].iloc[0])
metadata = {
    "random_state": RANDOM_STATE,
    "n_splits": N_SPLITS,
    "fold_source": "experiment_003_artifacts/fold_assignments.csv",
    "metric": "balanced_accuracy",
    "representation": "13 raw features plus 39 inner-cross-fitted exact-value multiclass target-encoding columns",
    "categorical_missing": "explicit <MISSING> category",
    "numeric_missing": "native NaN in raw numeric columns; explicit <MISSING> in target-encoding view",
    "hgbc_config": HGBC_CONFIG,
    "base_oof_balanced_accuracy": base_score,
    "hgbc_oof_balanced_accuracy": hgbc_score,
    "hgbc_delta_vs_base": hgbc_score - base_score,
    "crossfit_blend_oof_balanced_accuracy": crossfit_score,
    "crossfit_blend_delta_vs_base": crossfit_score - base_score,
    "full_oof_deployment_hgbc_weight": full_hgbc_weight,
    "full_oof_deployment_blend_score_same_data": full_blend_score,
    "device": "multicore CPU; sklearn HistGradientBoosting has no Apple MPS backend",
    "parallelism": f"OpenMP/native sklearn parallelism across {os.cpu_count()} logical CPUs; folds sequential to control memory",
    "elapsed_seconds": time.time() - start,
    "platform": platform.platform(),
    "versions": {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
    },
}
(OUT / "metadata.json").write_text(json.dumps(metadata, indent=2))

print("\nSUMMARY")
print(summary.to_string(index=False))
print("\nCROSSFIT BLEND")
print(blend_fold_scores.to_string(index=False))
print("\nSLEEP-MISSING ERROR SLICES")
print(error_slices.to_string(index=False))
print("\nMETADATA")
print(json.dumps(metadata, indent=2))
