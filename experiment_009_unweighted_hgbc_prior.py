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
TARGET, ID_COL = "health_condition", "id"
LABELS = ["unhealthy", "at-risk", "fit"]
LABEL_TO_INT = {label: i for i, label in enumerate(LABELS)}
INT_TO_LABEL = {i: label for label, i in LABEL_TO_INT.items()}
NUMERIC_COLS = [
    "sleep_duration", "heart_rate", "bmi", "calorie_expenditure",
    "step_count", "exercise_duration", "water_intake",
]
CATEGORICAL_COLS = [
    "diet_type", "stress_level", "sleep_quality",
    "physical_activity_level", "smoking_alcohol", "gender",
]
FEATURES = NUMERIC_COLS + CATEGORICAL_COLS
OUT = Path("experiment_009_artifacts")
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
    "class_weight": None,
    "random_state": RANDOM_STATE,
}
BETA_GRID = np.round(np.arange(0.0, 2.001, 0.01), 2)


def make_views(train, test):
    train_model, test_model = train[FEATURES].copy(), test[FEATURES].copy()
    for col in NUMERIC_COLS:
        train_model[col] = train_model[col].astype(np.float32)
        test_model[col] = test_model[col].astype(np.float32)
    for col in CATEGORICAL_COLS:
        train_values = train_model[col].astype("string").fillna("<MISSING>")
        test_values = test_model[col].astype("string").fillna("<MISSING>")
        categories = sorted(set(train_values.unique()) | set(test_values.unique()))
        train_model[col] = pd.Categorical(train_values, categories=categories)
        test_model[col] = pd.Categorical(test_values, categories=categories)
    train_te = pd.DataFrame(index=train.index)
    test_te = pd.DataFrame(index=test.index)
    for col in FEATURES:
        train_te[col] = train[col].astype("string").fillna("<MISSING>")
        test_te[col] = test[col].astype("string").fillna("<MISSING>")
    return train_model, test_model, train_te, test_te


def compose(raw, encoded, names):
    return pd.concat([
        raw.reset_index(drop=True),
        pd.DataFrame(np.asarray(encoded, dtype=np.float32), columns=names),
    ], axis=1)


def prior_correct(probability, prior, beta):
    corrected = probability / np.power(prior[None, :], beta)
    return corrected / corrected.sum(axis=1, keepdims=True)


def tune_beta(y_true, probability, prior):
    rows = []
    for beta in BETA_GRID:
        corrected = prior_correct(probability, prior, beta)
        score = balanced_accuracy_score(y_true, corrected.argmax(axis=1))
        rows.append((float(score), float(beta)))
    # Prefer beta=1 when scores tie, then the smaller beta.
    return max(rows, key=lambda item: (item[0], -abs(item[1] - 1.0), -item[1]))


def metric_row(model, y_true, probability):
    prediction = probability.argmax(axis=1)
    recalls = recall_score(y_true, prediction, labels=np.arange(3), average=None)
    return {
        "model": model,
        "oof_balanced_accuracy": balanced_accuracy_score(y_true, prediction),
        "oof_accuracy": float((prediction == y_true).mean()),
        "oof_errors": int((prediction != y_true).sum()),
        **{f"recall_{label}": float(recalls[i]) for i, label in enumerate(LABELS)},
    }


start = time.time()
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sample = pd.read_csv("sample_submission.csv")
y = train[TARGET].map(LABEL_TO_INT).to_numpy(dtype=np.int8)
global_prior = np.bincount(y, minlength=3).astype(float) / len(y)

fold_map = pd.read_csv("experiment_003_artifacts/fold_assignments.csv")
assert fold_map[ID_COL].equals(train[ID_COL])
fold_id = fold_map["fold"].to_numpy(dtype=np.int8)

exp7 = np.load("experiment_007_artifacts/oof_predictions.npz")
assert np.array_equal(exp7["y"], y)
assert np.array_equal(exp7["fold"], fold_id)
weighted_oof = exp7["hgbc_te"].astype(np.float32)

train_model, test_model, train_te, test_te = make_views(train, test)
te_names = [f"te_{column}_{klass}" for column in FEATURES for klass in range(3)]
raw_oof = np.zeros((len(train), 3), dtype=np.float32)
raw_test = np.zeros((len(test), 3), dtype=np.float64)
fold_rows = []

for fold in range(N_SPLITS):
    fit_idx = np.flatnonzero(fold_id != fold)
    eval_idx = np.flatnonzero(fold_id == fold)
    y_fit, y_eval = y[fit_idx], y[eval_idx]
    encoder = TargetEncoder(
        cv=5, smooth="auto", target_type="multiclass",
        shuffle=True, random_state=RANDOM_STATE + fold,
    )
    te_fit = encoder.fit_transform(train_te.iloc[fit_idx], y_fit)
    te_eval = encoder.transform(train_te.iloc[eval_idx])
    te_test = encoder.transform(test_te)
    X_fit = compose(train_model.iloc[fit_idx], te_fit, te_names)
    X_eval = compose(train_model.iloc[eval_idx], te_eval, te_names)
    X_test = compose(test_model, te_test, te_names)

    model = HistGradientBoostingClassifier(**HGBC_CONFIG)
    model.fit(X_fit, y_fit)
    eval_probability = model.predict_proba(X_eval).astype(np.float32)
    raw_oof[eval_idx] = eval_probability
    raw_test += model.predict_proba(X_test) / N_SPLITS
    raw_score = balanced_accuracy_score(y_eval, eval_probability.argmax(axis=1))
    beta1_score = balanced_accuracy_score(
        y_eval, prior_correct(eval_probability, global_prior, 1.0).argmax(axis=1)
    )
    fold_rows.append({
        "fold": fold, "rows": len(eval_idx), "raw_balanced_accuracy": raw_score,
        "beta1_balanced_accuracy": beta1_score, "n_iter": int(model.n_iter_),
    })
    print(
        f"Fold {fold}: raw={raw_score:.6f} beta1={beta1_score:.6f} n_iter={model.n_iter_}",
        flush=True,
    )
    del encoder, te_fit, te_eval, te_test, X_fit, X_eval, X_test, model, eval_probability
    gc.collect()

pd.DataFrame(fold_rows).to_csv(OUT / "model_fold_scores.csv", index=False)

beta1_oof = prior_correct(raw_oof, global_prior, 1.0).astype(np.float32)
crossfit_oof = np.zeros_like(raw_oof)
beta_rows = []
for fold in range(N_SPLITS):
    meta_fit = np.flatnonzero(fold_id != fold)
    meta_eval = np.flatnonzero(fold_id == fold)
    meta_prior = np.bincount(y[meta_fit], minlength=3).astype(float) / len(meta_fit)
    train_score, beta = tune_beta(y[meta_fit], raw_oof[meta_fit], meta_prior)
    corrected = prior_correct(raw_oof[meta_eval], meta_prior, beta)
    crossfit_oof[meta_eval] = corrected
    beta_rows.append({
        "fold": fold,
        "beta_learned_on_other_folds": beta,
        "meta_train_balanced_accuracy": train_score,
        "eval_balanced_accuracy": balanced_accuracy_score(y[meta_eval], corrected.argmax(axis=1)),
    })

beta_results = pd.DataFrame(beta_rows)
beta_results.to_csv(OUT / "crossfit_beta_results.csv", index=False)
full_score, full_beta = tune_beta(y, raw_oof, global_prior)
deployment_test = prior_correct(raw_test, global_prior, full_beta).astype(np.float32)

summary = pd.DataFrame([
    metric_row("experiment_007_weighted_hgbc", y, weighted_oof),
    metric_row("unweighted_raw", y, raw_oof),
    metric_row("unweighted_beta1", y, beta1_oof),
    metric_row("unweighted_crossfit_beta", y, crossfit_oof),
])
summary.to_csv(OUT / "summary.csv", index=False)

history_rows = []
for beta in BETA_GRID:
    probability = prior_correct(raw_oof, global_prior, beta)
    history_rows.append({
        "beta": float(beta),
        "balanced_accuracy": balanced_accuracy_score(y, probability.argmax(axis=1)),
    })
pd.DataFrame(history_rows).to_csv(OUT / "full_beta_history.csv", index=False)

np.savez_compressed(
    OUT / "oof_predictions.npz",
    y=y, fold=fold_id, weighted_hgbc=weighted_oof,
    unweighted_raw=raw_oof, beta1=beta1_oof, crossfit_beta=crossfit_oof,
)
np.savez_compressed(
    OUT / "test_probabilities.npz",
    unweighted_raw=raw_test.astype(np.float32),
    deployment_corrected=deployment_test,
)
submission = sample.copy()
submission[TARGET] = pd.Series(deployment_test.argmax(axis=1)).map(INT_TO_LABEL)
assert submission[ID_COL].equals(test[ID_COL].reset_index(drop=True))
assert submission[TARGET].isin(LABELS).all()
submission.to_csv("submission_experiment_009_unweighted_hgbc_prior.csv", index=False)

metadata = {
    "global_prior": dict(zip(LABELS, global_prior.tolist())),
    "beta_grid": [float(BETA_GRID.min()), float(BETA_GRID.max()), 0.01],
    "full_oof_selected_beta": full_beta,
    "full_oof_selected_beta_score_same_data": full_score,
    "hgbc_config": HGBC_CONFIG,
    "fold_source": "experiment_003_artifacts/fold_assignments.csv",
    "device": "multicore CPU; sklearn HGBC has no Apple MPS backend",
    "parallelism": f"OpenMP/native sklearn parallelism across {os.cpu_count()} logical CPUs",
    "elapsed_seconds": time.time() - start,
    "platform": platform.platform(),
    "versions": {
        "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__,
    },
}
(OUT / "metadata.json").write_text(json.dumps(metadata, indent=2))
print("\nSUMMARY")
print(summary.to_string(index=False))
print("\nCROSSFIT BETAS")
print(beta_results.to_string(index=False))
print("\nMETADATA")
print(json.dumps(metadata, indent=2))
