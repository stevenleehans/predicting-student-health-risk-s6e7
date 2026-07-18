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
    "sleep_duration", "heart_rate", "bmi", "calorie_expenditure",
    "step_count", "exercise_duration", "water_intake",
]
CATEGORICAL_COLS = [
    "diet_type", "stress_level", "sleep_quality",
    "physical_activity_level", "smoking_alcohol", "gender",
]
BASE_FEATURES = NUMERIC_COLS + CATEGORICAL_COLS
GENERATOR_NUMERIC = [
    "source_p_unhealthy", "source_p_at-risk", "source_p_fit",
    "source_prior_confidence", "source_prior_entropy", "source_rule_score",
]
GENERATOR_CATEGORICAL = ["source_rule_label", "source_generator_cell"]
MODEL_FEATURES = BASE_FEATURES + GENERATOR_NUMERIC + GENERATOR_CATEGORICAL

OUT = Path("experiment_008_artifacts")
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


def sleep_band(series):
    return pd.cut(
        series,
        [-np.inf, 6.0, 7.0, np.inf],
        right=False,
        labels=["lt6", "6to7", "ge7"],
    ).astype("string")


def source_probability_tables(source, alpha=20.0):
    work = source.copy()
    work["_sleep_band"] = sleep_band(work["sleep_duration"])
    keys = ["_sleep_band", "stress_level", "physical_activity_level"]
    global_counts = (
        work[TARGET].value_counts().reindex(LABELS, fill_value=0).to_numpy(float)
    )
    global_probability = global_counts / global_counts.sum()
    tables = {}
    for bits in range(1, 1 << len(keys)):
        subset = tuple(keys[i] for i in range(len(keys)) if bits & (1 << i))
        counts = (
            work.groupby(list(subset), observed=True)[TARGET]
            .value_counts()
            .unstack(fill_value=0)
            .reindex(columns=LABELS, fill_value=0)
        )
        probabilities = (
            counts.to_numpy(float) + alpha * global_probability[None, :]
        ) / (counts.sum(axis=1).to_numpy()[:, None] + alpha)
        tables[subset] = pd.DataFrame(
            probabilities,
            index=counts.index,
            columns=LABELS,
        )
    return keys, global_probability, tables


def add_generator_features(df, keys, global_probability, tables):
    work = df.copy()
    work["_sleep_band"] = sleep_band(work["sleep_duration"])
    probabilities = np.zeros((len(work), len(LABELS)), dtype=np.float32)

    availability = work[keys].notna()
    for pattern in availability.drop_duplicates().itertuples(index=False, name=None):
        mask = np.ones(len(work), dtype=bool)
        for key, present in zip(keys, pattern):
            mask &= work[key].notna().to_numpy() == present
        subset = tuple(key for key, present in zip(keys, pattern) if present)
        if not subset:
            probabilities[mask] = global_probability
            continue

        table = tables[subset]
        if len(subset) == 1:
            index = pd.Index(work.loc[mask, subset[0]])
        else:
            index = pd.MultiIndex.from_frame(work.loc[mask, list(subset)])
        values = table.reindex(index).to_numpy(dtype=np.float32)
        unseen = np.isnan(values).any(axis=1)
        values[unseen] = global_probability
        probabilities[mask] = values

    for klass, label in enumerate(LABELS):
        work[f"source_p_{label}"] = probabilities[:, klass]
    work["source_prior_confidence"] = probabilities.max(axis=1)
    work["source_prior_entropy"] = -np.sum(
        probabilities * np.log(np.clip(probabilities, 1e-9, 1.0)), axis=1
    )

    stress_score = work["stress_level"].map({"low": 0.0, "medium": 1.0, "high": 3.0})
    activity_score = work["physical_activity_level"].map(
        {"active": 0.0, "moderate": 1.0, "sedentary": 2.0}
    )
    deprived_score = (work["sleep_duration"] < 6.0).astype(float) * 2.0
    missing_rule = (
        work["sleep_duration"].isna()
        | work["stress_level"].isna()
        | work["physical_activity_level"].isna()
    )
    rule_score = deprived_score + stress_score + activity_score
    rule_score = rule_score.mask(missing_rule)
    work["source_rule_score"] = rule_score.astype(np.float32)
    work["source_rule_label"] = pd.cut(
        rule_score,
        [-np.inf, 0.0, 4.0, np.inf],
        labels=["fit", "at-risk", "unhealthy"],
    ).astype("string").fillna("<MISSING>")
    work["source_generator_cell"] = (
        work["_sleep_band"].fillna("<MISSING>")
        + "|" + work["stress_level"].astype("string").fillna("<MISSING>")
        + "|" + work["physical_activity_level"].astype("string").fillna("<MISSING>")
    )
    return work.drop(columns=["_sleep_band"])


def make_model_views(frames):
    views = [frame[MODEL_FEATURES].copy() for frame in frames]
    for col in NUMERIC_COLS + GENERATOR_NUMERIC:
        for view in views:
            view[col] = view[col].astype(np.float32)
    for col in CATEGORICAL_COLS + GENERATOR_CATEGORICAL:
        string_views = [view[col].astype("string").fillna("<MISSING>") for view in views]
        categories = sorted(set().union(*(set(values.unique()) for values in string_views)))
        for view, values in zip(views, string_views):
            view[col] = pd.Categorical(values, categories=categories)
    return views


def exact_te_view(df):
    view = pd.DataFrame(index=df.index)
    for col in BASE_FEATURES:
        view[col] = df[col].astype("string").fillna("<MISSING>")
    return view


def compose(raw, encoded, names):
    return pd.concat([
        raw.reset_index(drop=True),
        pd.DataFrame(np.asarray(encoded, dtype=np.float32), columns=names),
    ], axis=1)


def metric_row(model, y_true, probability):
    prediction = probability.argmax(axis=1)
    recalls = recall_score(
        y_true, prediction, labels=np.arange(len(LABELS)), average=None
    )
    return {
        "model": model,
        "oof_balanced_accuracy": balanced_accuracy_score(y_true, prediction),
        "oof_accuracy": float((prediction == y_true).mean()),
        "oof_errors": int((prediction != y_true).sum()),
        **{f"recall_{label}": float(recalls[i]) for i, label in enumerate(LABELS)},
    }


start = time.time()
train_raw = pd.read_csv("train.csv")
test_raw = pd.read_csv("test.csv")
sample = pd.read_csv("sample_submission.csv")
source_raw = pd.read_csv("data/original/student_health_dataset_50k.csv")
assert set(BASE_FEATURES + [TARGET]).issubset(source_raw.columns)
assert source_raw[BASE_FEATURES].isna().sum().sum() == 0

keys, source_global_probability, source_tables = source_probability_tables(source_raw)
train = add_generator_features(train_raw, keys, source_global_probability, source_tables)
test = add_generator_features(test_raw, keys, source_global_probability, source_tables)
source = add_generator_features(source_raw, keys, source_global_probability, source_tables)
train_model, test_model, source_model = make_model_views([train, test, source])
train_te, test_te, source_te = exact_te_view(train), exact_te_view(test), exact_te_view(source)
te_names = [f"te_{column}_{klass}" for column in BASE_FEATURES for klass in range(len(LABELS))]

y = train[TARGET].map(LABEL_TO_INT).to_numpy(dtype=np.int8)
y_source = source[TARGET].map(LABEL_TO_INT).to_numpy(dtype=np.int8)
fold_map = pd.read_csv("experiment_003_artifacts/fold_assignments.csv")
assert fold_map[ID_COL].equals(train[ID_COL])
fold_id = fold_map["fold"].to_numpy(dtype=np.int8)

exp7 = np.load("experiment_007_artifacts/oof_predictions.npz")
assert np.array_equal(exp7["y"], y)
assert np.array_equal(exp7["fold"], fold_id)
base_oof = exp7["hgbc_te"].astype(np.float32)

variants = ["source_prior_features", "source_augmented"]
oof = {name: np.zeros((len(train), len(LABELS)), dtype=np.float32) for name in variants}
test_probability = {
    name: np.zeros((len(test), len(LABELS)), dtype=np.float64) for name in variants
}
fold_rows = []

for fold in range(N_SPLITS):
    fit_idx = np.flatnonzero(fold_id != fold)
    eval_idx = np.flatnonzero(fold_id == fold)
    y_fit, y_eval = y[fit_idx], y[eval_idx]

    for variant in variants:
        if variant == "source_prior_features":
            encoder_fit = train_te.iloc[fit_idx]
            model_fit = train_model.iloc[fit_idx]
            target_fit = y_fit
        else:
            encoder_fit = pd.concat(
                [train_te.iloc[fit_idx], source_te], ignore_index=True
            )
            model_fit = pd.concat(
                [train_model.iloc[fit_idx], source_model], ignore_index=True
            )
            target_fit = np.concatenate([y_fit, y_source])

        encoder = TargetEncoder(
            cv=5, smooth="auto", target_type="multiclass",
            shuffle=True, random_state=RANDOM_STATE + fold,
        )
        te_fit = encoder.fit_transform(encoder_fit, target_fit)
        te_eval = encoder.transform(train_te.iloc[eval_idx])
        te_test = encoder.transform(test_te)
        X_fit = compose(model_fit, te_fit, te_names)
        X_eval = compose(train_model.iloc[eval_idx], te_eval, te_names)
        X_test = compose(test_model, te_test, te_names)

        model = HistGradientBoostingClassifier(**HGBC_CONFIG)
        model.fit(X_fit, target_fit)
        eval_probability = model.predict_proba(X_eval).astype(np.float32)
        oof[variant][eval_idx] = eval_probability
        test_probability[variant] += model.predict_proba(X_test) / N_SPLITS
        score = balanced_accuracy_score(y_eval, eval_probability.argmax(axis=1))
        fold_rows.append({
            "fold": fold,
            "variant": variant,
            "rows": len(eval_idx),
            "training_rows": len(target_fit),
            "balanced_accuracy": float(score),
            "n_iter": int(model.n_iter_),
        })
        print(
            f"Fold {fold} {variant}: {score:.6f} n_iter={model.n_iter_}",
            flush=True,
        )
        del encoder_fit, model_fit, target_fit, encoder, te_fit, te_eval, te_test
        del X_fit, X_eval, X_test, model, eval_probability
        gc.collect()

fold_scores = pd.DataFrame(fold_rows)
fold_scores.to_csv(OUT / "fold_scores.csv", index=False)
summary = pd.DataFrame(
    [metric_row("experiment_007_hgbc_te", y, base_oof)]
    + [metric_row(name, y, oof[name]) for name in variants]
)
summary.to_csv(OUT / "summary.csv", index=False)

sleep_missing = train["sleep_duration"].isna().to_numpy()
slice_rows = []
for name, probability in [("experiment_007_hgbc_te", base_oof)] + [
    (variant, oof[variant]) for variant in variants
]:
    prediction = probability.argmax(axis=1)
    for segment, mask in [("sleep_missing", sleep_missing), ("sleep_present", ~sleep_missing)]:
        slice_rows.append({
            "model": name,
            "segment": segment,
            "rows": int(mask.sum()),
            "balanced_accuracy": float(balanced_accuracy_score(y[mask], prediction[mask])),
            "error_rate": float((prediction[mask] != y[mask]).mean()),
        })
pd.DataFrame(slice_rows).to_csv(OUT / "sleep_missing_error_slices.csv", index=False)

np.savez_compressed(
    OUT / "oof_predictions.npz",
    y=y, fold=fold_id, baseline=base_oof,
    source_prior_features=oof["source_prior_features"],
    source_augmented=oof["source_augmented"],
)
np.savez_compressed(
    OUT / "test_probabilities.npz",
    source_prior_features=test_probability["source_prior_features"].astype(np.float32),
    source_augmented=test_probability["source_augmented"].astype(np.float32),
)

for variant in variants:
    submission = sample.copy()
    submission[TARGET] = pd.Series(test_probability[variant].argmax(axis=1)).map(INT_TO_LABEL)
    assert submission[ID_COL].equals(test[ID_COL].reset_index(drop=True))
    assert submission[TARGET].isin(LABELS).all()
    submission.to_csv(f"submission_experiment_008_{variant}.csv", index=False)

source_rule_prediction = source["source_rule_label"].replace("<MISSING>", "at-risk")
source_rule_ba = balanced_accuracy_score(source[TARGET], source_rule_prediction)
metadata = {
    "source_dataset": "ziya07/college-student-health-behavior-dataset",
    "source_file": "student_health_dataset_50k.csv",
    "source_license": "CC0-1.0",
    "source_rows": len(source),
    "source_rule_balanced_accuracy_on_source": source_rule_ba,
    "source_global_class_probability": dict(zip(LABELS, source_global_probability.tolist())),
    "generator_keys": keys,
    "probability_smoothing_alpha": 20.0,
    "fold_source": "experiment_003_artifacts/fold_assignments.csv",
    "hgbc_config": HGBC_CONFIG,
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
print("\nSLICES")
print(pd.DataFrame(slice_rows).to_string(index=False))
print("\nMETADATA")
print(json.dumps(metadata, indent=2))
