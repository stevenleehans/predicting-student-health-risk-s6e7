"""Experiment 011: honest Kaggle-GPU test of the three ranked remaining levers.

1. Exact-generator disagreement resolver.
2. TabPFN-3 orthogonal OOF probabilities.
3. FT-Transformer-v2 plus a restricted, nested OOF stack.

The script intentionally does not inspect or optimize against the Kaggle public
leaderboard.  All model and stack choices are evaluated on the fixed five folds
used by Experiments 003-010.
"""

from __future__ import annotations

import gc
import json
import os
import platform
import random
import subprocess
import sys
import time
import traceback
import urllib.request
from pathlib import Path
from glob import glob


def install_runtime():
    packages = ["tabpfn", "masamlp==0.3.0", "catstat==0.4.0"]
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", *packages],
        check=True,
    )


IS_KAGGLE = Path("/kaggle/working").exists()
if IS_KAGGLE:
    install_runtime()

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score


SEED = 42
N_FOLDS = 5
TARGET = "health_condition"
ID_COL = "id"
LABELS = ["unhealthy", "at-risk", "fit"]
LABEL_TO_INT = {label: i for i, label in enumerate(LABELS)}
BASE_FEATURES = [
    "sleep_duration", "heart_rate", "bmi", "calorie_expenditure",
    "step_count", "exercise_duration", "water_intake", "diet_type",
    "stress_level", "sleep_quality", "physical_activity_level",
    "smoking_alcohol", "gender",
]
CAT_COLS = [
    "diet_type", "stress_level", "sleep_quality",
    "physical_activity_level", "smoking_alcohol", "gender",
]
OUT = (
    Path("/kaggle/working/experiment_011_artifacts")
    if IS_KAGGLE else Path("experiment_011_artifacts")
)
OUT.mkdir(parents=True, exist_ok=True)
GITHUB_RAW = (
    "https://raw.githubusercontent.com/stevenleehans/"
    "predicting-student-health-risk-s6e7/main/experiment_010_artifacts"
)


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metric_row(name, y_true, probability=None, prediction=None):
    if prediction is None:
        prediction = np.asarray(probability).argmax(axis=1)
    recalls = recall_score(
        y_true, prediction, labels=np.arange(len(LABELS)), average=None
    )
    return {
        "candidate": name,
        "oof_balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "accuracy": float(np.mean(prediction == y_true)),
        "errors": int(np.sum(prediction != y_true)),
        **{f"recall_{label}": float(recalls[i]) for i, label in enumerate(LABELS)},
    }


def fold_rows(name, y, fold_id, probability=None, prediction=None):
    rows = []
    for fold in range(N_FOLDS):
        mask = fold_id == fold
        pred = prediction[mask] if prediction is not None else probability[mask].argmax(1)
        rows.append({
            "candidate": name,
            "fold": fold,
            "balanced_accuracy": float(balanced_accuracy_score(y[mask], pred)),
            "accuracy": float(np.mean(y[mask] == pred)),
            "errors": int(np.sum(y[mask] != pred)),
        })
    return rows


def save_state(name, **arrays):
    np.savez_compressed(OUT / name, **arrays)


def exact_rule(frame):
    complete = frame[[
        "sleep_duration", "stress_level", "physical_activity_level"
    ]].notna().all(axis=1).to_numpy()
    sleep = frame["sleep_duration"].to_numpy()
    stress_high = frame["stress_level"].eq("high").fillna(False).to_numpy(bool)
    stress_low = frame["stress_level"].eq("low").fillna(False).to_numpy(bool)
    activity_active = (
        frame["physical_activity_level"].eq("active").fillna(False).to_numpy(bool)
    )
    pred = np.full(len(frame), LABEL_TO_INT["at-risk"], dtype=np.int8)
    pred[(sleep < 6) & stress_high & complete] = LABEL_TO_INT["unhealthy"]
    pred[(sleep >= 7) & stress_low & activity_active & complete] = LABEL_TO_INT["fit"]
    return pred, complete


def gate_features(frame, hgbc, realmlp, blend, rule_pred):
    data = frame[BASE_FEATURES].copy()
    for col in CAT_COLS:
        data[col] = data[col].astype("string").fillna("<MISSING>").astype(str)
    for model_name, probability in [
        ("hgbc", hgbc), ("realmlp", realmlp), ("blend", blend)
    ]:
        clipped = np.clip(probability, 1e-7, 1.0)
        for k, label in enumerate(LABELS):
            data[f"{model_name}_p_{label}"] = clipped[:, k]
        ordered = np.sort(clipped, axis=1)
        data[f"{model_name}_confidence"] = ordered[:, -1]
        data[f"{model_name}_margin"] = ordered[:, -1] - ordered[:, -2]
        data[f"{model_name}_entropy"] = -(clipped * np.log(clipped)).sum(axis=1)
        data[f"{model_name}_prediction"] = probability.argmax(1).astype(str)
    data["rule_prediction"] = rule_pred.astype(str)
    data["sleep_distance_6"] = np.abs(data["sleep_duration"] - 6.0)
    data["sleep_distance_7"] = np.abs(data["sleep_duration"] - 7.0)
    return data


def run_exact_rule_resolver(train, test, y, fold_id, hgbc_oof, real_oof,
                            blend_oof, hgbc_test, real_test, blend_test):
    train_rule, complete = exact_rule(train)
    test_rule, test_complete = exact_rule(test)
    blend_pred = blend_oof.argmax(1)
    test_blend_pred = blend_test.argmax(1)
    disagreement = complete & (train_rule != blend_pred)
    test_disagreement = test_complete & (test_rule != test_blend_pred)
    one_wins = disagreement & ((train_rule == y) ^ (blend_pred == y))
    gate_target = (train_rule == y).astype(np.int8)
    x_gate = gate_features(train, hgbc_oof, real_oof, blend_oof, train_rule)
    x_gate_test = gate_features(test, hgbc_test, real_test, blend_test, test_rule)
    cat_indices = [i for i, c in enumerate(x_gate.columns) if x_gate[c].dtype == object]
    resolved = blend_pred.copy()
    test_vote_probability = np.zeros(len(test), dtype=np.float32)
    fold_detail = []
    for fold in range(N_FOLDS):
        fit_mask = (fold_id != fold) & one_wins
        valid_mask = (fold_id == fold) & disagreement
        model = CatBoostClassifier(
            iterations=500,
            depth=5,
            learning_rate=0.035,
            loss_function="Logloss",
            random_seed=SEED + fold,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        )
        model.fit(x_gate.loc[fit_mask], gate_target[fit_mask], cat_features=cat_indices)
        if valid_mask.any():
            use_rule = model.predict_proba(x_gate.loc[valid_mask])[:, 1] >= 0.5
            indices = np.flatnonzero(valid_mask)
            resolved[indices[use_rule]] = train_rule[indices[use_rule]]
        if test_disagreement.any():
            test_vote_probability[test_disagreement] += (
                model.predict_proba(x_gate_test.loc[test_disagreement])[:, 1] / N_FOLDS
            )
        fold_detail.append({
            "candidate": "exact_rule_resolver", "fold": fold,
            "fit_disagreements": int(fit_mask.sum()),
            "valid_disagreements": int(valid_mask.sum()),
        })
        del model
        gc.collect()
    resolved_test = test_blend_pred.copy()
    choose_rule = test_disagreement & (test_vote_probability >= 0.5)
    resolved_test[choose_rule] = test_rule[choose_rule]
    diagnostics = {
        "complete_rows": int(complete.sum()),
        "rule_blend_disagreements": int(disagreement.sum()),
        "rule_only_correct": int(np.sum(disagreement & (train_rule == y) & (blend_pred != y))),
        "blend_only_correct": int(np.sum(disagreement & (blend_pred == y) & (train_rule != y))),
        "both_wrong": int(np.sum(disagreement & (blend_pred != y) & (train_rule != y))),
        "test_disagreements": int(test_disagreement.sum()),
        "test_choose_rule": int(choose_rule.sum()),
    }
    return resolved, resolved_test, fold_detail, diagnostics


def reorder_probabilities(probability, model_classes):
    normalized = [str(x) for x in model_classes]
    if all(value in LABELS for value in normalized):
        order = [normalized.index(label) for label in LABELS]
    else:
        ints = [int(x) for x in model_classes]
        order = [ints.index(i) for i in range(len(LABELS))]
    return np.asarray(probability)[:, order]


def run_tabpfn(train, test, y, fold_id):
    from tabpfn import TabPFNClassifier

    model_cache = Path("/kaggle/input/models/prior-labsai/tabpfn-3/pytorch/default/1")
    if model_cache.exists():
        os.environ["TABPFN_MODEL_CACHE_DIR"] = str(model_cache)
    gpu_count = torch.cuda.device_count()
    device = [f"cuda:{i}" for i in range(gpu_count)] if gpu_count > 1 else "cuda"
    x = train[BASE_FEATURES].copy()
    x_test = test[BASE_FEATURES].copy()
    oof = np.zeros((len(train), len(LABELS)), dtype=np.float32)
    test_probability = np.zeros((len(test), len(LABELS)), dtype=np.float32)
    rows = []
    for fold in range(N_FOLDS):
        started = time.time()
        print(f"[TabPFN-3] fold {fold + 1}/{N_FOLDS} starting", flush=True)
        fit = fold_id != fold
        valid = ~fit
        model = TabPFNClassifier(
            device=device,
            n_estimators=2,
            balance_probabilities=True,
            fit_mode="fit_with_cache",
            random_state=SEED + fold,
        )
        model.fit(x.loc[fit], y[fit])
        valid_chunks = []
        for start in range(0, int(valid.sum()), 50000):
            part = x.loc[valid].iloc[start:start + 50000]
            valid_chunks.append(model.predict_proba(part))
        fold_probability = reorder_probabilities(np.vstack(valid_chunks), model.classes_)
        oof[valid] = fold_probability
        # CV is the goal of this run. Test inference is deterministic but slow,
        # so run it once with the final fold model instead of repeating the same
        # 295k-row pass five times. This does not change any OOF score.
        test_chunks = None
        if fold == N_FOLDS - 1:
            test_chunks = []
            for start in range(0, len(x_test), 50000):
                test_chunks.append(model.predict_proba(x_test.iloc[start:start + 50000]))
            test_probability = reorder_probabilities(np.vstack(test_chunks), model.classes_)
        fold_score = float(balanced_accuracy_score(y[valid], oof[valid].argmax(1)))
        rows.append({
            "candidate": "tabpfn3", "fold": fold,
            "balanced_accuracy": fold_score,
            "runtime_minutes": (time.time() - started) / 60,
        })
        save_state("tabpfn_checkpoint.npz", oof=oof, test=test_probability, completed_fold=fold)
        print(
            f"[TabPFN-3] fold {fold + 1}/{N_FOLDS} BA={fold_score:.6f} "
            f"runtime={(time.time() - started) / 60:.1f}m",
            flush=True,
        )
        del model, valid_chunks, test_chunks
        gc.collect()
        torch.cuda.empty_cache()
    return oof, test_probability, rows


def crossfit_prior_correction(probability, y, fold_id):
    prior = np.bincount(y, minlength=len(LABELS)).astype(float)
    prior /= prior.sum()
    grid = np.arange(0.0, 2.501, 0.05)

    def corrected(p, beta, local_prior):
        adjusted = p / np.power(local_prior[None, :], beta)
        return adjusted / adjusted.sum(axis=1, keepdims=True)

    output = np.zeros_like(probability, dtype=np.float32)
    selected = []
    for fold in range(N_FOLDS):
        meta = fold_id != fold
        valid = ~meta
        meta_prior = np.bincount(y[meta], minlength=len(LABELS)).astype(float)
        meta_prior /= meta_prior.sum()
        scores = [
            balanced_accuracy_score(y[meta], corrected(probability[meta], beta, meta_prior).argmax(1))
            for beta in grid
        ]
        beta = float(grid[int(np.argmax(scores))])
        output[valid] = corrected(probability[valid], beta, meta_prior)
        selected.append(beta)
    full_scores = [
        balanced_accuracy_score(y, corrected(probability, beta, prior).argmax(1))
        for beta in grid
    ]
    full_beta = float(grid[int(np.argmax(full_scores))])
    return output, selected, full_beta, prior


def exact_target_encoding(train_x, valid_x, test_x, y_labels):
    from catstat import TargetEncoder

    classes = np.sort(np.unique(y_labels))
    codes = pd.Series(y_labels).map({c: i for i, c in enumerate(classes)}).to_numpy()
    encoder = TargetEncoder(
        random_state=SEED,
        cols=BASE_FEATURES,
        stats=("mean",),
        target_type="multiclass",
        smooth="auto",
        numeric="direct",
    )
    train_encoded = encoder.fit_transform(train_x[BASE_FEATURES], codes)
    valid_encoded = encoder.transform(valid_x[BASE_FEATURES])
    test_encoded = encoder.transform(test_x[BASE_FEATURES])
    return (
        np.asarray(train_encoded, dtype=np.float32),
        np.asarray(valid_encoded, dtype=np.float32),
        np.asarray(test_encoded, dtype=np.float32),
    )


def run_ft_transformer(train, test, y, fold_id):
    from masamlp import MasaClassifier

    x = train[BASE_FEATURES].copy()
    x_test = test[BASE_FEATURES].copy()
    y_labels = np.asarray(LABELS, dtype=object)[y]
    raw_oof = np.zeros((len(train), len(LABELS)), dtype=np.float32)
    raw_test = np.zeros((len(test), len(LABELS)), dtype=np.float32)
    rows = []
    for fold in range(N_FOLDS):
        started = time.time()
        fit = fold_id != fold
        valid = ~fit
        train_te, valid_te, test_te = exact_target_encoding(
            x.loc[fit], x.loc[valid], x_test, y_labels[fit]
        )
        te_names = [f"te_{c}_{k}" for c in BASE_FEATURES for k in range(len(LABELS))]
        x_fit = pd.concat([
            x.loc[fit].reset_index(drop=True),
            pd.DataFrame(train_te, columns=te_names),
        ], axis=1)
        x_valid = pd.concat([
            x.loc[valid].reset_index(drop=True),
            pd.DataFrame(valid_te, columns=te_names),
        ], axis=1)
        x_deploy = pd.concat([
            x_test.reset_index(drop=True),
            pd.DataFrame(test_te, columns=te_names),
        ], axis=1)
        model = MasaClassifier(
            model="ft_transformer",
            n_epochs=16,
            batch_size=4096,
            learning_rate=0.001,
            weight_decay=1e-5,
            optimizer="adamw",
            lr_scheduler="cosine",
            num_embedding="plr-lite",
            numeric_scaler="quantile",
            cat_encoding="embedding",
            class_weight=None,
            label_smoothing=0,
            early_stopping_rounds=None,
            eval_metric="multi_logloss",
            n_ens=4,
            device="auto",
            amp="auto",
            ens_mode="loop",
            eval_batch_size=2048,
            model_params={
                "d_block": 128, "n_blocks": 2, "attention_n_heads": 8,
                "n_frequencies": 24, "sigma": 0.1,
            },
            categorical_features=CAT_COLS,
            random_state=SEED + fold,
            verbose=1,
        )
        model.fit(x_fit, y_labels[fit])
        fold_probability = reorder_probabilities(model.predict_proba(x_valid), model.classes_)
        raw_oof[valid] = fold_probability
        raw_test += reorder_probabilities(model.predict_proba(x_deploy), model.classes_) / N_FOLDS
        rows.append({
            "candidate": "ft_transformer_v2_raw", "fold": fold,
            "balanced_accuracy": float(balanced_accuracy_score(y[valid], raw_oof[valid].argmax(1))),
            "runtime_minutes": (time.time() - started) / 60,
        })
        save_state("ftt_checkpoint.npz", oof=raw_oof, test=raw_test, completed_fold=fold)
        del model, train_te, valid_te, test_te, x_fit, x_valid, x_deploy
        gc.collect()
        torch.cuda.empty_cache()
    corrected_oof, betas, full_beta, prior = crossfit_prior_correction(raw_oof, y, fold_id)
    corrected_test = raw_test / np.power(prior[None, :], full_beta)
    corrected_test /= corrected_test.sum(axis=1, keepdims=True)
    for row, beta in zip(rows, betas):
        row["crossfit_beta"] = beta
    return raw_oof, raw_test, corrected_oof, corrected_test, rows, full_beta


def restricted_stack(model_oof, model_test, y, fold_id):
    names = list(model_oof)
    count = len(names)
    candidates = [np.eye(count)[i] for i in range(count)]
    candidates.append(np.ones(count) / count)
    if "hgbc" in names and "realmlp" in names:
        vector = np.zeros(count)
        vector[names.index("hgbc")] = 0.44
        vector[names.index("realmlp")] = 0.56
        candidates.append(vector)
    # A deterministic pairwise grid keeps the nested stack honest and cheap.
    # The former 5,000-vector random simplex search added billions of full-OOF
    # operations for negligible extra resolution.
    for left in range(count):
        for right in range(left + 1, count):
            for alpha in np.arange(0.0, 1.001, 0.05):
                vector = np.zeros(count)
                vector[left] = alpha
                vector[right] = 1.0 - alpha
                candidates.append(vector)
    candidates = np.unique(np.asarray(candidates, dtype=np.float32), axis=0)

    def probabilities(weights, source):
        return sum(weights[i] * source[name] for i, name in enumerate(names))

    weighted_oof = np.zeros_like(next(iter(model_oof.values())))
    weight_rows = []
    for fold in range(N_FOLDS):
        meta = fold_id != fold
        valid = ~meta
        best_score, best_weight = -1.0, None
        for weight in candidates:
            pred = probabilities(weight, {n: model_oof[n][meta] for n in names}).argmax(1)
            score = balanced_accuracy_score(y[meta], pred)
            if score > best_score:
                best_score, best_weight = score, weight.copy()
        weighted_oof[valid] = probabilities(best_weight, {n: model_oof[n][valid] for n in names})
        weight_rows.append({
            "candidate": "restricted_nonnegative_stack", "fold": fold,
            "meta_balanced_accuracy": float(best_score),
            **{f"weight_{n}": float(best_weight[i]) for i, n in enumerate(names)},
        })
    best_score, full_weight = -1.0, None
    for weight in candidates:
        score = balanced_accuracy_score(y, probabilities(weight, model_oof).argmax(1))
        if score > best_score:
            best_score, full_weight = score, weight.copy()
    weighted_test = probabilities(full_weight, model_test)

    logit_oof = np.concatenate([
        np.log(np.clip(model_oof[n], 1e-6, 1.0)) for n in names
    ], axis=1)
    logit_test = np.concatenate([
        np.log(np.clip(model_test[n], 1e-6, 1.0)) for n in names
    ], axis=1)
    logistic_oof = np.zeros_like(weighted_oof)
    logistic_test = np.zeros_like(weighted_test)
    for fold in range(N_FOLDS):
        fit = fold_id != fold
        valid = ~fit
        meta = LogisticRegression(
            C=0.01, class_weight="balanced", max_iter=1000,
            solver="lbfgs", random_state=SEED + fold,
        )
        meta.fit(logit_oof[fit], y[fit])
        logistic_oof[valid] = reorder_probabilities(meta.predict_proba(logit_oof[valid]), meta.classes_)
        logistic_test += reorder_probabilities(meta.predict_proba(logit_test), meta.classes_) / N_FOLDS
    return weighted_oof, weighted_test, logistic_oof, logistic_test, weight_rows, names, full_weight


def main():
    seed_everything()
    started = time.time()
    input_dir = Path("/kaggle/input/playground-series-s6e7")
    if not (input_dir / "train.csv").exists():
        candidates = glob("/kaggle/input/**/train.csv", recursive=True)
        if not candidates:
            raise FileNotFoundError("Competition train.csv was not mounted")
        input_dir = Path(candidates[0]).parent
    train = pd.read_csv(input_dir / "train.csv")
    test = pd.read_csv(input_dir / "test.csv")
    sample = pd.read_csv(input_dir / "sample_submission.csv")
    y = train[TARGET].map(LABEL_TO_INT).to_numpy(dtype=np.int8)

    for filename in ["oof_predictions.npz", "test_probabilities.npz"]:
        destination = OUT / f"experiment010_{filename}"
        urllib.request.urlretrieve(f"{GITHUB_RAW}/{filename}", destination)
    base = np.load(OUT / "experiment010_oof_predictions.npz")
    base_test = np.load(OUT / "experiment010_test_probabilities.npz")
    assert np.array_equal(base["y"], y)
    fold_id = base["fold"].astype(np.int8)
    hgbc_oof = base["hgbc"].astype(np.float32)
    real_oof = base["realmlp"].astype(np.float32)
    blend_oof = base["crossfit_blend"].astype(np.float32)
    hgbc_test = base_test["hgbc"].astype(np.float32)
    real_test = base_test["realmlp"].astype(np.float32)
    blend_test = base_test["deployment_blend"].astype(np.float32)

    summary = [metric_row("experiment_010_crossfit_blend", y, blend_oof)]
    fold_scores = fold_rows("experiment_010_crossfit_blend", y, fold_id, blend_oof)
    errors = {}

    resolved, resolved_test, detail, rule_diagnostics = run_exact_rule_resolver(
        train, test, y, fold_id, hgbc_oof, real_oof, blend_oof,
        hgbc_test, real_test, blend_test,
    )
    summary.append(metric_row("exact_rule_resolver", y, prediction=resolved))
    fold_scores.extend(fold_rows("exact_rule_resolver", y, fold_id, prediction=resolved))
    fold_scores.extend(detail)
    pd.DataFrame(summary).to_csv(OUT / "rule_resolver_summary.csv", index=False)
    save_state(
        "rule_resolver_checkpoint.npz",
        y=y, fold=fold_id, prediction=resolved, test_prediction=resolved_test,
    )
    (OUT / "rule_resolver_diagnostics.json").write_text(
        json.dumps(rule_diagnostics, indent=2)
    )

    model_oof = {"hgbc": hgbc_oof, "realmlp": real_oof}
    model_test = {"hgbc": hgbc_test, "realmlp": real_test}

    try:
        tab_oof, tab_test, tab_rows = run_tabpfn(train, test, y, fold_id)
        model_oof["tabpfn3"] = tab_oof
        model_test["tabpfn3"] = tab_test
        summary.append(metric_row("tabpfn3", y, tab_oof))
        fold_scores.extend(tab_rows)
    except Exception as exc:
        errors["tabpfn3"] = {"error": repr(exc), "traceback": traceback.format_exc()}
        print(errors["tabpfn3"]["traceback"], flush=True)
        gc.collect()
        torch.cuda.empty_cache()

    ftt_raw_oof = ftt_raw_test = None
    try:
        ftt_raw_oof, ftt_raw_test, ftt_oof, ftt_test, ftt_rows, full_beta = run_ft_transformer(
            train, test, y, fold_id
        )
        model_oof["ft_transformer_v2"] = ftt_oof
        model_test["ft_transformer_v2"] = ftt_test
        summary.append(metric_row("ft_transformer_v2_raw", y, ftt_raw_oof))
        summary.append(metric_row("ft_transformer_v2_crossfit_prior", y, ftt_oof))
        fold_scores.extend(ftt_rows)
    except Exception as exc:
        full_beta = None
        errors["ft_transformer_v2"] = {"error": repr(exc), "traceback": traceback.format_exc()}
        print(errors["ft_transformer_v2"]["traceback"], flush=True)
        gc.collect()
        torch.cuda.empty_cache()

    weighted_oof, weighted_test, logistic_oof, logistic_test, weight_rows, stack_names, full_weights = restricted_stack(
        model_oof, model_test, y, fold_id
    )
    summary.append(metric_row("restricted_nonnegative_stack", y, weighted_oof))
    summary.append(metric_row("restricted_logistic_stack", y, logistic_oof))
    fold_scores.extend(fold_rows("restricted_nonnegative_stack", y, fold_id, weighted_oof))
    fold_scores.extend(fold_rows("restricted_logistic_stack", y, fold_id, logistic_oof))
    fold_scores.extend(weight_rows)

    summary_frame = pd.DataFrame(summary).sort_values("oof_balanced_accuracy", ascending=False)
    summary_frame.to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(fold_scores).to_csv(OUT / "fold_scores.csv", index=False)
    save_state(
        "oof_predictions.npz", y=y, fold=fold_id, hgbc=hgbc_oof,
        realmlp=real_oof, experiment010_blend=blend_oof,
        exact_rule_resolver=resolved, restricted_weighted=weighted_oof,
        restricted_logistic=logistic_oof,
        **({"tabpfn3": model_oof["tabpfn3"]} if "tabpfn3" in model_oof else {}),
        **({"ft_transformer_v2": model_oof["ft_transformer_v2"]} if "ft_transformer_v2" in model_oof else {}),
    )
    save_state(
        "test_probabilities.npz", hgbc=hgbc_test, realmlp=real_test,
        experiment010_blend=blend_test, exact_rule_resolver=resolved_test,
        restricted_weighted=weighted_test, restricted_logistic=logistic_test,
        **({"tabpfn3": model_test["tabpfn3"]} if "tabpfn3" in model_test else {}),
        **({"ft_transformer_v2": model_test["ft_transformer_v2"]} if "ft_transformer_v2" in model_test else {}),
    )
    best_name = summary_frame.iloc[0]["candidate"]
    test_candidates = {
        "experiment_010_crossfit_blend": blend_test,
        "exact_rule_resolver": resolved_test,
        "restricted_nonnegative_stack": weighted_test,
        "restricted_logistic_stack": logistic_test,
        **{name: probability for name, probability in model_test.items()},
    }
    if ftt_raw_test is not None:
        test_candidates["ft_transformer_v2_raw"] = ftt_raw_test
        test_candidates["ft_transformer_v2_crossfit_prior"] = model_test["ft_transformer_v2"]
    best_test = test_candidates[best_name]
    best_prediction = best_test if best_test.ndim == 1 else best_test.argmax(1)
    submission = sample.copy()
    submission[TARGET] = np.asarray(LABELS, dtype=object)[best_prediction.astype(int)]
    submission.to_csv("/kaggle/working/submission_experiment_011_best_cv.csv", index=False)
    metadata = {
        "experiment": 11,
        "fixed_folds": "Experiment 003 / StratifiedKFold(5, shuffle=True, random_state=42)",
        "labels": LABELS,
        "best_candidate": best_name,
        "best_oof_balanced_accuracy": float(summary_frame.iloc[0]["oof_balanced_accuracy"]),
        "baseline_oof_balanced_accuracy": float(summary[0]["oof_balanced_accuracy"]),
        "rule_diagnostics": rule_diagnostics,
        "tabpfn_test_probability": "single final-fold model; five-fold OOF is unchanged",
        "stack_models": stack_names,
        "full_stack_weights": {name: float(full_weights[i]) for i, name in enumerate(stack_names)},
        "ft_transformer_full_beta": full_beta,
        "errors": errors,
        "runtime_hours": (time.time() - started) / 3600,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (OUT / "errors.json").write_text(json.dumps(errors, indent=2))
    print("\nFINAL HONEST CV SUMMARY", flush=True)
    print(summary_frame.to_string(index=False), flush=True)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
