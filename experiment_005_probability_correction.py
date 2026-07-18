from pathlib import Path
import json
import time

import numpy as np
import pandas as pd


OUT = Path("experiment_005_artifacts")
OUT.mkdir(exist_ok=True)
LABELS = ["unhealthy", "at-risk", "fit"]
N_CLASSES = len(LABELS)


def balanced_accuracy_fast(y, pred):
    recalls = []
    for cls in range(N_CLASSES):
        mask = y == cls
        recalls.append((pred[mask] == cls).mean())
    return float(np.mean(recalls))


def corrected_predictions(prob, multipliers):
    return (prob * np.asarray(multipliers)).argmax(axis=1)


def optimize_coordinate(prob, y, start=(1.0, 1.0, 1.0)):
    # at-risk is the reference class (multiplier fixed to 1). CatBoost's class
    # weighting yields lower at-risk recall than minority recall, so search a
    # conservative range that can downweight either extreme class.
    multipliers = np.array(start, dtype=float)
    grid = np.arange(0.75, 1.1001, 0.0025)
    history = []
    for pass_id in range(3):
        changed = False
        for cls in [0, 2]:
            results = []
            for value in grid:
                trial = multipliers.copy()
                trial[cls] = value
                score = balanced_accuracy_fast(y, corrected_predictions(prob, trial))
                results.append((score, value))
            best_score, best_value = max(results)
            if abs(best_value - multipliers[cls]) > 1e-12:
                changed = True
            multipliers[cls] = best_value
            history.append({
                "pass": pass_id,
                "class": LABELS[cls],
                "multiplier": best_value,
                "train_score": best_score,
            })
        if not changed:
            break
    return multipliers, history


data = np.load("experiment_004_artifacts/oof_predictions.npz")
y = data["y"].astype(np.int8)
fold = data["fold"].astype(np.int8)
prob = data["blend"].astype(np.float32)

start_time = time.time()
crossfit_pred = np.empty(len(y), dtype=np.int8)
fold_rows = []
multiplier_rows = []

for heldout_fold in np.unique(fold):
    fit_mask = fold != heldout_fold
    eval_mask = fold == heldout_fold
    multipliers, history = optimize_coordinate(prob[fit_mask], y[fit_mask])
    base_pred = prob[eval_mask].argmax(axis=1)
    corrected_pred = corrected_predictions(prob[eval_mask], multipliers)
    crossfit_pred[eval_mask] = corrected_pred
    base_score = balanced_accuracy_fast(y[eval_mask], base_pred)
    corrected_score = balanced_accuracy_fast(y[eval_mask], corrected_pred)
    fold_rows.append({
        "fold": int(heldout_fold),
        "rows": int(eval_mask.sum()),
        "base_balanced_accuracy": base_score,
        "corrected_balanced_accuracy": corrected_score,
        "delta": corrected_score - base_score,
        "unhealthy_multiplier": multipliers[0],
        "at_risk_multiplier": multipliers[1],
        "fit_multiplier": multipliers[2],
    })
    for row in history:
        multiplier_rows.append({"heldout_fold": int(heldout_fold), **row})
    print(f"Fold {heldout_fold}: {base_score:.6f} -> {corrected_score:.6f} "
          f"({corrected_score-base_score:+.6f}) multipliers={multipliers}", flush=True)

fold_results = pd.DataFrame(fold_rows)
fold_results.to_csv(OUT / "crossfit_fold_results.csv", index=False)
pd.DataFrame(multiplier_rows).to_csv(OUT / "optimization_history.csv", index=False)

base_oof = balanced_accuracy_fast(y, prob.argmax(axis=1))
crossfit_oof = balanced_accuracy_fast(y, crossfit_pred)
full_multipliers, full_history = optimize_coordinate(prob, y)
full_corrected = corrected_predictions(prob, full_multipliers)
full_optimized_oof = balanced_accuracy_fast(y, full_corrected)

summary = {
    "base_oof_balanced_accuracy": base_oof,
    "crossfit_corrected_oof_balanced_accuracy": crossfit_oof,
    "crossfit_delta": crossfit_oof - base_oof,
    "crossfit_fold_delta_mean": float(fold_results["delta"].mean()),
    "crossfit_fold_delta_std": float(fold_results["delta"].std(ddof=1)),
    "folds_improved": int((fold_results["delta"] > 0).sum()),
    "full_oof_optimized_balanced_accuracy": full_optimized_oof,
    "full_oof_optimistic_delta": full_optimized_oof - base_oof,
    "deployment_multipliers": dict(zip(LABELS, full_multipliers.tolist())),
    "elapsed_seconds": time.time() - start_time,
}

test_prob_path = Path("experiment_004_artifacts/test_probabilities.npz")
if test_prob_path.exists():
    test_prob = np.load(test_prob_path)["blend"].astype(np.float32)
    base_test_pred = test_prob.argmax(axis=1)
    corrected_test_pred = corrected_predictions(test_prob, full_multipliers)
    sample = pd.read_csv("sample_submission.csv")
    label_map = {0: "unhealthy", 1: "at-risk", 2: "fit"}
    submission = sample.copy()
    submission["health_condition"] = pd.Series(corrected_test_pred).map(label_map)
    assert submission["health_condition"].notna().all()
    submission.to_csv("submission_probability_corrected_ensemble.csv", index=False)
    summary["test_labels_changed"] = int((base_test_pred != corrected_test_pred).sum())
    summary["corrected_prediction_counts"] = submission["health_condition"].value_counts().to_dict()

(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
pd.DataFrame(full_history).to_csv(OUT / "full_optimization_history.csv", index=False)
np.savez_compressed(
    OUT / "crossfit_predictions.npz",
    y=y,
    fold=fold,
    base_prob=prob,
    crossfit_pred=crossfit_pred,
    full_corrected_pred=full_corrected,
    deployment_multipliers=full_multipliers,
)
print(json.dumps(summary, indent=2))
