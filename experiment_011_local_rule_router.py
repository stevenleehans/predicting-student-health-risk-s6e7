"""Local fold-safe validation of exact-rule/blend disagreement routers."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import balanced_accuracy_score, recall_score

import kaggle_experiment_011_options_123_gpu as helpers


OUT = Path("experiment_011_local_artifacts")
OUT.mkdir(exist_ok=True)
LABELS = helpers.LABELS


def metric_row(name, y, prediction):
    recall = recall_score(y, prediction, labels=np.arange(3), average=None)
    return {
        "candidate": name,
        "oof_balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "accuracy": float(np.mean(y == prediction)),
        "errors": int(np.sum(y != prediction)),
        **{f"recall_{label}": float(recall[i]) for i, label in enumerate(LABELS)},
    }


train = pd.read_csv("train.csv")
base = np.load("experiment_010_artifacts/oof_predictions.npz")
y = base["y"]
fold_id = base["fold"]
blend = base["crossfit_blend"].astype(np.float32)
blend_pred = blend.argmax(1)
rule_pred, complete = helpers.exact_rule(train)
disagreement = complete & (rule_pred != blend_pred)
one_wins = disagreement & ((rule_pred == y) ^ (blend_pred == y))
gate_target = (rule_pred == y).astype(np.int8)
features = helpers.gate_features(
    train, base["hgbc"], base["realmlp"], blend, rule_pred
)
cat_indices = [
    i for i, col in enumerate(features.columns) if features[col].dtype == object
]

gate_probability = np.full(len(train), np.nan, dtype=np.float32)
for fold in range(5):
    fit = (fold_id != fold) & one_wins
    valid = (fold_id == fold) & disagreement
    model = CatBoostClassifier(
        iterations=500, depth=5, learning_rate=0.035,
        loss_function="Logloss", random_seed=42 + fold,
        verbose=False, allow_writing_files=False, thread_count=-1,
    )
    model.fit(features.loc[fit], gate_target[fit], cat_features=cat_indices)
    gate_probability[valid] = model.predict_proba(features.loc[valid])[:, 1]


def crossfit_threshold(score, grid, label):
    output = blend_pred.copy()
    rows = []
    for fold in range(5):
        meta = fold_id != fold
        valid = ~meta
        candidates = []
        for threshold in grid:
            prediction = blend_pred[meta].copy()
            use = disagreement[meta] & (score[meta] >= threshold)
            prediction[use] = rule_pred[meta][use]
            candidates.append((
                balanced_accuracy_score(y[meta], prediction), float(threshold)
            ))
        meta_score, threshold = max(candidates, key=lambda item: (item[0], item[1]))
        use = valid & disagreement & (score >= threshold)
        output[use] = rule_pred[use]
        rows.append({
            "router": label, "fold": fold, "threshold": threshold,
            "meta_balanced_accuracy": float(meta_score),
            "rules_used": int(use.sum()),
            "heldout_balanced_accuracy": float(
                balanced_accuracy_score(y[valid], output[valid])
            ),
        })
    return output, rows


gate_router, gate_rows = crossfit_threshold(
    gate_probability, np.arange(0.0, 1.001, 0.01), "catboost_gate"
)
rule_gap = blend[np.arange(len(y)), rule_pred] - blend.max(1)
gap_router, gap_rows = crossfit_threshold(
    rule_gap, np.arange(-0.5, 0.001, 0.005), "blend_rule_probability_gap"
)
threshold_half = blend_pred.copy()
use_half = disagreement & (gate_probability >= 0.5)
threshold_half[use_half] = rule_pred[use_half]

summary = pd.DataFrame([
    metric_row("experiment_010_trusted_blend", y, blend_pred),
    metric_row("catboost_gate_threshold_0.5", y, threshold_half),
    metric_row("catboost_gate_crossfit_threshold", y, gate_router),
    metric_row("probability_gap_crossfit_threshold", y, gap_router),
]).sort_values("oof_balanced_accuracy", ascending=False)
summary.to_csv(OUT / "router_summary.csv", index=False)
pd.DataFrame(gate_rows + gap_rows).to_csv(OUT / "router_thresholds.csv", index=False)
np.savez_compressed(
    OUT / "router_oof.npz", y=y, fold=fold_id,
    gate_probability=gate_probability, gate_router=gate_router,
    probability_gap_router=gap_router,
)
diagnostics = {
    "complete_rows": int(complete.sum()),
    "disagreement_rows": int(disagreement.sum()),
    "rule_only_correct": int(np.sum(disagreement & (rule_pred == y) & (blend_pred != y))),
    "blend_only_correct": int(np.sum(disagreement & (blend_pred == y) & (rule_pred != y))),
    "both_wrong": int(np.sum(disagreement & (blend_pred != y) & (rule_pred != y))),
}
(OUT / "router_diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
print(summary.to_string(index=False))
print("\nTHRESHOLDS")
print(pd.DataFrame(gate_rows + gap_rows).to_string(index=False))
