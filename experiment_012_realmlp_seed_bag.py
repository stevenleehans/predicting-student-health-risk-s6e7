"""Experiment 012: honest local aggregation of RealMLP seeds 2026 and 2027."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, recall_score


LABELS = ["unhealthy", "at-risk", "fit"]
N_FOLDS = 5
OUT = Path("experiment_012_seed_bag_artifacts")
OUT.mkdir(exist_ok=True)


def metric_row(name, y, probability):
    pred = probability.argmax(1)
    recall = recall_score(y, pred, labels=np.arange(3), average=None)
    return {
        "candidate": name,
        "oof_balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "accuracy": float(np.mean(y == pred)),
        "errors": int(np.sum(y != pred)),
        **{f"recall_{label}": float(recall[i]) for i, label in enumerate(LABELS)},
    }


def best_weight(y, left, right):
    rows = []
    for weight in np.arange(0.0, 1.001, 0.01):
        probability = (1.0 - weight) * left + weight * right
        score = balanced_accuracy_score(y, probability.argmax(1))
        rows.append((float(score), float(weight)))
    return max(rows, key=lambda item: (item[0], -item[1]))


def crossfit(y, fold_id, left, right, label):
    output = np.zeros_like(left)
    rows = []
    for fold in range(N_FOLDS):
        meta = fold_id != fold
        valid = ~meta
        meta_score, weight = best_weight(y[meta], left[meta], right[meta])
        output[valid] = (1.0 - weight) * left[valid] + weight * right[valid]
        rows.append({
            "blend": label, "fold": fold, "right_weight": weight,
            "meta_balanced_accuracy": meta_score,
            "heldout_balanced_accuracy": float(
                balanced_accuracy_score(y[valid], output[valid].argmax(1))
            ),
        })
    return output, rows


old = np.load("experiment_010_artifacts/oof_predictions.npz")
new = np.load("experiment_012_seed_2027_artifacts/oof_predictions.npz")
old_test = np.load("experiment_010_artifacts/test_probabilities.npz")
new_test = np.load("experiment_012_seed_2027_artifacts/test_probabilities.npz")
y = old["y"]
fold_id = old["fold"]
assert np.array_equal(new["y"], y) and np.array_equal(new["fold"], fold_id)

hgbc = old["hgbc"].astype(np.float32)
seed_2026 = old["realmlp"].astype(np.float32)
seed_2027 = new["realmlp"].astype(np.float32)
trusted = old["crossfit_blend"].astype(np.float32)
equal_bag = (seed_2026 + seed_2027) / 2.0
seed_crossfit, seed_rows = crossfit(
    y, fold_id, seed_2026, seed_2027, "crossfit_seed_weight"
)
equal_bag_hgbc, bag_hgbc_rows = crossfit(
    y, fold_id, hgbc, equal_bag, "hgbc_plus_equal_seed_bag"
)
seed_crossfit_hgbc, seedcf_hgbc_rows = crossfit(
    y, fold_id, hgbc, seed_crossfit, "hgbc_plus_crossfit_seed_bag"
)

summary = pd.DataFrame([
    metric_row("experiment_010_trusted_blend", y, trusted),
    metric_row("realmlp_seed_2026", y, seed_2026),
    metric_row("realmlp_seed_2027", y, seed_2027),
    metric_row("realmlp_equal_seed_bag", y, equal_bag),
    metric_row("realmlp_crossfit_seed_weight", y, seed_crossfit),
    metric_row("hgbc_plus_equal_seed_bag_crossfit", y, equal_bag_hgbc),
    metric_row("hgbc_plus_crossfit_seed_bag_crossfit", y, seed_crossfit_hgbc),
]).sort_values("oof_balanced_accuracy", ascending=False)

all_rows = seed_rows + bag_hgbc_rows + seedcf_hgbc_rows
pd.DataFrame(all_rows).to_csv(OUT / "crossfit_weights.csv", index=False)
summary.to_csv(OUT / "summary.csv", index=False)
np.savez_compressed(
    OUT / "oof_predictions.npz", y=y, fold=fold_id, hgbc=hgbc,
    seed_2026=seed_2026, seed_2027=seed_2027, equal_bag=equal_bag,
    seed_crossfit=seed_crossfit, equal_bag_hgbc=equal_bag_hgbc,
    seed_crossfit_hgbc=seed_crossfit_hgbc,
)

equal_bag_test = (
    old_test["realmlp"].astype(np.float32)
    + new_test["realmlp"].astype(np.float32)
) / 2.0
_, full_bag_weight = best_weight(y, hgbc, equal_bag)
deployment = (
    (1.0 - full_bag_weight) * old_test["hgbc"].astype(np.float32)
    + full_bag_weight * equal_bag_test
)
np.savez_compressed(
    OUT / "test_probabilities.npz", equal_bag=equal_bag_test,
    diagnostic_deployment_blend=deployment,
)

metadata = {
    "scope": "local fixed-fold CV only",
    "seeds": [2026, 2027],
    "fixed_equal_seed_bag": True,
    "fold_source": "Experiment 003 fixed five folds",
    "deployment_weight_same_oof_diagnostic_only": full_bag_weight,
    "decision": "reject unless honest OOF exceeds 0.950635921",
}
(OUT / "metadata.json").write_text(json.dumps(metadata, indent=2))
print(summary.to_string(index=False))
print("\nCROSSFIT WEIGHTS")
print(pd.DataFrame(all_rows).to_string(index=False))
