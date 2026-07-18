from pathlib import Path
import gc
import json
import math
import os
import platform
import random
import time

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.preprocessing import TargetEncoder
from sklearn.utils.class_weight import compute_class_weight


SEED = int(os.environ.get("REALMLP_SEED", "2026"))
N_FOLDS = 5
EPOCHS = 5
TRAIN_BS = 512
EVAL_BS = 10240
EMBED_DIM = 4
LR = 0.01
N_ENS = 16
ONEHOTMAX = 10
LABEL_SMOOTHING = 0.05
EMA_DECAY = 0.997
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
OUT = Path(os.environ.get("REALMLP_OUT", "experiment_010_artifacts"))
OUT.mkdir(exist_ok=True)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def feature_engineer(frame):
    frame = frame.copy()
    for col in CATEGORICAL_COLS:
        frame[col] = frame[col].fillna("missing").astype(str)
    for col in NUMERIC_COLS:
        frame[col] = frame[col].fillna(0)

    for col in NUMERIC_COLS:
        if col == "step_count":
            frame[f"{col}_cat"] = (frame[col] // 10).astype(str)
        elif col == "calorie_expenditure":
            frame[f"{col}_cat"] = (frame[col] // 5).astype(str)
        else:
            frame[f"{col}_cat"] = frame[col].astype(str)

    for col in NUMERIC_COLS:
        if col == "step_count":
            value = frame[col] // 20
        elif col == "calorie_expenditure":
            value = frame[col] // 50
        elif col == "water_intake":
            value = (frame[col] * 50).astype(np.int64)
        elif col in ("heart_rate", "bmi"):
            value = (frame[col] * 5).astype(np.int64)
        else:
            value = frame[col] // 2
        frame[f"{col}_cat2"] = value.astype(str)
    return frame


class RobustScaleSmoothClipTransform(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.median_ = np.median(X, axis=0)
        quant_diff = np.quantile(X, 0.75, axis=0) - np.quantile(X, 0.25, axis=0)
        zero = quant_diff == 0.0
        quant_diff[zero] = 0.5 * (np.max(X, axis=0)[zero] - np.min(X, axis=0)[zero])
        self.factors_ = 1.0 / (quant_diff + 1e-30)
        self.factors_[quant_diff == 0.0] = 0.0
        return self

    def transform(self, X, y=None):
        scaled = self.factors_[None, :] * (X - self.median_[None, :])
        return scaled / np.sqrt(1 + (scaled / 3) ** 2)


class ScalingLayer(nn.Module):
    def __init__(self, n_ens, n_features):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(n_ens, n_features))

    def forward(self, x):
        return x * self.scale[None, :, :]


class CategoricalFeatureLayer(nn.Module):
    def __init__(self, n_ens, cat_dims, embed_dim=8):
        super().__init__()
        self.n_ens = n_ens
        self.embed_dim = embed_dim
        self.cat_dims = cat_dims
        self.onehot_features, self.binary_features = [], []
        self.embed_features, self.embed_dims, self.embed_offsets = [], [], []
        for idx, dim in enumerate(cat_dims):
            if dim == 2:
                self.binary_features.append(idx)
            elif dim <= ONEHOTMAX:
                self.onehot_features.append(idx)
            else:
                self.embed_features.append(idx)
                self.embed_dims.append(dim)

        self.combined_emb = None
        if self.embed_features:
            self.combined_emb = nn.Embedding(
                int(sum(self.embed_dims) * n_ens), embed_dim, padding_idx=0
            )
            offset = 0
            for dim in self.embed_dims:
                self.embed_offsets.append(offset)
                offset += dim
            self.per_ens_offset = sum(self.embed_dims)

    def forward(self, x):
        batch_size, n_ens, _ = x.shape
        features = []
        if self.binary_features:
            features.append(2 * x[:, :, self.binary_features].float() - 1)
        if self.onehot_features:
            onehot_x = x[:, :, self.onehot_features]
            dims = [self.cat_dims[i] for i in self.onehot_features]
            encoded = torch.zeros(batch_size, n_ens, sum(dims), device=x.device)
            start = 0
            for idx, dim in enumerate(dims):
                position = onehot_x[:, :, idx:idx + 1].long().clamp(0, dim - 1)
                encoded.scatter_(2, position + start, 1.0)
                start += dim
            features.append(encoded)
        if self.embed_features:
            embed_x = x[:, :, self.embed_features].long()
            ens_offset = torch.arange(n_ens, device=x.device) * self.per_ens_offset
            feat_offset = torch.tensor(self.embed_offsets, device=x.device)
            indices = (
                embed_x + feat_offset[None, None, :] + ens_offset[None, :, None]
            )
            embedded = self.combined_emb(indices)
            features.append(embedded.reshape(batch_size, n_ens, -1))
        return torch.cat(features, dim=2)


class PBLDEmbedding(nn.Module):
    def __init__(self, n_ens, n_features, hidden_dim=16, out_dim=4, freq_scale=0.1):
        super().__init__()
        self.n_ens, self.n_features = n_ens, n_features
        self.w1 = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim) * freq_scale)
        self.b1 = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim))
        self.w2 = nn.Parameter(
            torch.randn(n_ens, n_features, hidden_dim, out_dim - 1)
            / np.sqrt(hidden_dim)
        )
        self.b2 = nn.Parameter(torch.randn(n_ens, n_features, out_dim - 1))
        self.act = nn.GELU()
        nn.init.uniform_(self.b1, -np.pi, np.pi)

    def forward(self, x):
        periodic = torch.cos(
            2 * np.pi * (x.unsqueeze(-1) * self.w1.unsqueeze(0) + self.b1.unsqueeze(0))
        )
        transformed = torch.einsum("bnfh,nfhd->bnfd", periodic, self.w2)
        transformed = self.act(transformed + self.b2.unsqueeze(0))
        result = torch.cat([x.unsqueeze(-1), transformed], dim=-1)
        return result.reshape(x.shape[0], self.n_ens, -1)


class NTPLinear(nn.Module):
    def __init__(self, n_ens, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.weight = nn.Parameter(torch.randn(n_ens, in_features, out_features))
        self.bias = nn.Parameter(torch.randn(n_ens, out_features)) if bias else None

    def forward(self, x):
        result = torch.einsum("bni,nio->bno", x, self.weight) / np.sqrt(self.in_features)
        return result + self.bias if self.bias is not None else result


class RankGLUHead(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=128, output_dim=3, gamma=0.5):
        super().__init__()
        self.gamma = gamma
        self.layer_norm = nn.LayerNorm(input_dim)
        self.linear_score = nn.Linear(input_dim, output_dim)
        self.glu_v = nn.Linear(input_dim, bottleneck_dim)
        self.glu_g = nn.Linear(input_dim, bottleneck_dim)
        self.glu_out = nn.Linear(bottleneck_dim, output_dim)

    def forward(self, x):
        normalized = self.layer_norm(x)
        gate = torch.sigmoid(self.glu_g(normalized))
        return self.linear_score(normalized) + self.gamma * self.glu_out(
            self.glu_v(normalized) * gate
        )


class RealMLP(nn.Module):
    def __init__(self, output_dim, cat_dims, n_numerical, n_ens, embed_dim):
        super().__init__()
        self.n_ens = n_ens
        self.cate = CategoricalFeatureLayer(n_ens, cat_dims, embed_dim)
        self.num_embed = PBLDEmbedding(
            n_ens, n_numerical, hidden_dim=20, out_dim=5, freq_scale=5.0
        )
        cat_emb_dim = sum(dim if dim <= ONEHOTMAX else embed_dim for dim in cat_dims)
        total_dim = n_numerical * 5 + cat_emb_dim
        self.dropout = nn.Dropout(0.03)
        self.first_linear = NTPLinear(n_ens, total_dim, 256)
        self.model = nn.Sequential(
            ScalingLayer(n_ens, total_dim), self.dropout, self.first_linear,
            NTPLinear(n_ens, 256, 512), nn.GELU(),
            NTPLinear(n_ens, 512, 128), nn.GELU(), self.dropout,
        )
        self.cls_head = RankGLUHead(128, 128, output_dim, 0.5)

    def forward(self, x_num, x_cat):
        x_num = x_num.unsqueeze(1).expand(-1, self.n_ens, -1)
        x_cat = x_cat.unsqueeze(1).expand(-1, self.n_ens, -1)
        return self.cls_head(self.model(torch.cat([
            self.num_embed(x_num), self.cate(x_cat)
        ], dim=2)))


def parameter_groups(model):
    first_weight_id = id(model.first_linear.weight)
    groups = ([], [], [], [], [])
    for name, parameter in model.named_parameters():
        if "scale" in name:
            groups[0].append(parameter)
        elif "num_embed" in name:
            groups[1].append(parameter)
        elif id(parameter) == first_weight_id:
            groups[2].append(parameter)
        elif "bias" in name:
            groups[4].append(parameter)
        else:
            groups[3].append(parameter)
    return groups


def flat_anneal(initial, progress, flat_ratio=0.5):
    if progress < flat_ratio:
        return initial
    return initial * (1 - (progress - flat_ratio) / (1 - flat_ratio))


def cosine_schedule(initial, progress):
    return initial * (math.cos(math.pi * progress) + 1) / 2


def predict_probability(model, x_num, x_cat):
    chunks = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(x_num), EVAL_BS):
            logits = model(x_num[start:start + EVAL_BS], x_cat[start:start + EVAL_BS])
            chunks.append(F.softmax(logits, dim=-1).mean(dim=1).cpu().numpy())
    return np.concatenate(chunks).astype(np.float32)


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


def best_realmlp_weight(y_true, base_probability, realmlp_probability):
    rows = []
    for weight in np.arange(0.0, 1.001, 0.01):
        probability = (1 - weight) * base_probability + weight * realmlp_probability
        rows.append((balanced_accuracy_score(y_true, probability.argmax(axis=1)), weight))
    return max(rows, key=lambda row: (row[0], -row[1]))


start_time = time.time()
seed_everything(SEED)
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
if device.type != "mps":
    print(f"WARNING: MPS unavailable; running RealMLP on {device}", flush=True)
else:
    print("Using Apple MPS", flush=True)

train_raw = pd.read_csv("train.csv")
test_raw = pd.read_csv("test.csv")
sample = pd.read_csv("sample_submission.csv")
y = train_raw[TARGET].map(LABEL_TO_INT).to_numpy(dtype=np.int64)
fold_map = pd.read_csv("experiment_003_artifacts/fold_assignments.csv")
assert fold_map[ID_COL].equals(train_raw[ID_COL])
fold_id = fold_map["fold"].to_numpy(dtype=np.int8)

base_saved = np.load("experiment_007_artifacts/oof_predictions.npz")
base_test_saved = np.load("experiment_007_artifacts/test_probabilities.npz")
assert np.array_equal(base_saved["y"], y)
assert np.array_equal(base_saved["fold"], fold_id)
base_oof = base_saved["hgbc_te"].astype(np.float32)
base_test = base_test_saved["hgbc_te"].astype(np.float32)

train = feature_engineer(train_raw.drop(columns=[TARGET]))
test = feature_engineer(test_raw)
feature_columns = [col for col in test.columns if col != ID_COL]
cat_cols = [
    col for col in feature_columns
    if not pd.api.types.is_numeric_dtype(test[col]) or train[col].nunique() <= ONEHOTMAX
]
num_cols = [col for col in feature_columns if col not in cat_cols]
te_cols = [col for col in cat_cols if "_cat" in col]

cat_dims = []
for col in cat_cols:
    mapping = {value: idx + 1 for idx, value in enumerate(train[col].unique())}
    train[col] = train[col].map(mapping).fillna(0).astype(np.int64)
    test[col] = test[col].map(mapping).fillna(0).astype(np.int64)
    cat_dims.append(int(train[col].max()) + 1)

X_cat_all = train[cat_cols].to_numpy(dtype=np.int64)
X_cat_test_all = test[cat_cols].to_numpy(dtype=np.int64)
X_num_all = train[num_cols].to_numpy(dtype=np.float32)
X_num_test_all = test[num_cols].to_numpy(dtype=np.float32)

oof = np.zeros((len(train), 3), dtype=np.float32)
test_probability = np.zeros((len(test), 3), dtype=np.float64)
fold_rows, epoch_rows = [], []

for fold in range(N_FOLDS):
    seed_everything(SEED + fold)
    fit_idx = np.flatnonzero(fold_id != fold)
    eval_idx = np.flatnonzero(fold_id == fold)
    y_fit, y_eval = y[fit_idx], y[eval_idx]
    X_num_fit, X_num_eval = X_num_all[fit_idx], X_num_all[eval_idx]
    X_cat_fit, X_cat_eval = X_cat_all[fit_idx], X_cat_all[eval_idx]

    encoder = TargetEncoder(
        cv=5, smooth="auto", target_type="multiclass",
        shuffle=True, random_state=SEED + fold,
    )
    fit_encoded = encoder.fit_transform(
        pd.DataFrame(X_cat_fit, columns=cat_cols)[te_cols], y_fit
    )
    eval_encoded = encoder.transform(
        pd.DataFrame(X_cat_eval, columns=cat_cols)[te_cols]
    )
    test_encoded = encoder.transform(
        pd.DataFrame(X_cat_test_all, columns=cat_cols)[te_cols]
    )
    X_num_fit = np.concatenate([X_num_fit, fit_encoded], axis=1)
    X_num_eval = np.concatenate([X_num_eval, eval_encoded], axis=1)
    X_num_test = np.concatenate([X_num_test_all, test_encoded], axis=1)

    scaler = RobustScaleSmoothClipTransform().fit(X_num_fit)
    X_num_fit = scaler.transform(X_num_fit).astype(np.float32)
    X_num_eval = scaler.transform(X_num_eval).astype(np.float32)
    X_num_test = scaler.transform(X_num_test).astype(np.float32)

    model = RealMLP(3, cat_dims, X_num_fit.shape[1], N_ENS, EMBED_DIM).to(device)
    scale_p, pbld_p, first_w_p, other_w_p, bias_p = parameter_groups(model)
    optimizer = torch.optim.AdamW([
        {"params": scale_p, "lr": LR * 20.0, "weight_decay": 1e-3},
        {"params": pbld_p, "lr": LR * 0.093, "weight_decay": 1e-2},
        {"params": first_w_p, "lr": LR, "weight_decay": 1e-3},
        {"params": other_w_p, "lr": LR, "weight_decay": 1e-2},
        {"params": bias_p, "lr": LR * 0.1, "weight_decay": 5e-3},
    ], betas=(0.9, 0.98))

    # Source notebook tweaks in alphabetical class order were [at-risk=.9,
    # fit=1.1, unhealthy=1.0]. Reordered here to LABELS.
    class_weights = compute_class_weight("balanced", classes=np.arange(3), y=y_fit)
    class_weights *= np.array([1.0, 0.9, 1.1])
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)

    x_num_fit_t = torch.as_tensor(X_num_fit, dtype=torch.float32, device=device)
    x_cat_fit_t = torch.as_tensor(X_cat_fit, dtype=torch.long, device=device)
    y_fit_t = torch.as_tensor(y_fit, dtype=torch.long, device=device)
    x_num_eval_t = torch.as_tensor(X_num_eval, dtype=torch.float32, device=device)
    x_cat_eval_t = torch.as_tensor(X_cat_eval, dtype=torch.long, device=device)
    x_num_test_t = torch.as_tensor(X_num_test, dtype=torch.float32, device=device)
    x_cat_test_t = torch.as_tensor(X_cat_test_all, dtype=torch.long, device=device)

    train_indices = np.arange(len(y_fit))
    ema_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_score, best_epoch, best_state = -1.0, -1, None

    for epoch in range(EPOCHS):
        epoch_loss, batches = 0.0, 0
        for batch_start in range(0, len(y_fit), TRAIN_BS):
            progress = epoch / EPOCHS + batch_start / (len(y_fit) * EPOCHS)
            batch_idx = train_indices[batch_start:batch_start + TRAIN_BS]
            model.train()
            for group, initial in zip(
                optimizer.param_groups, [LR * 20.0, LR * 0.093, LR, LR, LR * 0.1]
            ):
                group["lr"] = flat_anneal(initial, progress)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_num_fit_t[batch_idx], x_cat_fit_t[batch_idx])
            loss = F.cross_entropy(
                logits.reshape(-1, 3),
                y_fit_t[batch_idx].repeat_interleave(N_ENS),
                weight=class_weights_t,
                label_smoothing=cosine_schedule(LABEL_SMOOTHING, progress),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            batches += 1
            with torch.no_grad():
                for key, value in model.state_dict().items():
                    if torch.is_floating_point(value):
                        ema_state[key].mul_(EMA_DECAY).add_(
                            value.detach(), alpha=1.0 - EMA_DECAY
                        )
                    else:
                        ema_state[key].copy_(value)

        eval_score = np.nan
        if epoch > 0:
            live_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            model.load_state_dict(ema_state, strict=True)
            eval_probability = predict_probability(model, x_num_eval_t, x_cat_eval_t)
            eval_score = balanced_accuracy_score(y_eval, eval_probability.argmax(axis=1))
            if eval_score > best_score:
                best_score, best_epoch = float(eval_score), epoch
                best_state = {key: value.detach().cpu().clone() for key, value in ema_state.items()}
            model.load_state_dict(live_state, strict=True)
        epoch_rows.append({
            "fold": fold, "epoch": epoch, "loss": epoch_loss / batches,
            "eval_balanced_accuracy": eval_score, "best_score_so_far": best_score,
        })
        print(
            f"Fold {fold} epoch {epoch + 1}/{EPOCHS}: loss={epoch_loss / batches:.5f} "
            f"eval={eval_score:.6f} best={best_score:.6f}", flush=True,
        )
        np.random.shuffle(train_indices)

    assert best_state is not None
    model.load_state_dict(best_state, strict=True)
    model.to(device)
    eval_probability = predict_probability(model, x_num_eval_t, x_cat_eval_t)
    oof[eval_idx] = eval_probability
    test_probability += predict_probability(model, x_num_test_t, x_cat_test_t) / N_FOLDS
    final_fold_score = balanced_accuracy_score(y_eval, eval_probability.argmax(axis=1))
    fold_rows.append({
        "fold": fold, "rows": len(eval_idx), "best_epoch": best_epoch + 1,
        "balanced_accuracy": final_fold_score,
        "class_weights": json.dumps(class_weights.tolist()),
    })
    print(
        f"Fold {fold} complete: RealMLP={final_fold_score:.6f} best_epoch={best_epoch + 1}",
        flush=True,
    )
    np.savez_compressed(
        OUT / "checkpoint_predictions.npz",
        completed_fold=np.array([fold]), y=y, fold=fold_id,
        realmlp=oof, test_realmlp=test_probability.astype(np.float32),
    )

    del encoder, scaler, model, optimizer, ema_state, best_state
    del x_num_fit_t, x_cat_fit_t, y_fit_t, x_num_eval_t, x_cat_eval_t
    del x_num_test_t, x_cat_test_t, fit_encoded, eval_encoded, test_encoded
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

pd.DataFrame(fold_rows).to_csv(OUT / "fold_scores.csv", index=False)
pd.DataFrame(epoch_rows).to_csv(OUT / "epoch_scores.csv", index=False)
test_probability = test_probability.astype(np.float32)

crossfit_blend = np.zeros_like(oof)
blend_rows = []
for fold in range(N_FOLDS):
    meta_fit = np.flatnonzero(fold_id != fold)
    meta_eval = np.flatnonzero(fold_id == fold)
    train_score, weight = best_realmlp_weight(
        y[meta_fit], base_oof[meta_fit], oof[meta_fit]
    )
    probability = (1 - weight) * base_oof[meta_eval] + weight * oof[meta_eval]
    crossfit_blend[meta_eval] = probability
    blend_rows.append({
        "fold": fold, "realmlp_weight_learned_on_other_folds": weight,
        "meta_train_balanced_accuracy": train_score,
        "eval_balanced_accuracy": balanced_accuracy_score(
            y[meta_eval], probability.argmax(axis=1)
        ),
    })
pd.DataFrame(blend_rows).to_csv(OUT / "crossfit_blend_scores.csv", index=False)

full_blend_score, full_weight = best_realmlp_weight(y, base_oof, oof)
deployment_blend = (1 - full_weight) * base_test + full_weight * test_probability
summary = pd.DataFrame([
    metric_row("experiment_007_hgbc", y, base_oof),
    metric_row("realmlp", y, oof),
    metric_row("hgbc_realmlp_crossfit_blend", y, crossfit_blend),
])
summary.to_csv(OUT / "summary.csv", index=False)

sleep_missing = train_raw["sleep_duration"].isna().to_numpy()
slice_rows = []
for model_name, probability in [
    ("experiment_007_hgbc", base_oof), ("realmlp", oof),
    ("hgbc_realmlp_crossfit_blend", crossfit_blend),
]:
    for slice_name, mask in [
        ("sleep_missing", sleep_missing), ("sleep_present", ~sleep_missing),
    ]:
        pred = probability[mask].argmax(axis=1)
        slice_rows.append({
            "model": model_name, "slice": slice_name, "rows": int(mask.sum()),
            "balanced_accuracy": balanced_accuracy_score(y[mask], pred),
            "error_rate": float((y[mask] != pred).mean()),
        })
pd.DataFrame(slice_rows).to_csv(OUT / "error_slices.csv", index=False)

np.savez_compressed(OUT / "oof_predictions.npz", y=y, fold=fold_id, realmlp=oof,
                    crossfit_blend=crossfit_blend, hgbc=base_oof)
np.savez_compressed(OUT / "test_probabilities.npz", realmlp=test_probability,
                    deployment_blend=deployment_blend.astype(np.float32), hgbc=base_test)

for name, probability in [
    ("submission_experiment_010_realmlp.csv", test_probability),
    ("submission_experiment_010_hgbc_realmlp_blend.csv", deployment_blend),
]:
    submission = sample.copy()
    submission[TARGET] = pd.Series(probability.argmax(axis=1)).map(INT_TO_LABEL)
    assert submission[ID_COL].equals(test_raw[ID_COL].reset_index(drop=True))
    assert submission[TARGET].isin(LABELS).all()
    submission.to_csv(name, index=False)

metadata = {
    "device": str(device), "mps_built": torch.backends.mps.is_built(),
    "mps_available": torch.backends.mps.is_available(),
    "seed": SEED, "fold_source": "experiment_003_artifacts/fold_assignments.csv",
    "epochs": EPOCHS, "train_batch_size": TRAIN_BS, "eval_batch_size": EVAL_BS,
    "n_ens": N_ENS, "embed_dim": EMBED_DIM, "learning_rate": LR,
    "label_smoothing": LABEL_SMOOTHING, "ema_decay": EMA_DECAY,
    "categorical_columns": cat_cols, "numerical_columns": num_cols,
    "target_encoded_columns": te_cols, "categorical_dimensions": cat_dims,
    "full_oof_blend_realmlp_weight": float(full_weight),
    "full_oof_blend_score_same_data": float(full_blend_score),
    "elapsed_seconds": time.time() - start_time,
    "platform": platform.platform(),
    "versions": {
        "numpy": np.__version__, "pandas": pd.__version__,
        "sklearn": sklearn.__version__, "torch": torch.__version__,
    },
}
(OUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

print("\nSUMMARY", flush=True)
print(summary.to_string(index=False), flush=True)
print("\nCROSSFIT BLEND", flush=True)
print(pd.DataFrame(blend_rows).to_string(index=False), flush=True)
print("\nSLEEP SLICES", flush=True)
print(pd.DataFrame(slice_rows).to_string(index=False), flush=True)
print("\nMETADATA", flush=True)
print(json.dumps(metadata, indent=2), flush=True)
