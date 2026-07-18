from pathlib import Path
import gc
import json
import platform
import time

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier


RANDOM_STATE = 42
N_SPLITS = 5
TARGET = "health_condition"
ID_COL = "id"
LABELS = ["unhealthy", "at-risk", "fit"]
LABEL_TO_INT = {label: i for i, label in enumerate(LABELS)}
OUT = Path("experiment_003_artifacts")
OUT.mkdir(exist_ok=True)


def add_features(df):
    df = df.copy()
    eps = 1e-6
    base = [c for c in df.columns if c not in [ID_COL, TARGET]]
    df["missing_count"] = df[base].isna().sum(axis=1).astype("int8")
    df["bmi_distance_normal"] = (df["bmi"] - 22.0).abs()
    df["sleep_distance_8h"] = (df["sleep_duration"] - 8.0).abs()
    df["steps_per_exercise_min"] = df["step_count"] / (df["exercise_duration"] + eps)
    df["calories_per_step"] = df["calorie_expenditure"] / (df["step_count"] + eps)
    df["water_per_calorie"] = df["water_intake"] / (df["calorie_expenditure"] + eps)
    return df.replace([np.inf, -np.inf], np.nan)


def model_params():
    return dict(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=15,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_lambda=5,
        tree_method="hist",
        device="cpu",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def make_native_frames(X_fit, X_eval, cat_cols):
    X_fit = X_fit.copy()
    X_eval = X_eval.copy()
    for col in cat_cols:
        # Learn the category dictionary from the fitting fold only. Unknown
        # validation categories become NaN and follow the learned missing path.
        levels = pd.Index(X_fit[col].dropna().unique())
        X_fit[col] = pd.Categorical(X_fit[col], categories=levels)
        X_eval[col] = pd.Categorical(X_eval[col], categories=levels)
    return X_fit, X_eval


train = add_features(pd.read_csv("train.csv"))
features = [c for c in train.columns if c not in [ID_COL, TARGET]]
cat_cols = train[features].select_dtypes(exclude=np.number).columns.tolist()
num_cols = [c for c in features if c not in cat_cols]
y = train[TARGET].map(LABEL_TO_INT).to_numpy()

folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
fold_id = np.full(len(train), -1, dtype=np.int8)
oof_preimputed = np.zeros((len(train), len(LABELS)), dtype=np.float32)
oof_native = np.zeros_like(oof_preimputed)
fold_rows = []
start = time.time()

for fold, (fit_idx, eval_idx) in enumerate(folds.split(train, y)):
    fold_id[eval_idx] = fold
    X_fit = train.iloc[fit_idx][features]
    X_eval = train.iloc[eval_idx][features]
    y_fit, y_eval = y[fit_idx], y[eval_idx]

    counts = np.bincount(y_fit, minlength=len(LABELS))
    class_weight = len(y_fit) / (len(LABELS) * counts)
    sample_weight = class_weight[y_fit]

    # Current baseline representation: median/mode imputation, explicit
    # numerical indicators, and ordinal categorical encoding.
    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median", add_indicator=True), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                encoded_missing_value=-1,
                dtype=np.float32,
            )),
        ]), cat_cols),
    ], verbose_feature_names_out=False)
    fit_encoded = preprocessor.fit_transform(X_fit)
    eval_encoded = preprocessor.transform(X_eval)
    pre_model = XGBClassifier(**model_params())
    pre_model.fit(
        fit_encoded,
        y_fit,
        sample_weight=sample_weight,
        eval_set=[(eval_encoded, y_eval)],
        verbose=False,
    )
    pre_prob = pre_model.predict_proba(eval_encoded)
    oof_preimputed[eval_idx] = pre_prob
    pre_pred = pre_prob.argmax(axis=1)

    del fit_encoded, eval_encoded, pre_model, preprocessor, pre_prob
    gc.collect()

    # Native representation: preserve every numerical and categorical NaN.
    # XGBoost learns a default missing direction at each tree split.
    native_fit, native_eval = make_native_frames(X_fit, X_eval, cat_cols)
    native_model = XGBClassifier(
        **model_params(),
        enable_categorical=True,
        missing=np.nan,
    )
    native_model.fit(
        native_fit,
        y_fit,
        sample_weight=sample_weight,
        eval_set=[(native_eval, y_eval)],
        verbose=False,
    )
    native_prob = native_model.predict_proba(native_eval)
    oof_native[eval_idx] = native_prob
    native_pred = native_prob.argmax(axis=1)

    for config, pred in [("preimputed", pre_pred), ("native_nan", native_pred)]:
        recalls = recall_score(y_eval, pred, labels=np.arange(len(LABELS)), average=None)
        fold_rows.append({
            "fold": fold,
            "config": config,
            "rows": len(eval_idx),
            "balanced_accuracy": balanced_accuracy_score(y_eval, pred),
            "accuracy": (y_eval == pred).mean(),
            **{f"recall_{label}": recalls[i] for i, label in enumerate(LABELS)},
        })

    print(
        f"Fold {fold}: preimputed={fold_rows[-2]['balanced_accuracy']:.6f} "
        f"native_nan={fold_rows[-1]['balanced_accuracy']:.6f}",
        flush=True,
    )
    del native_fit, native_eval, native_model, native_prob, X_fit, X_eval
    gc.collect()

assert (fold_id >= 0).all()
fold_scores = pd.DataFrame(fold_rows)
fold_scores.to_csv(OUT / "fold_scores.csv", index=False)
pd.DataFrame({ID_COL: train[ID_COL], "fold": fold_id}).to_csv(
    OUT / "fold_assignments.csv", index=False
)
np.savez_compressed(
    OUT / "oof_predictions.npz",
    y=y.astype(np.int8),
    fold=fold_id,
    preimputed=oof_preimputed,
    native_nan=oof_native,
)

summary_rows = []
confusions = {}
for config, prob in [("preimputed", oof_preimputed), ("native_nan", oof_native)]:
    pred = prob.argmax(axis=1)
    scores = fold_scores.loc[fold_scores.config == config, "balanced_accuracy"]
    recalls = recall_score(y, pred, labels=np.arange(len(LABELS)), average=None)
    summary_rows.append({
        "config": config,
        "cv_mean_balanced_accuracy": scores.mean(),
        "cv_std_balanced_accuracy": scores.std(ddof=1),
        "oof_balanced_accuracy": balanced_accuracy_score(y, pred),
        "oof_accuracy": (y == pred).mean(),
        "oof_errors": int((y != pred).sum()),
        **{f"oof_recall_{label}": recalls[i] for i, label in enumerate(LABELS)},
    })
    confusions[config] = confusion_matrix(y, pred, labels=np.arange(len(LABELS))).tolist()

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT / "summary.csv", index=False)
metadata = {
    "random_state": RANDOM_STATE,
    "n_splits": N_SPLITS,
    "split": "StratifiedKFold(shuffle=True)",
    "metric": "balanced_accuracy",
    "device": "cpu (XGBoost has no Apple MPS backend)",
    "parallelism": "n_jobs=-1 within each fold; folds run sequentially to avoid CPU oversubscription",
    "platform": platform.platform(),
    "versions": {
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "xgboost": xgboost.__version__,
    },
    "model_params": model_params(),
    "elapsed_seconds": time.time() - start,
    "confusion_matrices": confusions,
}
(OUT / "metadata.json").write_text(json.dumps(metadata, indent=2))
print("\nFOLD SCORES")
print(fold_scores.to_string(index=False))
print("\nSUMMARY")
print(summary.to_string(index=False))
print("\nMETADATA")
print(json.dumps(metadata, indent=2))
