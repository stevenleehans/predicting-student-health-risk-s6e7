from pathlib import Path
import json
import platform
import time

import numpy as np
import pandas as pd
import sklearn
import xgboost
from xgboost import XGBClassifier


RANDOM_STATE = 42
TARGET = "health_condition"
ID_COL = "id"
LABELS = ["unhealthy", "at-risk", "fit"]
LABEL_TO_INT = {label: i for i, label in enumerate(LABELS)}
INT_TO_LABEL = {i: label for label, i in LABEL_TO_INT.items()}
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


params = dict(
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
    enable_categorical=True,
    missing=np.nan,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

train = add_features(pd.read_csv("train.csv"))
test = add_features(pd.read_csv("test.csv"))
sample = pd.read_csv("sample_submission.csv")
features = [c for c in test.columns if c != ID_COL]
cat_cols = train[features].select_dtypes(exclude=np.number).columns.tolist()

for col in cat_cols:
    levels = pd.Index(train[col].dropna().unique())
    train[col] = pd.Categorical(train[col], categories=levels)
    test[col] = pd.Categorical(test[col], categories=levels)

y = train[TARGET].map(LABEL_TO_INT).to_numpy()
counts = np.bincount(y, minlength=len(LABELS))
class_weight = len(y) / (len(LABELS) * counts)
sample_weight = class_weight[y]

start = time.time()
model = XGBClassifier(**params)
model.fit(train[features], y, sample_weight=sample_weight, verbose=False)
test_prob = model.predict_proba(test[features])
test_pred = test_prob.argmax(axis=1)

submission = sample.copy()
submission[TARGET] = pd.Series(test_pred).map(INT_TO_LABEL)
assert submission.shape == sample.shape
assert submission[ID_COL].equals(test[ID_COL].reset_index(drop=True))
assert submission[TARGET].isin(LABELS).all()
assert not submission.isna().any().any()
submission.to_csv("submission_native_nan_xgboost.csv", index=False)

importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)
importance["rank"] = np.arange(1, len(importance) + 1)
importance.to_csv(OUT / "full_model_feature_importance.csv", index=False)

metadata = {
    "train_rows": len(train),
    "test_rows": len(test),
    "submission_file": "submission_native_nan_xgboost.csv",
    "prediction_counts": submission[TARGET].value_counts().to_dict(),
    "prediction_shares": submission[TARGET].value_counts(normalize=True).to_dict(),
    "device": "cpu (XGBoost has no Apple MPS backend)",
    "parallelism": "n_jobs=-1",
    "elapsed_seconds": time.time() - start,
    "platform": platform.platform(),
    "versions": {
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "xgboost": xgboost.__version__,
    },
    "params": params,
}
(OUT / "full_train_metadata.json").write_text(json.dumps(metadata, indent=2))
print(json.dumps(metadata, indent=2))
print(importance.head(20).to_string(index=False))
