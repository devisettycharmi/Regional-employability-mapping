from __future__ import annotations

import json
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None

from .config import MODELS_DIR, RANDOM_STATE


def get_training_data(students: pd.DataFrame) -> pd.DataFrame:
    data = students.copy()
    data = data[data.get("year", "").astype(str).str.lower().str.contains("final year", case=False, na=False)]
    feature_cols = [
        "cgpa", "backlogs", "has_internship", "num_projects", "num_certifications",
        "hackathons_participated", "active_github", "communication_rating",
        "n_skills", "role_coverage", "hybrid_score", "branch", "college", "is_cs_branch"
    ]
    for col in feature_cols:
        if col not in data.columns:
            data[col] = 0
    return data[[*feature_cols, "is_placed"]].dropna()


def build_model_pipeline(model_name: str = "logistic") -> Pipeline:
    categorical = ["branch", "college"]
    numerical = [
        "cgpa", "backlogs", "has_internship", "num_projects", "num_certifications",
        "hackathons_participated", "active_github", "communication_rating",
        "n_skills", "role_coverage", "hybrid_score", "is_cs_branch"
    ]
    preprocess = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numerical),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    if model_name.lower() == "random_forest":
        model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, class_weight="balanced")
    elif model_name.lower() == "xgboost":
        if xgb is None:
            raise ImportError("xgboost is not available")
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=RANDOM_STATE, scale_pos_weight=1.5)
    else:
        model = LogisticRegression(max_iter=500, class_weight="balanced", random_state=RANDOM_STATE)
    return Pipeline([("prep", preprocess), ("model", model)])


def evaluate_model(model_name: str, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    pipeline = build_model_pipeline(model_name)
    y_pred = cross_val_predict(pipeline, X, y, cv=skf, method="predict")
    y_proba = cross_val_predict(pipeline, X, y, cv=skf, method="predict_proba")[:, 1]
    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y, y_proba),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
    }
    return metrics


def train_and_save_model(students: pd.DataFrame, model_name: str = "logistic") -> dict[str, Any]:
    training = get_training_data(students)
    X = training.drop(columns=["is_placed"])
    y = training["is_placed"]
    pipeline = build_model_pipeline(model_name)
    pipeline.fit(X, y)
    model_path = MODELS_DIR / "placement_model.joblib"
    joblib.dump(pipeline, model_path)
    metrics = evaluate_model(model_name, X, y)
    return {"model_name": model_name, "metrics": metrics, "model_path": str(model_path)}


if __name__ == "__main__":
    print("Placement model module ready.")
