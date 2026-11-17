from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .config import MODELS_DIR, RANDOM_STATE


def choose_best_k(vector_matrix: np.ndarray, k_min: int = 3, k_max: int = 10) -> tuple[int, KMeans, float]:
    best_k = None
    best_model = None
    best_score = -1.0
    if len(vector_matrix) < k_min:
        return 3, KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10), 0.0
    for k in range(k_min, min(k_max, len(vector_matrix)) + 1):
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = model.fit_predict(vector_matrix)
        score = silhouette_score(vector_matrix, labels) if k > 1 else 0.0
        if score > best_score:
            best_score = score
            best_k = k
            best_model = model
    return best_k or 3, best_model or KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10), best_score


def label_clusters(model: KMeans, feature_vectors: np.ndarray) -> dict[int, str]:
    labels = model.fit_predict(feature_vectors)
    cluster_skills: dict[int, list[str]] = {int(label): [] for label in set(labels)}
    # Here we only assign a generic label. Feature-level labels can be replaced in a richer pipeline.
    for label in sorted(cluster_skills):
        cluster_skills[label] = [f"Cluster {label}"]
    return {key: values[0] for key, values in cluster_skills.items()}


def cluster_students(student_vectors: np.ndarray) -> tuple[dict, float, KMeans]:
    k, model, score = choose_best_k(student_vectors)
    labels = model.fit_predict(student_vectors)
    labels_map = label_clusters(model, student_vectors)
    return {"k": k, "silhouette": score, "labels": labels, "label_map": labels_map}, score, model


if __name__ == "__main__":
    print("Clustering module ready.")
