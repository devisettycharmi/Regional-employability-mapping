from __future__ import annotations

import numpy as np
import pandas as pd

from .embeddings import build_job_profile, build_text_profile, compute_job_similarity


def assign_top_matches(student_df: pd.DataFrame, job_df: pd.DataFrame, student_vectors: np.ndarray, job_vectors: np.ndarray, top_k: int = 3) -> pd.DataFrame:
    similarity = compute_job_similarity(student_vectors, job_vectors)
    top_rows = []
    for idx, row in student_df.iterrows():
        scores = similarity[idx]
        order = np.argsort(scores)[::-1][:top_k]
        top_roles = [job_df.iloc[j]["role"] for j in order]
        top_scores = [float(scores[j]) for j in order]
        target_role = row.get("target_role", "")
        match_score = float(scores[0]) if len(scores) else 0.0
        if target_role:
            target_index = job_df[job_df["role"].str.lower() == str(target_role).lower()].index
            if len(target_index):
                match_score = float(scores[target_index[0]])
        top_rows.append({
            "student_id": row.get("student_id"),
            "top3_roles": top_roles,
            "top3_scores": top_scores,
            "match_score_target_role": match_score,
        })
    return pd.DataFrame(top_rows)


def compute_hybrid_score(student_skills: set[str], job_required: set[str], embedding_similarity: float, weight_embedding: float = 0.5, weight_overlap: float = 0.5) -> float:
    if not job_required:
        return float(embedding_similarity)
    overlap = len(student_skills & job_required) / max(1, len(job_required))
    return float(weight_embedding * embedding_similarity + weight_overlap * overlap)


def match_students_to_jobs(student_df: pd.DataFrame, job_df: pd.DataFrame, student_vectors: np.ndarray, job_vectors: np.ndarray, top_k: int = 3) -> pd.DataFrame:
    rows = []
    similarity = compute_job_similarity(student_vectors, job_vectors)
    for idx, row in student_df.iterrows():
        order = np.argsort(similarity[idx])[::-1][:top_k]
        target_role = str(row.get("target_role", "")).strip()
        target_idx = job_df[job_df["role"].str.lower() == target_role.lower()].index
        match_score = 0.0
        if len(target_idx):
            match_score = float(similarity[idx, target_idx[0]])
        top_roles = [job_df.iloc[j]["role"] for j in order]
        top_scores = [float(similarity[idx, j]) for j in order]
        rows.append({
            "student_id": row.get("student_id"),
            "target_role": target_role,
            "top3_roles": top_roles,
            "top3_scores": top_scores,
            "match_score_target_role": match_score,
        })
    return pd.DataFrame(rows)
