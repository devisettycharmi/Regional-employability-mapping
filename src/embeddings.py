from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None

from .config import MODELS_DIR, RANDOM_STATE


MODEL_CACHE_DIR = MODELS_DIR


def _sentence_model(model_name: str = "all-MiniLM-L6-v2"):
    if SentenceTransformer is None:
        return None
    try:
        return SentenceTransformer(model_name)
    except Exception:
        return None


def build_text_profile(student_row: pd.Series) -> str:
    parts = []
    for field in ["skills_str", "skills_text", "project", "internship_detail", "target_role"]:
        if field in student_row.index and isinstance(student_row[field], str) and student_row[field].strip():
            parts.append(student_row[field])
    return " ".join(parts)


def build_job_profile(job_row: pd.Series) -> str:
    parts = []
    for field in ["role", "job_description", "required_skills", "good_to_have_skills"]:
        if field in job_row.index and isinstance(job_row[field], str) and job_row[field].strip():
            parts.append(job_row[field])
    return " ".join(parts)


def encode_profiles(
    student_df: pd.DataFrame,
    job_df: pd.DataFrame,
    model_name: str = "all-MiniLM-L6-v2",
    use_sentence_transformer: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    text_student = student_df.apply(build_text_profile, axis=1).fillna("")
    text_job = job_df.apply(build_job_profile, axis=1).fillna("")

    results: dict = {"method": "tfidf-fallback"}
    if use_sentence_transformer and SentenceTransformer is not None:
        model = _sentence_model(model_name)
        if model is not None:
            student_vectors = model.encode(text_student.tolist(), normalize_embeddings=True)
            job_vectors = model.encode(text_job.tolist(), normalize_embeddings=True)
            results["method"] = "sentence-transformers"
            np.save(MODELS_DIR / "student_embeddings.npy", student_vectors)
            np.save(MODELS_DIR / "job_embeddings.npy", job_vectors)
            return student_vectors, job_vectors, results

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    combined = pd.concat([text_student, text_job], ignore_index=True)
    matrix = vectorizer.fit_transform(combined)
    student_vectors = matrix[: len(text_student)]
    job_vectors = matrix[len(text_student) :]
    np.save(MODELS_DIR / "student_embeddings.npy", student_vectors.toarray())
    np.save(MODELS_DIR / "job_embeddings.npy", job_vectors.toarray())
    return student_vectors, job_vectors, results


def compute_job_similarity(student_vectors: np.ndarray, job_vectors: np.ndarray) -> np.ndarray:
    if hasattr(student_vectors, "toarray"):
        student_vectors = student_vectors.toarray()
    if hasattr(job_vectors, "toarray"):
        job_vectors = job_vectors.toarray()
    return cosine_similarity(student_vectors, job_vectors)


def get_top_roles(similarity_matrix: np.ndarray, job_df: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    rows = []
    for student_idx, row in enumerate(similarity_matrix):
        order = np.argsort(row)[::-1][:top_k]
        top = []
        for job_idx in order:
            top.append({
                "job_id": job_df.iloc[job_idx]["job_id"],
                "role": job_df.iloc[job_idx]["role"],
                "score": float(row[job_idx]),
            })
        rows.append({"top3_roles": [item["role"] for item in top], "top3_scores": [item["score"] for item in top]})
    return pd.DataFrame(rows)


def build_required_skill_overlap(student_skills: set[str], job_required: set[str]) -> float:
    if not job_required:
        return 0.0
    union = student_skills | job_required
    if not union:
        return 0.0
    return len(student_skills & job_required) / len(union)


if __name__ == "__main__":
    print("Embedding helpers ready.")
