from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import MODELS_DIR, RAW_DIR, REPORTS_DIR
from .embeddings import encode_profiles
from .matching import match_students_to_jobs
from .model import train_and_save_model
from .preprocessing import preprocess_data
from .scoring import build_gap_and_score
from .skill_extraction import run_skill_extraction


def run_pipeline() -> dict:
    students, jobs, summary = preprocess_data()
    print("Phase 1 complete:")
    print(json.dumps({
        "shape": list(summary["shape"]),
        "duplicate_rows": summary["duplicate_rows"],
        "missing_items": summary["missing_values"],
    }, indent=2, default=str))

    students = run_skill_extraction(students, jobs)
    print("Phase 2 complete: extracted student skills")

    from .embeddings import encode_profiles
    student_vectors, job_vectors, embed_meta = encode_profiles(students, jobs)
    matches = match_students_to_jobs(students, jobs, student_vectors, job_vectors, top_k=3)
    students = students.merge(matches, on="student_id", how="left")
    print("Phase 3 complete: embeddings and role matching generated")

    for idx, row in students.iterrows():
        student_skills = set(str(row.get("skills_str", "")).split(";")) if row.get("skills_str") else set()
        target_role = row.get("target_role", "")
        job_match = jobs[jobs["role"].str.lower() == str(target_role).lower()]
        required = set()
        if not job_match.empty:
            required = set(str(job_match.iloc[0].get("required_skills", "")).split(";"))
        if row.get("top3_roles"):
            students.at[idx, "hybrid_score"] = 0.5 * float(row.get("match_score_target_role", 0.0)) + 0.5 * (len(student_skills & required) / max(1, len(required)))
        else:
            students.at[idx, "hybrid_score"] = 0.0

    students = build_gap_and_score(students, jobs)
    print("Phase 4 complete: employability scores and skill gaps computed")

    model_results = train_and_save_model(students)
    print("Phase 6 complete:")
    print(json.dumps(model_results, indent=2, default=str))

    return {
        "students": students,
        "jobs": jobs,
        "summary": summary,
        "embedding_method": embed_meta,
        "model": model_results,
    }


if __name__ == "__main__":
    run_pipeline()
