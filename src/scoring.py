from __future__ import annotations

import json
from typing import Iterable

import pandas as pd

from .config import WEIGHTS


def compute_skill_gap(student_skills: Iterable[str], required_skills: Iterable[str]) -> list[str]:
    student_set = {str(skill).strip().lower() for skill in student_skills if str(skill).strip()}
    required_set = {str(skill).strip().lower() for skill in required_skills if str(skill).strip()}
    return sorted(required_set - student_set)


def compute_role_coverage(student_skills: Iterable[str], required_skills: Iterable[str]) -> float:
    student_set = {str(skill).strip().lower() for skill in student_skills if str(skill).strip()}
    required_set = {str(skill).strip().lower() for skill in required_skills if str(skill).strip()}
    if not required_set:
        return 0.0
    return len(student_set & required_set) / len(required_set)


def get_readiness_band(score: float) -> str:
    if score < 40:
        return "Needs Improvement"
    if score < 60:
        return "Developing"
    if score < 80:
        return "Job Ready"
    return "Highly Employable"


def compute_employability_score(student_row: pd.Series) -> tuple[float, str, list[str]]:
    cgpa = float(student_row.get("cgpa", 0.0) or 0.0)
    role_match = float(student_row.get("hybrid_score", 0.0) or 0.0)
    internship = 1.0 if int(student_row.get("has_internship", 0) or 0) == 1 else 0.0
    projects = min(float(student_row.get("num_projects", 0) or 0) / 5.0, 1.0)
    certifications = min(float(student_row.get("num_certifications", 0) or 0) / 4.0, 1.0)
    hackathons = min(float(student_row.get("hackathons_participated", 0) or 0) / 3.0, 1.0)
    github = 1.0 if int(student_row.get("active_github", 0) or 0) == 1 else 0.0
    communication = min(float(student_row.get("communication_rating", 0) or 0) / 5.0, 1.0)
    backlogs = float(student_row.get("backlogs", 0) or 0)

    score = (
        WEIGHTS.get("role_skill_match", 0.35) * role_match * 100.0
        + WEIGHTS.get("cgpa", 0.20) * min((cgpa / 10.0) * 100.0, 100.0)
        + WEIGHTS.get("internship", 0.15) * internship * 100.0
        + WEIGHTS.get("projects", 0.10) * projects * 100.0
        + WEIGHTS.get("certifications", 0.08) * certifications * 100.0
        + WEIGHTS.get("hackathons", 0.05) * hackathons * 100.0
        + WEIGHTS.get("github", 0.04) * github * 100.0
        + WEIGHTS.get("communication", 0.03) * communication * 100.0
        - WEIGHTS.get("backlog_penalty", 0.10) * min(backlogs, 5) * 10.0
    )

    score = max(0.0, min(100.0, score))
    band = get_readiness_band(score)

    missing_skills = student_row.get("skill_gap", [])
    recommendations = []
    if isinstance(missing_skills, list):
        for skill in missing_skills[:5]:
            recommendations.append({
                "skill": skill,
                "advice": f"Practice {skill} with a mini-project and targeted exercises to close the gap." 
            })

    return round(score, 2), band, recommendations


def build_gap_and_score(student_df: pd.DataFrame, job_df: pd.DataFrame) -> pd.DataFrame:
    output = student_df.copy()
    output["skill_gap"] = [[] for _ in range(len(output))]
    output["role_coverage"] = 0.0
    output["employability_score"] = 0.0
    output["readiness_band"] = "Needs Improvement"
    output["recommendations"] = [[] for _ in range(len(output))]
    for idx, row in output.iterrows():
        required = []
        target = str(row.get("target_role", "")).strip()
        match = job_df[job_df["role"].str.lower() == target.lower()]
        if not match.empty:
            required = [skill.strip() for skill in str(match.iloc[0].get("required_skills", "")).split(";") if skill.strip()]
        row_skills = row.get("skills", [])
        if isinstance(row_skills, str):
            row_skills = [row_skills]
        gap = compute_skill_gap(row_skills, required)
        coverage = compute_role_coverage(row_skills, required)
        score_row = row.copy()
        score_row["skill_gap"] = gap
        score_row["role_coverage"] = coverage
        score, band, recommendations = compute_employability_score(score_row)
        output.at[idx, "skill_gap"] = gap
        output.at[idx, "role_coverage"] = coverage
        output.at[idx, "employability_score"] = score
        output.at[idx, "readiness_band"] = band
        output.at[idx, "recommendations"] = recommendations
    return output


if __name__ == "__main__":
    print("Scoring functions loaded.")
