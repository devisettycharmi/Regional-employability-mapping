from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .config import FIGURES_DIR, PROCESSED_DIR, RAW_DIR, WEIGHTS


def normalize_column_names(columns: list[str]) -> list[str]:
    renamed: list[str] = []
    for col in columns:
        transformed = col.strip().lower()
        transformed = re.sub(r"[^a-z0-9]+", "_", transformed)
        transformed = re.sub(r"_+", "_", transformed).strip("_")
        if transformed == "student_id":
            renamed.append("student_id")
        elif transformed in {"full_name", "name"}:
            renamed.append("name")
        elif transformed in {"college_institution", "college"}:
            renamed.append("college")
        elif transformed in {"college_locality", "college_locality"}:
            renamed.append("college_locality")
        elif transformed in {"home_locality", "home_location"}:
            renamed.append("home_locality")
        elif transformed in {"branch_department", "department"}:
            renamed.append("branch")
        elif transformed in {"current_year", "year"}:
            renamed.append("year")
        elif transformed in {"current_cgpa", "cgpa"}:
            renamed.append("cgpa")
        elif transformed in {"which_technical_skills_do_you_currently_have", "skills_listed"}:
            renamed.append("skills_listed")
        elif transformed in {"describe_your_technical_skills_in_your_own_words", "skills_text"}:
            renamed.append("skills_text")
        elif transformed in {"describe_your_most_important_project", "project"}:
            renamed.append("project")
        elif transformed in {"do_you_have_any_internship_experience", "has_internship"}:
            renamed.append("has_internship")
        elif transformed in {"if_yes_mention_domain_and_technologies", "internship_detail"}:
            renamed.append("internship_detail")
        elif transformed in {"which_job_role_are_you_interested_in", "target_role"}:
            renamed.append("target_role")
        elif transformed in {"where_would_you_prefer_to_work", "pref_location"}:
            renamed.append("pref_location")
        elif transformed in {"which_skills_do_you_feel_you_are_currently_lagging", "lagging"}:
            renamed.append("lagging")
        elif transformed in {"which_skills_do_you_want_to_learn_in_next_6_12_months", "want_to_learn"}:
            renamed.append("want_to_learn")
        elif transformed in {"what_is_your_placement_status", "placement_status"}:
            renamed.append("placement_status")
        elif transformed in {"how_job_ready_do_you_consider_yourself_1_10", "self_readiness"}:
            renamed.append("self_readiness")
        elif transformed in {"num_projects", "num_projects"}:
            renamed.append("num_projects")
        elif transformed in {"num_certifications", "num_certifications"}:
            renamed.append("num_certifications")
        elif transformed in {"hackathons_participated", "hackathons_participated"}:
            renamed.append("hackathons_participated")
        elif transformed in {"active_github", "active_github"}:
            renamed.append("active_github")
        elif transformed in {"backlogs", "backlogs"}:
            renamed.append("backlogs")
        elif transformed in {"communication_rating", "communication_rating"}:
            renamed.append("communication_rating")
        elif transformed in {"is_placed", "is_placed"}:
            renamed.append("is_placed")
        elif transformed in {"is_cs_branch", "is_cs_branch"}:
            renamed.append("is_cs_branch")
        elif transformed in {"n_skills", "n_skills"}:
            renamed.append("n_skills")
        elif transformed in {"n_required", "n_required"}:
            renamed.append("n_required")
        elif transformed in {"role_coverage", "role_coverage"}:
            renamed.append("role_coverage")
        elif transformed in {"skills_str", "skills_str"}:
            renamed.append("skills_str")
        elif transformed in {"gap_str", "gap_str"}:
            renamed.append("gap_str")
        else:
            renamed.append(transformed)
    return renamed


def clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    text = text.replace("\u00a0", " ")
    text = text.replace("�", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def normalise_text(value: Any) -> str:
    text = clean_text(value).lower()
    replacements = {
        "ml": "machine learning",
        "ai": "artificial intelligence",
        "ds": "data science",
        "dsa": "data structures and algorithms",
        "js": "javascript",
        "ts": "typescript",
        "sql": "sql",
        "rdbms": "relational database systems",
        "ui": "user interface",
        "ux": "user experience",
        "cv": "computer vision",
        "nlp": "natural language processing",
    }
    for old, new in replacements.items():
        text = re.sub(rf"\b{re.escape(old)}\b", new, text)
    text = re.sub(r"[^a-z0-9\s+/.-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def coerce_numeric(series: pd.Series) -> pd.Series:
    cleaned = series.copy()
    for idx, val in cleaned.items():
        if pd.isna(val):
            continue
        text = str(val).strip()
        if text == "":
            continue
        try:
            cleaned.at[idx] = float(text)
        except ValueError:
            cleaned.at[idx] = pd.NA
    return pd.to_numeric(cleaned, errors="coerce")


def load_student_job_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    student_path = RAW_DIR / "student_employability_dataset.csv"
    job_path = RAW_DIR / "job_descriptions.csv"
    if not student_path.exists():
        raise FileNotFoundError(f"Student dataset not found: {student_path}")
    if not job_path.exists():
        raise FileNotFoundError(f"Job dataset not found: {job_path}")

    students = pd.read_csv(student_path)
    jobs = pd.read_csv(job_path)
    students.columns = normalize_column_names(list(students.columns))
    jobs.columns = normalize_column_names(list(jobs.columns))
    return students, jobs


def derive_boolean_features(students: pd.DataFrame) -> pd.DataFrame:
    data = students.copy()
    data["has_internship"] = data.get("has_internship", 0).apply(lambda v: 1 if str(v).strip().lower() in {"1", "yes", "y", "true"} else 0)
    data["active_github"] = data.get("active_github", 0).apply(lambda v: 1 if str(v).strip().lower() in {"1", "yes", "y", "true"} else 0)
    data["is_placed"] = data.get("placement_status", "").apply(lambda v: 1 if str(v).strip().lower() == "placed" else 0)
    data["is_cs_branch"] = data.get("branch", "").apply(lambda v: 1 if str(v).strip().lower() in {"cse", "cs", "computer science", "it"} else 0)
    data["cgpa"] = coerce_numeric(data.get("cgpa", pd.Series([0] * len(data))))
    data["num_projects"] = coerce_numeric(data.get("num_projects", pd.Series([0] * len(data))))
    data["num_certifications"] = coerce_numeric(data.get("num_certifications", pd.Series([0] * len(data))))
    data["hackathons_participated"] = coerce_numeric(data.get("hackathons_participated", pd.Series([0] * len(data))))
    data["backlogs"] = coerce_numeric(data.get("backlogs", pd.Series([0] * len(data))))
    data["communication_rating"] = coerce_numeric(data.get("communication_rating", pd.Series([0] * len(data))))
    return data


def make_data_quality_summary(students: pd.DataFrame) -> dict[str, Any]:
    missing = students.isna().sum().to_dict()
    duplicated = int(students.duplicated().sum())
    cgpa_min = float(students.get("cgpa", pd.Series([0])).min()) if "cgpa" in students.columns else None
    cgpa_max = float(students.get("cgpa", pd.Series([0])).max()) if "cgpa" in students.columns else None
    summary = {
        "shape": students.shape,
        "missing_values": missing,
        "duplicate_rows": duplicated,
        "cgpa_range": [cgpa_min, cgpa_max],
        "placement_counts": students.get("placement_status", pd.Series(["unknown"] * len(students))).value_counts().to_dict(),
        "year_counts": students.get("year", pd.Series(["unknown"] * len(students))).value_counts().to_dict(),
    }
    return summary


def save_eda_plots(students: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if "year" in students.columns:
        plt.figure(figsize=(10, 4))
        students["placement_status"].fillna("Unknown").groupby(students["year"]).value_counts().unstack(fill_value=0).plot(kind="bar")
        plt.title("Placement status by year")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "placement_by_year.png")
        plt.close()

    if "cgpa" in students.columns:
        plt.figure(figsize=(8, 4))
        students["cgpa"].dropna().hist(bins=20)
        plt.title("CGPA distribution")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "cgpa_distribution.png")
        plt.close()

    if "target_role" in students.columns:
        plt.figure(figsize=(10, 4))
        students["target_role"].value_counts().head(10).plot(kind="bar")
        plt.title("Top target roles")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "top_target_roles.png")
        plt.close()

    if "pref_location" in students.columns:
        plt.figure(figsize=(8, 4))
        students["pref_location"].fillna("Unknown").value_counts().head(10).plot(kind="bar")
        plt.title("Preferred work locations")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "preferred_work_locations.png")
        plt.close()


def preprocess_data() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    students, jobs = load_student_job_data()
    students = derive_boolean_features(students)
    for column in ["skills_listed", "skills_text", "project", "internship_detail", "target_role", "lagging", "want_to_learn"]:
        if column in students.columns:
            students[column] = students[column].map(clean_text)
    for column in ["job_description", "required_skills", "good_to_have_skills"]:
        if column in jobs.columns:
            jobs[column] = jobs[column].map(clean_text)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    students.to_csv(PROCESSED_DIR / "students_processed.csv", index=False)
    jobs.to_csv(PROCESSED_DIR / "jobs_processed.csv", index=False)
    summary = make_data_quality_summary(students)
    save_eda_plots(students)
    return students, jobs, summary


if __name__ == "__main__":
    students, jobs, summary = preprocess_data()
    print(json.dumps({"shape": list(summary["shape"]), "duplicate_rows": summary["duplicate_rows"]}, indent=2))
    print("Processed files saved to", PROCESSED_DIR)
