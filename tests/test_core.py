from __future__ import annotations

import pandas as pd
import pytest

import src.data_access as data_access
from src.data_access import processed_csv_path, read_processed_csv
from src.scoring import compute_employability_score, compute_role_coverage, compute_skill_gap
from src.skill_extraction import build_taxonomy, normalize_skill_token


def test_read_processed_csv_uses_configured_directory(tmp_path, monkeypatch):
    data_file = tmp_path / "students_processed.csv"
    data_file.write_text("student_id\nDEMO-001\n", encoding="utf-8")
    monkeypatch.setenv("EMPLOYABILITY_PROCESSED_DATA_DIR", str(tmp_path))

    students = read_processed_csv("students_processed.csv")

    assert students["student_id"].tolist() == ["DEMO-001"]


def test_configured_processed_data_missing_file_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("EMPLOYABILITY_PROCESSED_DATA_DIR", str(tmp_path))

    with pytest.raises(FileNotFoundError, match="students_processed.csv"):
        processed_csv_path("students_processed.csv")


def test_processed_csv_path_falls_back_to_synthetic_demo(tmp_path, monkeypatch):
    private_dir = tmp_path / "private"
    demo_dir = tmp_path / "data" / "demo" / "processed"
    demo_dir.mkdir(parents=True)
    demo_file = demo_dir / "students_processed.csv"
    demo_file.write_text("student_id\nDEMO-001\n", encoding="utf-8")
    monkeypatch.delenv("EMPLOYABILITY_PROCESSED_DATA_DIR", raising=False)
    monkeypatch.setattr(data_access, "PROCESSED_DIR", private_dir)
    monkeypatch.setattr(data_access, "DATA_DIR", tmp_path / "data")

    assert processed_csv_path("students_processed.csv") == demo_file


def test_normalize_skill_token_handles_aliases():
    assert normalize_skill_token("ML") == "ml"
    assert normalize_skill_token("rest API") == "rest api"


def test_gap_and_coverage():
    student_skills = ["python", "sql", "git"]
    required = ["python", "sql", "django", "flask"]
    assert compute_skill_gap(student_skills, required) == ["django", "flask"]
    assert compute_role_coverage(student_skills, required) == 0.5


def test_employability_score_in_range():
    row = pd.Series({
        "cgpa": 8.5,
        "hybrid_score": 0.7,
        "has_internship": 1,
        "num_projects": 3,
        "num_certifications": 2,
        "hackathons_participated": 1,
        "active_github": 1,
        "communication_rating": 4,
        "backlogs": 0,
        "skill_gap": ["django"],
    })
    score, band, recommendations = compute_employability_score(row)
    assert 0 <= score <= 100
    assert band in {"Needs Improvement", "Developing", "Job Ready", "Highly Employable"}
    assert recommendations


def test_taxonomy_building():
    jobs = pd.DataFrame([
        {"required_skills": "python;sql;django", "good_to_have_skills": "docker;aws"}
    ])
    taxonomy, alias_map = build_taxonomy(jobs)
    assert "python" in taxonomy
    assert "docker" in taxonomy
    assert alias_map.get("ml") == "machine learning"
