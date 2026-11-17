from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components import badge, badge_group, empty_state, kpi_card, page_header, section_title
from src.config import MODELS_DIR
from src.data_access import read_processed_csv
from src.embeddings import encode_profiles
from src.matching import match_students_to_jobs
from theme import COLORS, apply_theme

apply_theme()


@st.cache_data
def load_students() -> pd.DataFrame:
    return read_processed_csv("students_processed.csv")


@st.cache_data
def load_role_matches() -> pd.DataFrame:
    students_for_match = read_processed_csv("students_processed.csv")
    jobs = read_processed_csv("jobs_processed.csv")
    student_embeddings = MODELS_DIR / "student_embeddings.npy"
    job_embeddings = MODELS_DIR / "job_embeddings.npy"
    if student_embeddings.is_file() and job_embeddings.is_file():
        student_vectors = np.load(student_embeddings)
        job_vectors = np.load(job_embeddings)
    else:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        student_vectors, job_vectors, _ = encode_profiles(
            students_for_match,
            jobs,
            use_sentence_transformer=False,
        )
    if len(students_for_match) != len(student_vectors):
        raise ValueError("Student records and saved student embeddings are not aligned.")
    if len(jobs) != len(job_vectors):
        raise ValueError("Job records and saved job embeddings are not aligned.")
    return match_students_to_jobs(
        students_for_match,
        jobs,
        student_vectors,
        job_vectors,
        top_k=3,
    )


def compute_estimated_score(row: pd.Series) -> float:
    cgpa = float(row.get("cgpa", 0.0) or 0.0)
    coverage = float(row.get("role_coverage", 0.0) or 0.0)
    placement = 1 if str(row.get("placement_status", "")).lower() == "placed" else 0
    score = 0.5 * (cgpa / 10.0) * 100 + 0.35 * coverage * 100 + 0.15 * placement * 100
    return round(min(100.0, max(0.0, score)), 2)


def get_readiness_band(score: float) -> tuple[str, str]:
    if score < 40:
        return "Needs Improvement", "needs-improvement"
    if score < 60:
        return "Developing", "developing"
    if score < 80:
        return "Job Ready", "job-ready"
    return "Highly Employable", "highly-employable"


students = load_students()
if students.empty:
    page_header("Student View", "No student records are available.", "👩‍🎓")
    empty_state("Run the data pipeline to create processed student profiles.")
    st.stop()

page_header(
    "Student View",
    "Explore a saved student profile, role similarities, and the skills to focus on next.",
    "👩‍🎓",
)

left, right = st.columns([0.85, 1.65], gap="large")
with left:
    with st.container(border=True):
        section_title("Choose a profile", "Select an existing student record.")
        student_ids = students["student_id"].astype(str).tolist()
        selected_id = st.selectbox("Student ID", student_ids, label_visibility="collapsed")
        student = students.loc[students["student_id"].astype(str) == selected_id].iloc[0]
        st.markdown(f"**{student.get('name', 'Student')}**")
        st.caption(f"{student.get('college', 'College')} · {student.get('branch', 'Branch')}")
        st.markdown(f"**Target role**  \n{student.get('target_role', 'Not specified')}")
        skill_search = st.text_input("Filter skills", placeholder="Search a skill or gap")
        extra_skill_input = st.text_area(
            "Extra skills to preview",
            placeholder="Optional, comma-separated",
            height=86,
            help="Preview only. This does not change saved scores or role matches.",
        )
        st.caption("Added skills are shown for preview only; saved model results stay unchanged.")

score = compute_estimated_score(student)
band, band_kind = get_readiness_band(score)
role_matches = load_role_matches()
matched_row = role_matches.loc[role_matches["student_id"].astype(str) == selected_id]

with right:
    overview_tab, gap_tab, score_tab = st.tabs(
        ["Overview", "Skill Gap", "Why this score"]
    )

    with overview_tab:
        gauge_col, band_col = st.columns([1.25, 1])
        with gauge_col:
            figure = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=score,
                    number={"suffix": " / 100", "font": {"size": 28}},
                    title={"text": "Employability score"},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1},
                        "bar": {"color": COLORS["primary"], "thickness": 0.28},
                        "bgcolor": "#F1F5F9",
                        "borderwidth": 0,
                        "steps": [
                            {"range": [0, 40], "color": "#FEE2E2"},
                            {"range": [40, 60], "color": "#FEF3C7"},
                            {"range": [60, 80], "color": "#DBEAFE"},
                            {"range": [80, 100], "color": "#D1FAE5"},
                        ],
                    },
                )
            )
            figure.update_layout(height=230, margin=dict(l=20, r=20, t=52, b=8))
            st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
        with band_col:
            kpi_card("CGPA", f"{float(student.get('cgpa', 0) or 0):.2f}/10", icon="🎓")
            st.write("")
            st.markdown("**Readiness band**")
            badge(band, band_kind)
            st.caption(f"Placement status: {student.get('placement_status', 'Unknown')}")

        section_title("Top role similarities", "Ranked using the existing saved profile embeddings.")
        if matched_row.empty:
            empty_state("Role similarity results are unavailable for this student.")
        else:
            roles = matched_row.iloc[0]["top3_roles"]
            scores = matched_row.iloc[0]["top3_scores"]
            role_columns = st.columns(3)
            for index, (role, similarity) in enumerate(zip(roles, scores)):
                percent = max(0.0, min(100.0, float(similarity) * 100))
                with role_columns[index]:
                    with st.container(border=True):
                        st.caption(f"ROLE MATCH {index + 1}")
                        st.markdown(f"**{role}**")
                        st.progress(percent / 100, text=f"{percent:.1f}% similarity")

        section_title("Profile snapshot")
        details = [
            ("College", student.get("college", "Unknown")),
            ("Branch", student.get("branch", "Unknown")),
            ("Year", student.get("year", "Unknown")),
            ("Preferred location", student.get("pref_location", "Unknown")),
        ]
        detail_columns = st.columns(2)
        for index, (label, value) in enumerate(details):
            with detail_columns[index % 2]:
                st.markdown(f"**{label}**  \n{value}")

    with gap_tab:
        section_title(
            "Role-specific skills",
            "Matched, missing, and additional skills from the saved profile.",
        )
        jobs = read_processed_csv("jobs_processed.csv")
        target_role = str(student.get("target_role", "")).strip()
        job_row = jobs.loc[jobs["role"].str.lower() == target_role.lower()]
        required = (
            set(str(job_row.iloc[0].get("required_skills", "")).lower().split(";"))
            if not job_row.empty
            else set()
        )
        student_skills = {
            skill.strip().lower()
            for skill in str(student.get("skills_str", "")).split(";")
            if skill.strip()
        }
        missing_skills = {
            skill.strip().lower()
            for skill in str(student.get("gap_str", "")).split(";")
            if skill.strip()
        }
        matched_skills = student_skills & required
        extra_skills = student_skills - required
        preview_skills = {
            skill.strip()
            for skill in extra_skill_input.replace("\n", ",").split(",")
            if skill.strip()
        }

        if skill_search:
            query = skill_search.casefold()
            matched_skills = {skill for skill in matched_skills if query in skill.casefold()}
            missing_skills = {skill for skill in missing_skills if query in skill.casefold()}
            extra_skills = {skill for skill in extra_skills if query in skill.casefold()}
            preview_skills = {skill for skill in preview_skills if query in skill.casefold()}

        chip_columns = st.columns(3)
        chip_groups = [
            ("Matched", sorted(matched_skills), "matched"),
            ("Missing", sorted(missing_skills), "missing"),
            ("Additional", sorted(extra_skills | preview_skills), "extra"),
        ]
        for column, (title, items, kind) in zip(chip_columns, chip_groups):
            with column:
                st.markdown(f"**{title}**")
                if items:
                    badge_group(items, kind)
                else:
                    st.caption("None identified")

        section_title("Recommended next steps")
        if missing_skills:
            for index, skill in enumerate(sorted(missing_skills)[:5]):
                st.checkbox(
                    f"Practice {skill} with targeted exercises or a mini-project",
                    key=f"skill-plan-{selected_id}-{index}",
                )
        else:
            st.success("No target-role skill gaps were identified in the saved profile.")
        st.caption("Preview-only skills do not modify this saved gap analysis.")

    with score_tab:
        section_title(
            "Transparent score breakdown",
            "This view explains the current dashboard estimate using its existing weighted inputs.",
        )
        cgpa_value = float(student.get("cgpa", 0) or 0)
        coverage_value = float(student.get("role_coverage", 0) or 0)
        placed_value = 1 if str(student.get("placement_status", "")).lower() == "placed" else 0
        contributions = pd.DataFrame(
            {
                "Input": ["CGPA", "Target-role coverage", "Placement status"],
                "Contribution": [
                    0.5 * (cgpa_value / 10) * 100,
                    0.35 * coverage_value * 100,
                    0.15 * placed_value * 100,
                ],
            }
        )
        st.bar_chart(contributions.set_index("Input"), color=COLORS["primary"])
        st.caption(
            "SHAP is not available for this dashboard estimate. The chart shows its direct formula contributions; changing the input text above does not recalculate the score."
        )
