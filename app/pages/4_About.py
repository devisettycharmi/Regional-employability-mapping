from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from components import (
    info_card,
    kpi_card,
    page_header,
    section_title,
    tech_badges,
    timeline,
)
from src.config import load_config
from src.data_access import read_processed_csv
from src.skill_extraction import build_taxonomy
from theme import apply_theme

apply_theme()

config = load_config()
project_title = str(config.get("app_title", "Employability Mapper"))
mission = "Helping engineering students in Warangal understand their job readiness and close skill gaps."


@st.cache_data
def load_project_stats() -> tuple[int, int, int, int]:
    students = read_processed_csv("students_processed.csv")
    jobs = read_processed_csv("jobs_processed.csv")
    taxonomy, _ = build_taxonomy(jobs)
    return (
        len(students),
        students["college"].nunique(),
        jobs["role"].nunique(),
        len(taxonomy),
    )


student_count, college_count, role_count, skill_count = load_project_stats()

page_header(project_title, mission, "🎓")
st.page_link(
    "pages/1_Student_View.py",
    label="Try the Student View",
    icon="👩‍🎓",
    width="content",
)

section_title("The problem we address")
problem_columns = st.columns(3, gap="large")
problems = [
    ("Students need clarity", "It can be difficult to know which skills employers expect for a target role.", "🧭"),
    ("Colleges need context", "A regional view can help institutions compare readiness patterns across branches.", "🏫"),
    ("Gaps emerge too late", "Important skill gaps can go unnoticed until placement preparation begins.", "⏳"),
]
for column, problem in zip(problem_columns, problems):
    with column:
        info_card(*problem)

section_title("What you can do here")
feature_columns = st.columns(4, gap="medium")
features = [
    ("Skill Extraction", "Identify skills found in student profiles and project text.", "🧩"),
    ("Role Matching", "Compare profile embeddings with job-role descriptions.", "🎯"),
    ("Employability & Gaps", "Review the existing dashboard score and role-specific skill gaps.", "📊"),
    ("Regional Map", "Explore a Warangal regional centroid and college comparisons.", "🗺️"),
]
for column, feature in zip(feature_columns, features):
    with column:
        info_card(*feature)

section_title("How it works")
timeline(
    [
        ("Data", "Student profiles and job descriptions"),
        ("Skills", "Extracted and normalised skill signals"),
        ("Match & Score", "Compare roles and summarize readiness"),
        ("Insights", "Explore individual and regional patterns"),
    ]
)

section_title("Built with")
tech_badges(
    [
        "Python",
        "spaCy",
        "Sentence-BERT",
        "scikit-learn",
        "XGBoost",
        "SHAP",
        "Folium",
        "Plotly",
        "Streamlit",
    ]
)

section_title("Project at a glance")
kpi_columns = st.columns(4)
kpis = [
    ("Students", f"{student_count:,}", "👥"),
    ("Colleges", f"{college_count:,}", "🏫"),
    ("Roles analysed", f"{role_count:,}", "🎯"),
    ("Skills in taxonomy", f"{skill_count:,}", "🧩"),
]
for column, (label, value, icon) in zip(kpi_columns, kpis):
    with column:
        kpi_card(label, value, icon=icon)

section_title("Data and transparency")
with st.container(border=True):
    st.markdown("**Dataset description**")
    st.write(config.get("data_description", "No data description provided yet."))
    st.markdown("**Limitations**")
    st.markdown(
        "- Results depend on the quality and size of the input data.\n"
        "- Scores are guidance, not guarantees of placement.\n"
        "- The model should not be the sole basis for decisions about students.\n"
        "- Campus-level coordinates and clustering outputs are not present in the current processed data."
    )

section_title("Frequently asked questions")
weights = config.get("weights", {})
weight_labels = [
    ("role_skill_match", "Role / skill match"),
    ("cgpa", "CGPA"),
    ("internship", "Internship"),
    ("projects", "Projects"),
    ("certifications", "Certifications"),
    ("hackathons", "Hackathons"),
    ("github", "GitHub activity"),
    ("communication", "Communication"),
]
configured_weights = ", ".join(
    f"{label} {float(weights.get(key, 0)):.0%}" for key, label in weight_labels
)
backlog_penalty = float(weights.get("backlog_penalty", 0))

with st.expander("How is the employability score calculated?"):
    st.write(
        f"The source scoring pipeline reads these contribution weights from config.yaml: {configured_weights}. "
        f"It also applies a configured backlog penalty of {backlog_penalty:.0%}. "
        "The Student View retains its existing dashboard estimate (50% CGPA, 35% target-role coverage, and 15% placement status); "
        "it is an indicator for exploration, not a guarantee."
    )

with st.expander("What do the readiness bands mean?"):
    st.write(
        "The current Student View labels scores below 40 as Needs Improvement, 40–59 as Developing, "
        "60–79 as Job Ready, and 80 or above as Highly Employable."
    )

with st.expander("How are skill gaps found?"):
    st.write(
        "The processed profile compares the student's extracted skills with required skills for the selected target role. "
        "Required skills that are not found in the profile are presented as gaps."
    )

with st.expander("Can I use my own data?"):
    st.write(
        "Yes. Replace the input CSV files with compatible datasets, then run the preprocessing and pipeline steps. "
        "Review the resulting processed files before using them in the dashboard."
    )

st.markdown(
    f'<footer class="about-footer"><strong>{project_title}</strong> · {datetime.now().year} · Built with Streamlit</footer>',
    unsafe_allow_html=True,
)
