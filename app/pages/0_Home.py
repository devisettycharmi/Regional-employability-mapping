from __future__ import annotations

import pandas as pd
import streamlit as st

from components import info_card, kpi_card, page_header, section_title
from src.data_access import read_processed_csv
from theme import apply_theme

apply_theme()


@st.cache_data
def load_summary_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    students = read_processed_csv("students_processed.csv")
    jobs = read_processed_csv("jobs_processed.csv")
    return students, jobs


students, jobs = load_summary_data()
if students.empty:
    st.warning("No processed student data is available yet.")
    st.stop()

estimated_scores = (
    0.5 * (students["cgpa"].fillna(0).astype(float) / 10.0) * 100
    + 0.35 * students["role_coverage"].fillna(0).astype(float) * 100
    + 0.15 * students["is_placed"].fillna(0).astype(float) * 100
).clip(0, 100)

page_header(
    "Understand readiness. Close skill gaps.",
    "A regional view of student skills, role alignment, and placement readiness for engineering talent in Warangal.",
    "🎓",
)

action_left, action_right = st.columns([1, 1])
with action_left:
    st.page_link(
        "pages/1_Student_View.py",
        label="Check my employability",
        icon="👩‍🎓",
        width="stretch",
    )
with action_right:
    st.page_link(
        "pages/2_Regional_Dashboard.py",
        label="Explore the map",
        icon="📍",
        width="stretch",
    )

section_title("Project at a glance", "A quick summary of the processed dataset and role catalogue.")
kpi_columns = st.columns(4)
kpi_values = [
    ("Students analysed", f"{len(students):,}", "👥"),
    ("Average employability", f"{estimated_scores.mean():.1f}/100", "✨"),
    ("Placement rate", f"{students['is_placed'].mean() * 100:.1f}%", "📈"),
    ("Roles covered", f"{jobs['role'].nunique():,}", "🧭"),
]
for column, (label, value, icon) in zip(kpi_columns, kpi_values):
    with column:
        kpi_card(label, value, icon=icon)

section_title("What you can explore", "From individual profiles to region-wide patterns.")
feature_columns = st.columns(3)
features = [
    (
        "Skill Extraction",
        "See the technical skills identified in student profiles and the capabilities frequently missing.",
        "🧩",
    ),
    (
        "Role Matching",
        "Compare student profiles with job-role descriptions and review the strongest role similarities.",
        "🎯",
    ),
    (
        "Regional Insights",
        "Explore college and branch-level patterns to understand readiness across the region.",
        "🗺️",
    ),
]
for column, feature in zip(feature_columns, features):
    with column:
        info_card(*feature)

st.caption(
    "Employability scores are an analytical guide based on available profile data, not a guarantee of placement."
)
