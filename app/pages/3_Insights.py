from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from components import empty_state, page_header, section_title
from src.data_access import read_processed_csv
from theme import COLORS, apply_theme

apply_theme()


@st.cache_data
def load_insight_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    students = read_processed_csv("students_processed.csv")
    jobs = read_processed_csv("jobs_processed.csv")
    return students, jobs


students, jobs = load_insight_data()
page_header(
    "Insights",
    "Browse the skills, roles, and branch patterns shaping student readiness.",
    "📊",
)

if students.empty:
    empty_state("No processed student data is available yet.")
    st.stop()

skill_supply: dict[str, int] = {}
gap_supply: dict[str, int] = {}
for value in students["skills_str"].fillna("").astype(str):
    for skill in {item.strip().lower() for item in value.split(";") if item.strip()}:
        skill_supply[skill] = skill_supply.get(skill, 0) + 1
for value in students["gap_str"].fillna("").astype(str):
    for skill in {item.strip().lower() for item in value.split(";") if item.strip()}:
        gap_supply[skill] = gap_supply.get(skill, 0) + 1

skills_tab, roles_tab, branches_tab, clusters_tab = st.tabs(
    ["Skills", "Roles", "Branches", "Clusters"]
)

with skills_tab:
    section_title("Skill supply and gaps", "Counts represent student profiles containing each skill.")
    supply = pd.DataFrame(
        sorted(skill_supply.items(), key=lambda item: item[1], reverse=True)[:12],
        columns=["skill", "students"],
    )
    gaps = pd.DataFrame(
        sorted(gap_supply.items(), key=lambda item: item[1], reverse=True)[:12],
        columns=["skill", "profiles_with_gap"],
    )
    supply_col, gap_col = st.columns(2, gap="large")
    with supply_col:
        st.markdown("**Most common student skills**")
        if supply.empty:
            empty_state("No extracted skills are available.")
        else:
            chart = px.bar(
                supply.sort_values("students"),
                x="students",
                y="skill",
                orientation="h",
                color_discrete_sequence=[COLORS["primary"]],
                labels={"students": "Students", "skill": ""},
                text="students",
            )
            chart.update_layout(template="employability", margin=dict(l=8, r=8, t=16, b=8))
            st.plotly_chart(chart, width="stretch", config={"displayModeBar": False})
            st.caption("Python, SQL, and Git are among the most frequently listed skills.")
    with gap_col:
        st.markdown("**Most frequent skill gaps**")
        if gaps.empty:
            empty_state("No skill-gap records are available.")
        else:
            chart = px.bar(
                gaps.sort_values("profiles_with_gap"),
                x="profiles_with_gap",
                y="skill",
                orientation="h",
                color_discrete_sequence=[COLORS["danger"]],
                labels={"profiles_with_gap": "Profiles with gap", "skill": ""},
                text="profiles_with_gap",
            )
            chart.update_layout(template="employability", margin=dict(l=8, r=8, t=16, b=8))
            st.plotly_chart(chart, width="stretch", config={"displayModeBar": False})
            st.caption("Counts highlight where targeted practice may help the most students.")

with roles_tab:
    section_title("Role landscape", "Compare student target-role preferences with the job catalogue.")
    target_roles = (
        students["target_role"]
        .fillna("Not specified")
        .astype(str)
        .value_counts()
        .head(15)
        .rename_axis("role")
        .reset_index(name="students")
    )
    role_col, catalogue_col = st.columns([1.3, 1], gap="large")
    with role_col:
        chart = px.bar(
            target_roles.sort_values("students"),
            x="students",
            y="role",
            orientation="h",
            color_discrete_sequence=[COLORS["accent"]],
            labels={"students": "Students", "role": ""},
            text="students",
        )
        chart.update_layout(template="employability", margin=dict(l=8, r=8, t=16, b=8))
        st.plotly_chart(chart, width="stretch", config={"displayModeBar": False})
        st.caption("Target roles show student interest, not confirmed placements.")
    with catalogue_col:
        st.markdown("**Roles in the job catalogue**")
        st.dataframe(
            jobs[["role", "domain", "experience_level"]]
            .drop_duplicates()
            .sort_values(["domain", "role"]),
            width="stretch",
            hide_index=True,
        )
        st.caption(f"The processed job catalogue contains {jobs['role'].nunique()} distinct roles.")

with branches_tab:
    section_title("Branch comparison", "Academic and placement summaries are based on current student records.")
    branch_summary = students.groupby("branch", as_index=False).agg(
        average_cgpa=("cgpa", "mean"),
        student_count=("student_id", "count"),
        placement_rate=("is_placed", "mean"),
    )
    chart = px.bar(
        branch_summary.sort_values("average_cgpa"),
        x="average_cgpa",
        y="branch",
        orientation="h",
        color_discrete_sequence=[COLORS["primary"]],
        labels={"average_cgpa": "Average CGPA", "branch": ""},
        hover_data={"student_count": True, "placement_rate": ":.1%"},
        text_auto=".2f",
    )
    chart.update_layout(
        template="employability",
        margin=dict(l=8, r=8, t=16, b=8),
        xaxis=dict(range=[0, 10], showgrid=True, gridcolor="#EEF2F7"),
    )
    st.plotly_chart(chart, width="stretch", config={"displayModeBar": False})
    st.caption("Hover over a branch to compare its student count and placement rate.")

with clusters_tab:
    section_title("Student clusters", "Cluster labels are shown when clustering outputs are generated.")
    empty_state(
        "No clustering results are present in the current processed data. This view will populate when cluster outputs are added."
    )
