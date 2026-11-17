from __future__ import annotations

import folium
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

from components import empty_state, kpi_card, page_header, section_title
from src.data_access import read_processed_csv
from src.geo import MANUAL_GEO_OVERRIDES
from theme import COLORS, apply_theme

apply_theme()


@st.cache_data
def load_students() -> pd.DataFrame:
    df = read_processed_csv("students_processed.csv")
    df["estimated_score"] = (
        0.5 * (df["cgpa"].fillna(0).astype(float) / 10.0) * 100
        + 0.35 * df["role_coverage"].fillna(0).astype(float) * 100
        + 0.15 * df["is_placed"].fillna(0).astype(float) * 100
    ).clip(0, 100)
    return df


students = load_students()
if students.empty:
    page_header("Regional Dashboard", "Regional comparisons are not available.", "📍")
    empty_state("No processed student data is available yet.")
    st.stop()

page_header(
    "Regional Dashboard",
    "Compare student readiness, placement outcomes, and branch representation across colleges.",
    "📍",
)

college_options = sorted(students["college"].dropna().astype(str).unique().tolist())
branch_options = sorted(students["branch"].dropna().astype(str).unique().tolist())
year_options = sorted(students["year"].dropna().astype(str).unique().tolist())

with st.container(border=True):
    section_title("Filter the region", "Filters update the map, metrics, and college comparison.")
    college_col, branch_col, year_col = st.columns([1.5, 1, 1])
    with college_col:
        selected_colleges = st.multiselect(
            "College",
            college_options,
            default=college_options,
            placeholder="Select one or more colleges",
        )
    with branch_col:
        selected_branch = st.selectbox("Branch", ["All"] + branch_options)
    with year_col:
        selected_year = st.selectbox("Year", ["All"] + year_options)

filtered = students[students["college"].isin(selected_colleges)].copy()
if selected_branch != "All":
    filtered = filtered[filtered["branch"] == selected_branch]
if selected_year != "All":
    filtered = filtered[filtered["year"].astype(str) == selected_year]

if filtered.empty:
    empty_state("No students match these filters. Try another college, branch, or year.")
    st.stop()

summary = filtered.groupby("college", as_index=False).agg(
    student_count=("student_id", "count"),
    avg_score=("estimated_score", "mean"),
    avg_cgpa=("cgpa", "mean"),
    placement_rate=("is_placed", "mean"),
)

kpi_columns = st.columns(4)
kpis = [
    ("Students", f"{len(filtered):,}", "👥"),
    ("Average score", f"{filtered['estimated_score'].mean():.1f}/100", "✨"),
    ("Average CGPA", f"{filtered['cgpa'].mean():.2f}/10", "🎓"),
    ("Placement rate", f"{filtered['is_placed'].mean() * 100:.1f}%", "📈"),
]
for column, (label, value, icon) in zip(kpi_columns, kpis):
    with column:
        kpi_card(label, value, icon=icon)

section_title("Regional view", "The map marks the regional study area; campus-level coordinates are not present in the dataset.")
map_col, chart_col = st.columns([1.05, 1], gap="large")
with map_col:
    with st.container(border=True):
        st.markdown("**Warangal study area**")
        center = MANUAL_GEO_OVERRIDES["Warangal"]
        regional_map = folium.Map(
            location=center,
            zoom_start=11,
            tiles="OpenStreetMap",
            name="OpenStreetMap",
            max_zoom=20,
            control_scale=True,
        )
        radius = min(30, max(10, 10 + len(filtered) / 70))
        folium.CircleMarker(
            location=center,
            radius=radius,
            color=COLORS["primary"],
            weight=2,
            fill=True,
            fill_color=COLORS["accent"],
            fill_opacity=0.72,
            tooltip=f"Warangal region · {len(filtered):,} students in current selection",
        ).add_to(regional_map)
        folium.Marker(
            location=center,
            tooltip="Warangal regional centroid",
            icon=folium.Icon(color="blue", icon="info-sign"),
        ).add_to(regional_map)
        components.html(regional_map._repr_html_(), height=520, scrolling=False)
        st.caption(
            "Legend: the marker is the Warangal regional centroid; its circle size reflects the filtered student count."
        )

with chart_col:
    with st.container(border=True):
        section_title("College comparison", "Average estimated employability score within the current selection.")
        chart = px.bar(
            summary.sort_values("avg_score"),
            x="avg_score",
            y="college",
            orientation="h",
            color_discrete_sequence=[COLORS["primary"]],
            labels={"avg_score": "Average score", "college": ""},
            text_auto=".1f",
        )
        chart.update_layout(
            template="employability",
            font_family="Inter",
            margin=dict(l=10, r=10, t=18, b=10),
            showlegend=False,
            xaxis=dict(range=[0, 100], showgrid=True, gridcolor="#EEF2F7"),
            yaxis=dict(showgrid=False),
            height=440,
        )
        st.plotly_chart(chart, width="stretch", config={"displayModeBar": False})

section_title("College details")
table = summary.sort_values("avg_score", ascending=False)
st.dataframe(
    table,
    width="stretch",
    hide_index=True,
    column_config={
        "college": st.column_config.TextColumn("College", width="large"),
        "student_count": st.column_config.NumberColumn("Students", format="%d", width="small"),
        "avg_score": st.column_config.ProgressColumn(
            "Average score", min_value=0, max_value=100, format="%.1f", width="medium"
        ),
        "avg_cgpa": st.column_config.NumberColumn("Avg CGPA", format="%.2f", width="small"),
        "placement_rate": st.column_config.NumberColumn(
            "Placement rate", format="%.1f%%", width="medium"
        ),
    },
)
