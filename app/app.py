from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
PAGES_DIR = APP_DIR / "pages"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from components import sidebar_brand, sidebar_footer
from theme import apply_theme

st.set_page_config(
    page_title="Employability Mapper",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme()

home_page = st.Page(
    str(PAGES_DIR / "0_Home.py"),
    title="Home",
    icon="🏠",
    default=True,
)
student_page = st.Page(
    str(PAGES_DIR / "1_Student_View.py"),
    title="Student View",
    icon="👩‍🎓",
)
regional_page = st.Page(
    str(PAGES_DIR / "2_Regional_Dashboard.py"),
    title="Regional Dashboard",
    icon="📍",
)
insights_page = st.Page(
    str(PAGES_DIR / "3_Insights.py"),
    title="Insights",
    icon="📊",
)
about_page = st.Page(
    str(PAGES_DIR / "4_About.py"),
    title="About",
    icon="ℹ️",
)

navigation = st.navigation(
    [home_page, student_page, regional_page, insights_page, about_page],
    position="hidden",
)

sidebar_brand()
st.sidebar.markdown('<div class="sidebar-section">Explore</div>', unsafe_allow_html=True)
st.sidebar.page_link(home_page, label="Home", icon="🏠")
st.sidebar.page_link(student_page, label="Student View", icon="👩‍🎓")
st.sidebar.page_link(regional_page, label="Regional Dashboard", icon="📍")
st.sidebar.page_link(insights_page, label="Insights", icon="📊")
st.sidebar.markdown('<div class="sidebar-section">Project</div>', unsafe_allow_html=True)
st.sidebar.page_link(about_page, label="About", icon="ℹ️")
sidebar_footer()
navigation.run()
