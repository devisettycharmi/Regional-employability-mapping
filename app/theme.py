from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

APP_DIR = Path(__file__).resolve().parent

COLORS = {
    "primary": "#4F46E5",
    "primary_dark": "#3730A3",
    "accent": "#0F766E",
    "success": "#047857",
    "warning": "#B45309",
    "danger": "#B91C1C",
    "bg": "#F8FAFC",
    "surface": "#FFFFFF",
    "border": "#E2E8F0",
    "text": "#0F172A",
    "text_muted": "#64748B",
    "palette": ["#4F46E5", "#0F766E", "#047857", "#B45309", "#B91C1C", "#6D28D9"],
}


def _configure_plotly() -> None:
    pio.templates["employability"] = go.layout.Template(
        layout=go.Layout(
            font=dict(family="Inter, system-ui, sans-serif", color=COLORS["text"]),
            paper_bgcolor=COLORS["surface"],
            plot_bgcolor=COLORS["surface"],
            colorway=COLORS["palette"],
            margin=dict(l=12, r=12, t=48, b=12),
            hoverlabel=dict(
                bgcolor=COLORS["text"],
                bordercolor=COLORS["text"],
                font=dict(color="#FFFFFF", family="Inter, system-ui, sans-serif"),
            ),
            xaxis=dict(showgrid=False, zeroline=False, linecolor=COLORS["border"]),
            yaxis=dict(
                showgrid=True,
                gridcolor="#EEF2F7",
                zeroline=False,
                linecolor=COLORS["border"],
            ),
        )
    )
    pio.templates.default = "employability"


def apply_theme() -> None:
    css = (APP_DIR / "styles.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    _configure_plotly()
