from __future__ import annotations

import streamlit as st


def apply_dashboard_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #14213d;
            --muted: #64748b;
            --line: rgba(148, 163, 184, 0.2);
            --surface: rgba(255, 255, 255, 0.88);
            --accent: #4f46e5;
        }
        .stApp {
            background:
                radial-gradient(ellipse at 92% 0%, rgba(129, 140, 248, 0.13), transparent 34rem),
                linear-gradient(135deg, #f8fafc 0%, #f1f5ff 58%, #effcff 100%);
        }
        .block-container { max-width: 1500px; padding-top: 1.65rem; padding-bottom: 3rem; }
        h1, h2, h3 { color: var(--ink); letter-spacing: -0.025em; }
        h1 { font-weight: 750; }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #111b35 0%, #172554 62%, #312e81 100%);
        }
        [data-testid="stSidebar"] * { color: #e2e8f0; }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #b8c5dd; }
        [data-testid="stSidebar"] a { border-radius: 0.7rem; }
        [data-testid="stSidebar"] a:hover { background: rgba(255, 255, 255, 0.12); }
        [data-testid="stMetric"] {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 1rem;
            padding: 1rem 1.1rem;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.055);
            transition: transform 160ms ease, box-shadow 160ms ease;
        }
        [data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 14px 28px rgba(15, 23, 42, 0.09);
        }
        [data-testid="stMetricLabel"] { color: var(--muted); font-weight: 600; }
        [data-testid="stMetricValue"] { color: var(--ink); font-weight: 750; }
        [data-testid="stPlotlyChart"], [data-testid="stDataFrame"] {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 1rem;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.045);
            overflow: hidden;
        }
        [data-testid="stSelectbox"], [data-testid="stMultiSelect"] {
            background: rgba(255, 255, 255, 0.7);
            border-radius: 0.8rem;
        }
        [data-testid="stButton"] button {
            border-radius: 0.75rem;
            font-weight: 650;
            transition: transform 160ms ease, box-shadow 160ms ease;
        }
        [data-testid="stButton"] button:hover {
            transform: translateY(-1px);
            box-shadow: 0 8px 18px rgba(79, 70, 229, 0.16);
        }
        .card {
            padding: 1rem 1.15rem;
            border: 1px solid var(--line);
            border-radius: 1rem;
            background: var(--surface);
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.045);
        }
        .pill {
            display: inline-block;
            margin: 0.22rem 0.3rem 0.22rem 0;
            padding: 0.42rem 0.75rem;
            border: 1px solid rgba(79, 70, 229, 0.1);
            border-radius: 999px;
            background: #eef2ff;
            color: #312e81;
            font-size: 0.82rem;
            font-weight: 650;
        }
        .badge, .warning, .info, .danger {
            display: inline-block;
            padding: 0.48rem 0.82rem;
            border-radius: 999px;
            font-size: 0.86rem;
            font-weight: 750;
        }
        .badge { background: #dcfce7; color: #166534; }
        .warning { background: #fef3c7; color: #92400e; }
        .info { background: #dbeafe; color: #1e40af; }
        .danger { background: #fee2e2; color: #991b1b; }
        .dashboard-hero {
            position: relative;
            overflow: hidden;
            padding: 1.55rem 1.7rem;
            margin: 0.1rem 0 1.35rem;
            border: 1px solid rgba(255, 255, 255, 0.24);
            border-radius: 1.25rem;
            background: linear-gradient(115deg, #111b35 0%, #253b82 55%, #4f46e5 100%);
            box-shadow: 0 18px 38px rgba(30, 41, 90, 0.19);
        }
        .dashboard-hero:after {
            content: "";
            position: absolute;
            width: 18rem;
            height: 18rem;
            top: -11rem;
            right: -3rem;
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 50%;
            box-shadow: 0 0 0 2rem rgba(255, 255, 255, 0.04), 0 0 0 4rem rgba(255, 255, 255, 0.035);
        }
        .dashboard-kicker {
            margin-bottom: 0.35rem;
            color: #c7d2fe;
            font-size: 0.74rem;
            font-weight: 750;
            letter-spacing: 0.14em;
            text-transform: uppercase;
        }
        .dashboard-title {
            margin: 0;
            color: #fff;
            font-size: clamp(1.7rem, 3vw, 2.45rem);
            font-weight: 780;
            letter-spacing: -0.04em;
            line-height: 1.15;
        }
        .dashboard-subtitle {
            max-width: 54rem;
            margin-top: 0.55rem;
            color: #dbeafe;
            font-size: 0.98rem;
        }
        .section-caption {
            margin: -0.35rem 0 0.85rem;
            color: var(--muted);
            font-size: 0.92rem;
        }
        @media (max-width: 700px) {
            .block-container { padding-top: 1rem; }
            .dashboard-hero { padding: 1.2rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(title: str, subtitle: str, kicker: str = "Employability insights") -> None:
    st.markdown(
        f"""
        <div class="dashboard-hero">
            <div class="dashboard-kicker">{kicker}</div>
            <div class="dashboard-title">{title}</div>
            <div class="dashboard-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
