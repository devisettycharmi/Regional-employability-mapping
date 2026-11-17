from __future__ import annotations

from html import escape

import streamlit as st


def page_header(title: str, subtitle: str, icon: str) -> None:
    st.markdown(
        f"""
        <header class="page-header">
            <div class="page-header-icon" aria-hidden="true">{escape(icon)}</div>
            <div>
                <h1>{escape(title)}</h1>
                <p>{escape(subtitle)}</p>
            </div>
        </header>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, delta: str | None = None, icon: str | None = None) -> None:
    delta_html = f'<span class="kpi-delta">{escape(delta)}</span>' if delta else ""
    icon_html = f'<span class="kpi-icon" aria-hidden="true">{escape(icon)}</span>' if icon else ""
    st.markdown(
        f"""
        <section class="kpi-card" aria-label="{escape(label)}: {escape(value)}">
            <div class="kpi-top"><span class="kpi-label">{escape(label)}</span>{icon_html}</div>
            <div class="kpi-value">{escape(value)}</div>
            {delta_html}
        </section>
        """,
        unsafe_allow_html=True,
    )


def info_card(title: str, body: str, icon: str) -> None:
    st.markdown(
        f"""
        <article class="info-card">
            <div class="info-icon" aria-hidden="true">{escape(icon)}</div>
            <h3>{escape(title)}</h3>
            <p>{escape(body)}</p>
        </article>
        """,
        unsafe_allow_html=True,
    )


def badge(text: str, kind: str) -> None:
    normalized_kind = kind.strip().lower().replace(" ", "-")
    allowed_kinds = {
        "needs-improvement",
        "developing",
        "job-ready",
        "highly-employable",
        "matched",
        "missing",
        "extra",
    }
    css_kind = normalized_kind if normalized_kind in allowed_kinds else "extra"
    st.markdown(
        f'<span class="badge badge-{css_kind}">{escape(text)}</span>',
        unsafe_allow_html=True,
    )


def badge_group(items: list[str], kind: str) -> None:
    normalized_kind = kind.strip().lower().replace(" ", "-")
    allowed_kinds = {
        "needs-improvement",
        "developing",
        "job-ready",
        "highly-employable",
        "matched",
        "missing",
        "extra",
    }
    css_kind = normalized_kind if normalized_kind in allowed_kinds else "extra"
    badges = "".join(
        f'<span class="badge badge-{css_kind}">{escape(item)}</span>' for item in items
    )
    st.markdown(f'<div class="badge-group">{badges}</div>', unsafe_allow_html=True)


def section_title(text: str, subtitle: str | None = None) -> None:
    subtitle_html = f"<p>{escape(subtitle)}</p>" if subtitle else ""
    st.markdown(
        f'<div class="section-title"><h2>{escape(text)}</h2>{subtitle_html}</div>',
        unsafe_allow_html=True,
    )


def empty_state(message: str) -> None:
    st.markdown(
        f'<div class="empty-state" role="status">{escape(message)}</div>',
        unsafe_allow_html=True,
    )


def sidebar_brand() -> None:
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-logo" aria-hidden="true">EM</div>
            <div>
                <div class="sidebar-name">Employability Mapper</div>
                <div class="sidebar-tagline">Warangal Engineering Students</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_footer() -> None:
    st.sidebar.markdown(
        """
        <div class="sidebar-footer">
            <strong>Regional Employability Mapping</strong>
            <div class="sidebar-team">Synthetic demo · not real student data</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def timeline_step(number: int, title: str, body: str) -> None:
    st.markdown(
        f"""
        <article class="timeline-step">
            <div class="timeline-number" aria-label="Step {number}">{number}</div>
            <strong>{escape(title)}</strong>
            <span>{escape(body)}</span>
        </article>
        """,
        unsafe_allow_html=True,
    )


def timeline(steps: list[tuple[str, str]]) -> None:
    items = "".join(
        f'<article class="timeline-step"><div class="timeline-number" aria-label="Step {index}">'
        f"{index}</div><strong>{escape(title)}</strong><span>{escape(body)}</span></article>"
        for index, (title, body) in enumerate(steps, start=1)
    )
    st.markdown(f'<div class="timeline">{items}</div>', unsafe_allow_html=True)


def tech_badges(items: list[str]) -> None:
    badges = "".join(f'<span class="tech-badge">{escape(item)}</span>' for item in items)
    st.markdown(f'<div class="tech-badges">{badges}</div>', unsafe_allow_html=True)
