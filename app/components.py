"""
components.py - Reusable Luxury FinTech Components for UPI Fraud Radar.
Stripe Radar / Linear Aesthetics.
"""

import streamlit as st
import numpy as np


def kpi_card(label: str, value: str, card_class: str = "", sub: str = "") -> str:
    """Return HTML for a modern FinTech KPI card."""
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return f"""
    <div class="kpi-card {card_class}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {sub_html}
    </div>
    """


def glass_card(content: str) -> str:
    return f'<div class="glass-card">{content}</div>'


def risk_badge(level: str) -> str:
    cls = {"LOW": "badge-low", "MEDIUM": "badge-medium", "HIGH": "badge-high"}.get(level, "badge-low")
    icon = {"LOW": "●", "MEDIUM": "▲", "HIGH": "✕"}.get(level, "●")
    return f'<span class="badge {cls}"><span style="font-size:0.7em;">{icon}</span> {level}</span>'


def demo_ribbon() -> str:
    return '<div class="demo-ribbon">⚡ DEMO SIMULATION</div>'


def threat_level_badge(recall: float) -> str:
    """Map recall to a threat level badge."""
    if recall >= 0.90:
        lvl, cls, dot = "THREAT LEVEL: CRITICAL", "badge-high", "#F43F5E"
    elif recall >= 0.75:
        lvl, cls, dot = "THREAT LEVEL: ELEVATED", "badge-medium", "#F59E0B"
    else:
        lvl, cls, dot = "THREAT LEVEL: MODERATE", "badge-low", "#10B981"
    return f'<span class="badge {cls}"><span style="width:6px;height:6px;border-radius:50%;background:{dot};display:inline-block;"></span> {lvl}</span>'


def radar_html(fraud_dots: list, safe_dots: list) -> str:
    """
    Generate the animated luxury FinTech radar widget with blip markers.
    fraud_dots and safe_dots are lists of (left%, top%) tuples.
    """
    dots_html = ""
    for x, y in fraud_dots[:12]:
        dots_html += f'<div class="radar-dot fraud" style="left:{x}%;top:{y}%;"></div>'
    for x, y in safe_dots[:20]:
        dots_html += f'<div class="radar-dot safe" style="left:{x}%;top:{y}%;"></div>'

    return f"""
    <div class="radar-container">
        <div class="radar-rings"></div>
        <div class="radar-sweep"></div>
        <div class="radar-crosshair"></div>
        <div class="radar-crosshair-v"></div>
        {dots_html}
    </div>
    """


def gauge_html(probability: float) -> str:
    """
    Modern luxury semicircle risk gauge (0-100%).
    Uses SVG arc with smooth gradient coloring.
    """
    pct = min(100, max(0, probability * 100))
    if probability < 0.3:
        color = "#10B981"
        status_label = "LEGITIMATE / LOW RISK"
    elif probability < 0.7:
        color = "#F59E0B"
        status_label = "ELEVATED RISK / REVIEW"
    else:
        color = "#F43F5E"
        status_label = "CRITICAL / FRAUD DETECTED"

    circumference = 251.2
    dash = circumference * (pct / 100)
    gap = circumference - dash

    return f"""
    <div style="text-align:center; padding:1.2rem 0;">
        <svg width="220" height="125" viewBox="0 0 200 115">
            <!-- Defs for glowing gradient -->
            <defs>
                <linearGradient id="gaugeGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stop-color="#10B981" />
                    <stop offset="50%" stop-color="#F59E0B" />
                    <stop offset="100%" stop-color="#F43F5E" />
                </linearGradient>
            </defs>
            <!-- Background arc -->
            <path d="M 15 100 A 85 85 0 0 1 185 100"
                  fill="none" stroke="#1E293B" stroke-width="14" stroke-linecap="round"/>
            <!-- Value arc -->
            <path d="M 15 100 A 85 85 0 0 1 185 100"
                  fill="none" stroke="{color}" stroke-width="14" stroke-linecap="round"
                  stroke-dasharray="{dash:.1f} {gap:.1f}"
                  style="filter: drop-shadow(0 0 8px {color}88);"/>
            <!-- Center readout -->
            <text x="100" y="85" text-anchor="middle"
                  fill="#F8FAFC" font-family="'JetBrains Mono', monospace"
                  font-size="28" font-weight="700">{pct:.1f}%</text>
            <text x="100" y="105" text-anchor="middle"
                  fill="#94A3B8" font-family="'Plus Jakarta Sans', sans-serif"
                  font-size="9" font-weight="600" letter-spacing="0.08em">PREDICTED RISK PROBABILITY</text>
        </svg>
        <div style="margin-top:6px; font-size:0.82rem; font-weight:700; color:{color}; font-family:'Plus Jakarta Sans', sans-serif; letter-spacing:0.04em;">
            {status_label}
        </div>
    </div>
    """


def pipeline_diagram_html() -> str:
    steps = [
        ("📥", "Raw Ingestion"),
        ("🧹", "Sanitization"),
        ("⚙️", "Feature Engineering"),
        ("🤖", "Ensemble Training"),
        ("📊", "Validation & CI"),
        ("🛡️", "Radar Radar Triage")
    ]
    items = []
    for i, (icon, title) in enumerate(steps):
        items.append(f'<span class="pipeline-step"><span style="margin-right:4px;">{icon}</span>{title}</span>')
        if i < len(steps) - 1:
            items.append('<span class="pipeline-arrow">→</span>')
    return f"""
    <div style="display:flex; align-items:center; flex-wrap:wrap; gap:0.4rem; padding:0.6rem 0;">
        {"".join(items)}
    </div>
    """


def syllabus_table_html() -> str:
    rows = [
        ("Unit I — Data Exploration",
         "Full EDA distributions, PaySim distribution metrics, Outlier IQR/Z-score audits"),
        ("Unit II — Regression",
         "Simple, Multiple & Polynomial Balance-Flow regression engines"),
        ("Unit III — Classification",
         "Ensemble: Random Forest, Decision Tree, Logistic Regression, KNN, SVM, GaussianNB"),
        ("Unit IV — Neural Networks",
         "Multi-Layer Perceptron (MLP) multi-architecture benchmarking & loss convergence"),
        ("Unit V — Unsupervised & Rules",
         "K-Means, K-Medoids, Agglomerative clustering, Apriori frequent fraud itemset mining"),
    ]
    html = """
    <table style="width:100%; border-collapse:collapse; font-size:0.86rem; border:1px solid rgba(255,255,255,0.06); border-radius:8px; overflow:hidden;">
    <thead>
    <tr style="background:rgba(99,102,241,0.08);">
        <th style="border-bottom:1px solid rgba(255,255,255,0.08); padding:0.75rem 1rem; text-align:left;
                   color:#A5B4FC; font-weight:600; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.05em;">Academic Syllabus Module</th>
        <th style="border-bottom:1px solid rgba(255,255,255,0.08); padding:0.75rem 1rem; text-align:left;
                   color:#A5B4FC; font-weight:600; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.05em;">Implemented Capability in Project</th>
    </tr>
    </thead>
    <tbody>
    """
    for unit, coverage in rows:
        html += f"""
        <tr style="border-bottom:1px solid rgba(255,255,255,0.05); transition: background 0.2s;">
            <td style="padding:0.75rem 1rem; color:#F1F5F9; font-weight:600;">{unit}</td>
            <td style="padding:0.75rem 1rem; color:#94A3B8;">{coverage}</td>
        </tr>"""
    html += "</tbody></table>"
    return html
