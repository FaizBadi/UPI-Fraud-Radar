"""
style.py - Modern Luxury FinTech Dark Theme (Stripe Radar / Linear Style)
UPI Fraud Radar Design System
"""

GOOGLE_FONTS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
"""

CSS = """
<style>
/* ── Global Reset & Luxury Base ────────────────────────────────────────── */
*, *::before, *::after {
    box-sizing: border-box;
}

html, body, [data-testid="stAppViewContainer"] {
    background: radial-gradient(120% 120% at 50% -10%, #151F38 0%, #0D1527 35%, #080D19 75%, #050811 100%) !important;
    background-attachment: fixed !important;
    color: #E2E8F0 !important;
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    -webkit-font-smoothing: antialiased;
}

/* Subtle micro-dot grid */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    pointer-events: none;
    background-image: radial-gradient(rgba(99, 102, 241, 0.08) 1px, transparent 1px);
    background-size: 28px 28px;
    z-index: 0;
}

.block-container {
    padding-top: 2rem !important;
    padding-bottom: 3.5rem !important;
    max-width: 1280px !important;
    position: relative;
    z-index: 1;
}

/* Header configuration: Keep header transparent, NEVER hide expand/collapse controls */
header[data-testid="stHeader"] {
    background: transparent !important;
    height: 3.5rem !important;
    z-index: 99990 !important;
}

/* Allow stToolbar to render the stExpandSidebarButton */
[data-testid="stToolbar"] {
    background: transparent !important;
    display: flex !important;
    visibility: visible !important;
    right: auto !important;
    left: 14px !important;
    top: 10px !important;
    z-index: 99995 !important;
}

/* Hide deploy button and developer menu from toolbar */
[data-testid="stToolbarActions"],
[data-testid="stHeaderActionElements"],
[data-testid="stStatusWidget"],
#MainMenu,
footer,
[data-testid="stDecoration"] {
    display: none !important;
    visibility: hidden !important;
}

/* ── Floating Action Bar Reopen / Expand Button (Stripe Luxury Pill) ── */
[data-testid="stExpandSidebarButton"],
button[data-testid="stExpandSidebarButton"],
[data-testid="collapsedControl"] {
    display: inline-flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    position: fixed !important;
    top: 14px !important;
    left: 16px !important;
    z-index: 9999999 !important;
    background: linear-gradient(135deg, rgba(20, 29, 52, 0.96) 0%, rgba(13, 20, 37, 0.98) 100%) !important;
    border: 1.5px solid rgba(99, 102, 241, 0.6) !important;
    border-radius: 12px !important;
    padding: 8px 16px !important;
    box-shadow: 0 8px 25px -4px rgba(0, 0, 0, 0.8), 0 0 20px rgba(99, 102, 241, 0.45) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    cursor: pointer !important;
    min-height: 42px !important;
    align-items: center !important;
    justify-content: center !important;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

[data-testid="stExpandSidebarButton"]:hover,
button[data-testid="stExpandSidebarButton"]:hover,
[data-testid="collapsedControl"]:hover {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.35) 0%, rgba(139, 92, 246, 0.3) 100%) !important;
    border-color: #818CF8 !important;
    transform: scale(1.08) translateX(2px) !important;
    box-shadow: 0 12px 30px -4px rgba(0, 0, 0, 0.9), 0 0 28px rgba(99, 102, 241, 0.7) !important;
}

[data-testid="stExpandSidebarButton"] svg,
button[data-testid="stExpandSidebarButton"] svg,
[data-testid="collapsedControl"] svg {
    fill: #A5B4FC !important;
    stroke: #A5B4FC !important;
    width: 22px !important;
    height: 22px !important;
    transition: transform 0.2s ease !important;
}

[data-testid="stExpandSidebarButton"]:hover svg,
button[data-testid="stExpandSidebarButton"]:hover svg,
[data-testid="collapsedControl"]:hover svg {
    fill: #FFFFFF !important;
    stroke: #FFFFFF !important;
    transform: translateX(3px) !important;
}

/* ── Custom Scrollbars ─────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 7px; height: 7px; }
::-webkit-scrollbar-track { background: #0A0F1D; }
::-webkit-scrollbar-thumb {
    background: rgba(99, 102, 241, 0.25);
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(99, 102, 241, 0.45);
}

/* ── Typography & Headings ─────────────────────────────────────────────── */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: -0.025em !important;
    color: #F8FAFC !important;
}

h1 {
    font-size: 2.15rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em !important;
    background: linear-gradient(135deg, #FFFFFF 0%, #E2E8F0 50%, #A5B4FC 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem !important;
}

h2 {
    font-size: 1.45rem !important;
    font-weight: 700 !important;
    color: #F1F5F9 !important;
    margin-top: 1.2rem !important;
    margin-bottom: 0.6rem !important;
}

h3 {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    color: #CBD5E1 !important;
}

p, span, label, div {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

code, pre, .mono {
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Sidebar Redesign (Modern FinTech Executive Style) ────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0C1222 0%, #070B16 100%) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    box-shadow: 6px 0 32px rgba(0, 0, 0, 0.5) !important;
}

/* Sidebar chevron / collapse button inside open sidebar */
[data-testid="stSidebarCollapseButton"] button,
button[aria-label="Close sidebar"] {
    background: rgba(20, 29, 52, 0.7) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 9px !important;
    color: #94A3B8 !important;
    transition: all 0.2s ease !important;
}
[data-testid="stSidebarCollapseButton"] button:hover,
button[aria-label="Close sidebar"]:hover {
    background: rgba(99, 102, 241, 0.2) !important;
    border-color: rgba(99, 102, 241, 0.5) !important;
    color: #F8FAFC !important;
    box-shadow: 0 0 12px rgba(99, 102, 241, 0.3) !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #94A3B8;
}

[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] {
    gap: 0.45rem !important;
}

[data-testid="stSidebar"] .stRadio label {
    background: rgba(20, 29, 52, 0.5) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 11px !important;
    padding: 0.65rem 0.95rem !important;
    color: #94A3B8 !important;
    font-weight: 500 !important;
    font-size: 0.92rem !important;
    transition: all 0.22s cubic-bezier(0.16, 1, 0.3, 1) !important;
    cursor: pointer !important;
    display: flex !important;
    align-items: center !important;
}

[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(99, 102, 241, 0.12) !important;
    border-color: rgba(99, 102, 241, 0.35) !important;
    color: #F8FAFC !important;
    transform: translateX(4px) !important;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25), 0 0 12px rgba(99, 102, 241, 0.15) !important;
}

/* Active Selected Tab in Sidebar */
[data-testid="stSidebar"] .stRadio label:has(input:checked) {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.22) 0%, rgba(139, 92, 246, 0.16) 100%) !important;
    border: 1px solid rgba(99, 102, 241, 0.55) !important;
    border-left: 3.5px solid #6366F1 !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.35), 0 0 16px rgba(99, 102, 241, 0.25) !important;
}

/* ── Glassmorphic Cards ───────────────────────────────────────────────── */
.glass-card {
    background: rgba(15, 23, 42, 0.65);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    backdrop-filter: blur(20px) saturate(180%);
    -webkit-backdrop-filter: blur(20px) saturate(180%);
    padding: 1.35rem 1.6rem;
    margin-bottom: 1.25rem;
    position: relative;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.08);
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
}

.glass-card:hover {
    border-color: rgba(99, 102, 241, 0.3);
    box-shadow: 0 16px 36px -10px rgba(0, 0, 0, 0.6), 0 0 24px -4px rgba(99, 102, 241, 0.12), inset 0 1px 0 rgba(255, 255, 255, 0.12);
    transform: translateY(-2px);
}

/* ── KPI Metric Cards ─────────────────────────────────────────────────── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1.1rem;
    margin-bottom: 1.5rem;
}

.kpi-card {
    background: linear-gradient(145deg, rgba(20, 29, 52, 0.7) 0%, rgba(13, 20, 37, 0.85) 100%);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    backdrop-filter: blur(16px);
    box-shadow: 0 8px 24px -8px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.06);
    position: relative;
    overflow: hidden;
    transition: all 0.25s ease;
}

.kpi-card::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #6366F1, #8B5CF6);
    opacity: 0.8;
}

.kpi-card.fraud::before {
    background: linear-gradient(90deg, #F43F5E, #FB7185);
}

.kpi-card.warn::before {
    background: linear-gradient(90deg, #F59E0B, #FBBF24);
}

.kpi-card.safe::before {
    background: linear-gradient(90deg, #10B981, #34D399);
}

.kpi-card:hover {
    border-color: rgba(99, 102, 241, 0.35);
    transform: translateY(-2px);
    box-shadow: 0 14px 32px -8px rgba(0, 0, 0, 0.6), 0 0 20px rgba(99, 102, 241, 0.1);
}

.kpi-label {
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #94A3B8;
    margin-bottom: 0.4rem;
}

.kpi-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.95rem;
    font-weight: 700;
    color: #F8FAFC;
    letter-spacing: -0.02em;
    line-height: 1.15;
}

.kpi-card.fraud .kpi-value {
    color: #FB7185;
}

.kpi-card.warn .kpi-value {
    color: #FBBF24;
}

.kpi-card.safe .kpi-value {
    color: #34D399;
}

.kpi-sub {
    font-size: 0.74rem;
    color: #64748B;
    margin-top: 0.4rem;
    display: flex;
    align-items: center;
    gap: 0.35rem;
}

/* ── Badges & Status Pills ────────────────────────────────────────────── */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.35rem 0.85rem;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    font-family: 'JetBrains Mono', monospace;
}

.badge-low, .badge-safe {
    background: rgba(16, 185, 129, 0.12);
    color: #34D399;
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.badge-medium, .badge-warn {
    background: rgba(245, 158, 11, 0.12);
    color: #FBBF24;
    border: 1px solid rgba(245, 158, 11, 0.3);
}

.badge-high, .badge-fraud {
    background: rgba(244, 63, 94, 0.14);
    color: #FB7185;
    border: 1px solid rgba(244, 63, 94, 0.35);
    box-shadow: 0 0 16px rgba(244, 63, 94, 0.2);
}

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    background: rgba(99, 102, 241, 0.1);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 9999px;
    padding: 0.3rem 0.85rem;
    font-size: 0.74rem;
    font-weight: 600;
    color: #A5B4FC;
    letter-spacing: 0.05em;
    margin-bottom: 0.8rem;
    font-family: 'JetBrains Mono', monospace;
}

.pulse-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #6366F1;
    box-shadow: 0 0 8px #6366F1;
    animation: pulse-glow 2s infinite ease-in-out;
}

@keyframes pulse-glow {
    0%, 100% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.3); opacity: 0.5; }
}

/* ── Info Box ─────────────────────────────────────────────────────────── */
.box {
    background: linear-gradient(135deg, rgba(20, 29, 52, 0.75) 0%, rgba(15, 23, 42, 0.6) 100%);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-left: 4px solid #6366F1;
    border-radius: 12px;
    padding: 1rem 1.4rem;
    margin-bottom: 1.4rem;
    color: #CBD5E1;
    font-size: 0.92rem;
    line-height: 1.55;
    backdrop-filter: blur(16px);
    box-shadow: 0 8px 24px -6px rgba(0, 0, 0, 0.4);
}
.box b { color: #A5B4FC; }

/* ── Radar Visualizer (Stripe Radar Aesthetic) ────────────────────────── */
.radar-container {
    position: relative;
    width: 290px;
    height: 290px;
    margin: 0 auto 1.5rem;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(15, 23, 42, 0.95) 0%, rgba(9, 14, 26, 0.98) 100%);
    border: 1px solid rgba(99, 102, 241, 0.25);
    box-shadow: 0 16px 40px -10px rgba(0, 0, 0, 0.7), 0 0 30px rgba(99, 102, 241, 0.15), inset 0 0 40px rgba(99, 102, 241, 0.08);
    overflow: hidden;
}

.radar-rings {
    position: absolute;
    top: 0; left: 0; width: 100%; height: 100%;
    border-radius: 50%;
    background:
        radial-gradient(circle, transparent 24%, rgba(99, 102, 241, 0.15) 24.5%, transparent 25%),
        radial-gradient(circle, transparent 49%, rgba(99, 102, 241, 0.15) 49.5%, transparent 50%),
        radial-gradient(circle, transparent 74%, rgba(99, 102, 241, 0.15) 74.5%, transparent 75%);
}

.radar-sweep {
    position: absolute;
    top: 0; left: 0; width: 100%; height: 100%;
    border-radius: 50%;
    background: conic-gradient(
        from 0deg,
        transparent 0deg,
        rgba(99, 102, 241, 0.3) 25deg,
        rgba(139, 92, 246, 0.08) 60deg,
        transparent 90deg
    );
    animation: radar-spin 3.2s linear infinite;
}

@keyframes radar-spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

.radar-crosshair {
    position: absolute;
    top: 50%; left: 0; width: 100%; height: 1px;
    background: rgba(99, 102, 241, 0.18);
    transform: translateY(-50%);
}

.radar-crosshair-v {
    position: absolute;
    top: 0; left: 50%; width: 1px; height: 100%;
    background: rgba(99, 102, 241, 0.18);
    transform: translateX(-50%);
}

.radar-dot {
    position: absolute;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    transform: translate(-50%, -50%);
    z-index: 2;
}

.radar-dot.fraud {
    background: #FB7185;
    box-shadow: 0 0 10px #F43F5E, 0 0 20px rgba(244, 63, 94, 0.6);
    animation: pulse-dot-fraud 1.8s infinite;
}

.radar-dot.safe {
    background: #34D399;
    box-shadow: 0 0 8px #10B981;
    animation: pulse-dot-safe 2.5s infinite;
}

@keyframes pulse-dot-fraud {
    0%, 100% { transform: translate(-50%, -50%) scale(1); opacity: 1; }
    50% { transform: translate(-50%, -50%) scale(1.4); opacity: 0.6; }
}

@keyframes pulse-dot-safe {
    0%, 100% { transform: translate(-50%, -50%) scale(1); opacity: 0.85; }
    50% { transform: translate(-50%, -50%) scale(0.85); opacity: 0.35; }
}

/* ── Streamlit Form & Controls Overrides ───────────────────────────────── */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {
    background-color: rgba(15, 23, 42, 0.75) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 9px !important;
    color: #F8FAFC !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.95rem !important;
    transition: all 0.2s ease !important;
}

div[data-baseweb="select"] > div:hover,
div[data-baseweb="input"] > div:hover,
[data-testid="stNumberInput"] input:hover {
    border-color: rgba(99, 102, 241, 0.35) !important;
}

div[data-baseweb="select"] > div:focus-within,
div[data-baseweb="input"] > div:focus-within,
[data-testid="stNumberInput"] input:focus {
    border-color: #6366F1 !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2) !important;
}

/* Primary and Secondary Buttons */
button[kind="primary"], .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%) !important;
    color: #FFFFFF !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.4rem !important;
    box-shadow: 0 4px 16px rgba(99, 102, 241, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
    transition: all 0.2s ease !important;
}

button[kind="primary"]:hover, .stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #4F46E5 0%, #4338CA 100%) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5) !important;
}

button[kind="secondary"], .stButton > button {
    background: rgba(20, 29, 52, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    color: #CBD5E1 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 500 !important;
    border-radius: 9px !important;
    transition: all 0.2s ease !important;
}

button[kind="secondary"]:hover, .stButton > button:hover {
    background: rgba(99, 102, 241, 0.1) !important;
    border-color: rgba(99, 102, 241, 0.4) !important;
    color: #F8FAFC !important;
    transform: translateY(-1px) !important;
}

/* ── Dataframes & Tables ──────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    background: rgba(15, 23, 42, 0.5) !important;
}

/* ── Live Stream / Transaction Feed Card ──────────────────────────────── */
.txn-card {
    border-left: 3px solid #6366F1;
    background: rgba(15, 23, 42, 0.7);
    border-top: 1px solid rgba(255, 255, 255, 0.05);
    border-right: 1px solid rgba(255, 255, 255, 0.05);
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 0 10px 10px 0;
    padding: 0.65rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.88rem;
    display: flex;
    align-items: center;
    gap: 1.2rem;
    transition: all 0.2s ease;
}

.txn-card:hover {
    background: rgba(20, 29, 52, 0.85);
    transform: translateX(2px);
}

.txn-card.fraud {
    border-left-color: #F43F5E;
    background: rgba(244, 63, 94, 0.07);
    border-right-color: rgba(244, 63, 94, 0.2);
}

/* ── Pipeline Steps ───────────────────────────────────────────────────── */
.pipeline-step {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    background: rgba(99, 102, 241, 0.08);
    border: 1px solid rgba(99, 102, 241, 0.22);
    border-radius: 8px;
    padding: 0.45rem 0.9rem;
    font-size: 0.85rem;
    font-weight: 600;
    color: #A5B4FC;
}

.pipeline-arrow {
    color: rgba(99, 102, 241, 0.4);
    font-size: 1.1rem;
    margin: 0 0.25rem;
}

/* ── Demo Ribbon ──────────────────────────────────────────────────────── */
.demo-ribbon {
    position: fixed;
    top: 14px;
    right: 18px;
    background: linear-gradient(135deg, #F59E0B, #D97706);
    color: #0F172A;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 0.35rem 0.85rem;
    border-radius: 9999px;
    z-index: 99999;
    letter-spacing: 0.04em;
    box-shadow: 0 4px 14px rgba(245, 158, 11, 0.4);
}
</style>
"""

def get_css() -> str:
    """Return the full CSS block including Google Fonts link."""
    return GOOGLE_FONTS + CSS
