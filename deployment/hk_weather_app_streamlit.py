from __future__ import annotations

from pathlib import Path
import json
import math

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from shapely.geometry import box, Point
from shapely.prepared import prep

st.set_page_config(page_title="Hong Kong Weather Center", page_icon="🌦️", layout="wide")

# ─── Paths / config ─────────────────────────────────────────────────────────
OUT_DIR = Path("outputs")
DATA_DIR = Path("data")
INTERIM_DIR = DATA_DIR / "interim"
APP_DIR = OUT_DIR / "app"
PRED_PATH = APP_DIR / "hk_best_model_predictions.parquet"
META_PATH = APP_DIR / "hk_best_model_metadata.json"

GRID_CANDIDATES = [
    INTERIM_DIR / "grid_HK_land_clipped.geojson",
    INTERIM_DIR / "grid_HK.geojson",
    OUT_DIR / "grid_HK.geojson",
    OUT_DIR / "grids" / "HK_grid.geojson",
]
BOUNDARY_CANDIDATES = [
    INTERIM_DIR / "hk_land_boundary.geojson",
    INTERIM_DIR / "hk_raw_ib1000.geojson",
]

GRID_SIZE_DEG = 0.1
VERY_HOT_TEMP_C = 33.0
HOT_TEMP_C = 31.0
SMOOTH_CELL_STEP_DEG = 0.008
SMOOTH_IDW_POWER = 2.0
SMOOTH_K_NEAREST = 6
SMOOTH_OPACITY = 0.82
COARSE_OPACITY = 0.32

PLOTLY_CONFIG = {
    "scrollZoom": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"],
}

# ─── Design tokens ───────────────────────────────────────────────────────────
ACCENT_WARM = "#f9a84d"
ACCENT_COOL = "#1e6fa8"
ACCENT_RAIN = "#365f96"
ACCENT_GREEN = "#5d9a63"
TX0 = "rgba(255,255,255,0.98)"
TX1 = "rgba(255,255,255,0.92)"
TX2 = "rgba(255,255,255,0.80)"

TEMP_COLORS = [
    [0.00, "#6d94bd"], [0.18, "#9cbad5"], [0.36, "#d4dbe3"],
    [0.52, "#f2e5c8"], [0.70, "#e7b689"], [0.85, "#d4906f"], [1.00, "#a25d52"],
]
HUMID_COLORS = [
    [0.00, "#f3f8f1"], [0.22, "#d6e6d0"], [0.42, "#a8cfa4"],
    [0.62, "#76a878"], [0.82, "#538258"], [1.00, "#365a3c"],
]
RAIN_COLORS = [
    [0.00, "#f4f8fd"], [0.18, "#d8e4f1"], [0.38, "#a3bcd8"],
    [0.58, "#6f91ba"], [0.78, "#476793"], [1.00, "#243d60"],
]
ERR_COLORS = [[0.00, "#5b84ae"], [0.50, "#f2ead5"], [1.00, "#a56962"]]

TARGET_CFG = {
    "Temperature (°C)": {
        "pred_col": "pred_t2m_mean_tplus1",
        "actual_col": "actual_t2m_mean_tplus1",
        "err_col": "err_t2m_mean_tplus1",
        "colors": TEMP_COLORS,
        "diff_colors": ERR_COLORS,
        "fmt": "{:.1f}",
        "unit": "°C",
        "icon": "🌡️",
    },
    "Relative Humidity (%)": {
        "pred_col": "pred_rh2m_mean_tplus1",
        "actual_col": "actual_rh2m_mean_tplus1",
        "err_col": "err_rh2m_mean_tplus1",
        "colors": HUMID_COLORS,
        "diff_colors": ERR_COLORS,
        "fmt": "{:.0f}",
        "unit": "%",
        "icon": "💧",
    },
    "Rainfall (mm)": {
        "pred_col": "pred_tp_sum_mm_tplus1",
        "actual_col": "actual_tp_sum_mm_tplus1",
        "err_col": "err_tp_sum_mm_tplus1",
        "colors": RAIN_COLORS,
        "diff_colors": ERR_COLORS,
        "fmt": "{:.1f}",
        "unit": "mm",
        "icon": "☔",
    },
}


def hex_to_rgba(c: str, a: float = 0.18) -> str:
    h = c.lstrip("#")
    if len(h) == 3:
        h = "".join(x * 2 for x in h)
    return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})"


def compute_heat_index(T_c: float, RH: float) -> float | None:
    try:
        T_c = float(T_c)
        RH = float(RH)
    except (TypeError, ValueError):
        return None
    if T_c < 27 or RH < 40:
        return round(T_c, 1)
    T_f = T_c * 9 / 5 + 32
    c1, c2, c3, c4, c5, c6, c7, c8, c9 = (
        -42.379, 2.04901523, 10.14333127, -0.22475541,
        -0.00683783, -0.05481717, 0.00122874, 0.00085282, -0.00000199,
    )
    HI_f = (
        c1 + c2 * T_f + c3 * RH + c4 * T_f * RH + c5 * T_f ** 2 + c6 * RH ** 2
        + c7 * T_f ** 2 * RH + c8 * T_f * RH ** 2 + c9 * T_f ** 2 * RH ** 2
    )
    return round((HI_f - 32) * 5 / 9, 1)


def heat_comfort_label(actual: float, feels: float | None, rh: float) -> tuple[str, str]:
    if feels is None:
        return "Comfortable", "Conditions are broadly manageable."
    diff = feels - actual
    if feels >= 40:
        return "Dangerous Heat", "Extreme heat stress risk. Avoid outdoor activity."
    if feels >= 35:
        return "Very Hot", "High heat stress. Stay hydrated and in the shade."
    if feels >= 30:
        return "Hot", "Warm and humid conditions. Hydration is recommended."
    if diff >= 4:
        return "Muggier Than It Looks", f"Humidity ({rh:.0f}%) makes it feel {diff:.1f}° hotter."
    if diff <= -2:
        return "Cooler Than Expected", f"Feels {abs(diff):.1f}° cooler than the thermometer reads."
    return "Feels About Right", "Conditions feel close to the measured temperature."


_CSS = """
.stApp {
    background:
        radial-gradient(ellipse 80% 55% at 5% -5%, rgba(30,90,190,0.65), transparent 52%),
        radial-gradient(ellipse 70% 50% at 95% 10%, rgba(20,130,200,0.40), transparent 52%),
        radial-gradient(ellipse 60% 50% at 50% 105%, rgba(100,190,240,0.35), transparent 55%),
        linear-gradient(175deg,#0e3f78 0%,#1565a8 15%,#2b84c8 30%,#4fa2d9 48%,#7dbde8 64%,#a8d4f2 78%,#cce5fb 90%,#e8f5fd 100%);
    background-attachment: fixed;
    min-height: 100vh;
}

html, body, .stApp,
[data-testid="stMarkdownContainer"],
.stTextInput input,
.stSelectbox,
.stMultiSelect,
.stRadio,
.stSlider,
.stDataFrame {
    font-family: 'Manrope', -apple-system, BlinkMacSystemFont, sans-serif;
}

[class*="material-symbols"],
[class^="material-symbols"],
[data-testid="stExpanderIcon"] span,
[data-testid="baseButton-header"] span,
[data-testid="collapsedControl"] span {
    font-family: "Material Symbols Rounded" !important;
    letter-spacing: normal !important;
    text-transform: none !important;
    font-weight: normal !important;
}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2.5rem;
    max-width: 1680px;
    position: relative;
    z-index: 1;
}

[data-testid="stSidebar"] {
    background: linear-gradient(175deg, rgba(8,40,100,0.82) 0%, rgba(12,55,130,0.88) 100%);
    backdrop-filter: blur(22px);
    -webkit-backdrop-filter: blur(22px);
    border-right: 1px solid rgba(255,255,255,0.14);
    box-shadow: 4px 0 30px rgba(5,30,80,0.25);
}
[data-testid="stSidebar"] * {
    color: rgba(255,255,255,0.92);
}
[data-testid="stSidebar"] h2 {
    font-family: 'Fraunces', serif;
    font-weight: 500;
    font-size: 1.35rem;
    letter-spacing: -0.01em;
    color: white;
    margin-bottom: 1.2rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid rgba(255,255,255,0.18);
}
[data-testid="stSidebar"] label {
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
    text-transform: none !important;
    color: rgba(255,255,255,0.82) !important;
}

.hero-wrap {
    position: relative;
    background: linear-gradient(135deg, rgba(255,255,255,0.26) 0%, rgba(255,255,255,0.10) 100%);
    backdrop-filter: blur(30px);
    -webkit-backdrop-filter: blur(30px);
    border: 1px solid rgba(255,255,255,0.38);
    border-radius: 28px;
    padding: 34px 38px;
    box-shadow: 0 10px 40px rgba(5,30,100,0.22), inset 0 1px 0 rgba(255,255,255,0.50);
    margin-bottom: 20px;
    overflow: hidden;
}
.hero-wrap::after {
    content: "";
    position: absolute;
    top: -60px;
    right: -60px;
    width: 220px;
    height: 220px;
    background: radial-gradient(circle, rgba(249,168,77,0.22), transparent 65%);
    pointer-events: none;
}
.hero-kicker {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    color: rgba(255,255,255,0.78);
    margin-bottom: 16px;
    display: inline-flex;
    align-items: center;
    gap: 10px;
}
.hero-kicker::before {
    content: "";
    width: 28px;
    height: 1px;
    background: rgba(255,255,255,0.60);
    display: inline-block;
}
.hero-title {
    font-family: 'Fraunces', serif;
    font-size: 2.7rem;
    font-weight: 500;
    letter-spacing: -0.025em;
    color: white;
    line-height: 1.08;
    margin-bottom: 14px;
    text-shadow: 0 2px 18px rgba(5,30,100,0.22);
}
.hero-title em {
    font-style: italic;
    font-weight: 400;
    color: TOKEN_WARM;
}
.hero-sub {
    color: rgba(255,255,255,0.92);
    font-size: 1.03rem;
    font-weight: 600;
    line-height: 1.58;
    max-width: 780px;
}

.glass-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.24) 0%, rgba(255,255,255,0.10) 100%);
    backdrop-filter: blur(22px);
    -webkit-backdrop-filter: blur(22px);
    border: 1px solid rgba(255,255,255,0.36);
    border-radius: 22px;
    padding: 22px 24px;
    box-shadow: 0 8px 32px rgba(5,30,100,0.18), inset 0 1px 0 rgba(255,255,255,0.42);
    color: white;
}
.glass-card * { color: white !important; }
.glass-card h3 {
    font-family: 'Fraunces', serif;
    font-weight: 500;
    font-size: 1.15rem;
    letter-spacing: -0.01em;
    margin-bottom: 10px;
}

.metric-card {
    position: relative;
    backdrop-filter: blur(22px);
    -webkit-backdrop-filter: blur(22px);
    border: 1px solid rgba(255,255,255,0.42);
    border-radius: 20px;
    padding: 22px 24px;
    height: 188px;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    box-shadow: 0 8px 32px rgba(5,30,100,0.18), inset 0 1px 0 rgba(255,255,255,0.50);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    overflow: hidden;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 16px 48px rgba(5,30,100,0.25), inset 0 1px 0 rgba(255,255,255,0.55);
}
.metric-card::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.65), transparent);
}
.metric-topline {
    font-family: 'Manrope', sans-serif;
    color: rgba(255,255,255,0.82);
    font-size: 0.82rem;
    font-weight: 800;
    text-transform: none;
    letter-spacing: 0.01em;
    margin-bottom: 10px;
}
.metric-value {
    font-family: 'Fraunces', serif;
    font-size: 2.15rem;
    font-weight: 500;
    letter-spacing: -0.02em;
    color: white;
    line-height: 1.00;
    margin-bottom: 8px;
    text-shadow: 0 2px 12px rgba(5,30,100,0.18);
    word-break: break-word;
}
.metric-value.compact {
    font-size: 1.55rem;
    line-height: 1.05;
}
.metric-sub {
    color: rgba(255,255,255,0.94);
    font-size: 0.9rem;
    font-weight: 600;
    line-height: 1.42;
    min-height: 3.3em;
}

.mini-pill {
    display: inline-block;
    padding: 5px 12px;
    border-radius: 999px;
    background: rgba(255,255,255,0.16);
    color: white;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 500;
    margin-right: 6px;
    margin-bottom: 6px;
    border: 1px solid rgba(255,255,255,0.28);
    letter-spacing: 0.02em;
}

.forecast-card {
    position: relative;
    background: linear-gradient(145deg, rgba(255,255,255,0.28) 0%, rgba(255,255,255,0.12) 100%);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border: 1px solid rgba(255,255,255,0.40);
    border-radius: 18px;
    padding: 16px 12px 18px;
    height: 220px;
    text-align: center;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    align-items: center;
    box-sizing: border-box;
    transition: transform 0.22s ease, box-shadow 0.22s ease;
    box-shadow: 0 4px 20px rgba(5,30,100,0.14), inset 0 1px 0 rgba(255,255,255,0.48);
    overflow: hidden;
}
.forecast-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 32px rgba(5,30,100,0.22), inset 0 1px 0 rgba(255,255,255,0.55);
    border-color: rgba(249,168,77,0.55);
}
.forecast-date {
    font-family: 'Manrope', sans-serif;
    color: rgba(255,255,255,0.82);
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.01em;
    text-transform: none;
    margin-bottom: 10px;
}
.forecast-main {
    font-family: 'Fraunces', serif;
    font-size: 1.38rem;
    font-weight: 500;
    color: white;
    letter-spacing: -0.02em;
    line-height: 1;
    margin-bottom: 12px;
    min-height: 2rem;
    display: flex;
    align-items: center;
    justify-content: center;
    text-shadow: 0 2px 10px rgba(5,30,100,0.15);
}
.forecast-sub {
    color: rgba(255,255,255,0.96);
    font-size: 0.76rem;
    font-weight: 700;
    line-height: 1.34;
    min-height: 58px;
    display: flex;
    flex-direction: column;
    gap: 4px;
    align-items: center;
    justify-content: flex-start;
}
.forecast-sub strong { color: white; font-weight: 800; }
.forecast-line {
    white-space: nowrap;
}

.dashboard-bottom-gap {
    height: 28px;
}

.legend-list {
    display: flex;
    flex-direction: column;
    gap: 12px;
    margin-top: 14px;
}
.legend-item {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 12px 14px;
    border-radius: 14px;
    background: linear-gradient(135deg, rgba(255,255,255,0.14), rgba(255,255,255,0.06));
    border: 1px solid rgba(255,255,255,0.18);
}
.legend-emoji {
    font-size: 1.08rem;
    line-height: 1.2;
    margin-top: 1px;
}
.legend-text {
    color: rgba(255,255,255,0.96);
    font-size: 0.96rem;
    font-weight: 650;
    line-height: 1.55;
}

.advice-card {
    position: relative;
    background: linear-gradient(145deg, rgba(20,60,120,0.48) 0%, rgba(255,255,255,0.14) 100%);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border: 1px solid rgba(255,255,255,0.44);
    border-left: 4px solid TOKEN_WARM;
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 12px;
    box-shadow: 0 6px 22px rgba(5,30,100,0.18);
}
.advice-title {
    font-family: 'Fraunces', serif;
    font-weight: 700;
    font-size: 1.1rem;
    color: rgba(255,255,255,1.0);
    margin-bottom: 10px;
    letter-spacing: -0.01em;
    text-shadow: 0 1px 8px rgba(5,30,100,0.20);
}
.advice-body,
.advice-card > div:last-child {
    color: rgba(255,255,255,0.98);
    font-size: 1rem;
    line-height: 1.62;
    font-weight: 650;
}
.advice-card em {
    color: rgba(255,255,255,0.94) !important;
    font-style: italic;
    font-weight: 600;
}

.section-label {
    font-family: 'Fraunces', serif;
    color: white;
    font-size: 1.42rem;
    font-weight: 550;
    letter-spacing: -0.015em;
    margin-top: 10px;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 12px;
    text-shadow: 0 1px 8px rgba(5,30,100,0.18);
    padding-left: 2px;
}
.section-label::after {
    content: "";
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(255,255,255,0.30), transparent);
    margin-left: 4px;
}
.small-note {
    color: rgba(255,255,255,0.86);
    font-size: 0.87rem;
    font-style: italic;
    line-height: 1.55;
    margin-top: 12px;
    font-weight: 550;
}

/* Tabs */
div[data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 1px solid rgba(255,255,255,0.20);
    padding-bottom: 2px;
    margin-bottom: 16px;
}
div[data-baseweb="tab-list"] button {
    background: transparent !important;
    border: none !important;
    padding: 10px 18px !important;
    border-radius: 10px 10px 0 0 !important;
    transition: background 0.18s ease;
}
div[data-baseweb="tab-list"] button:hover {
    background: rgba(255,255,255,0.09) !important;
}
div[data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-family: 'Manrope', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    color: rgba(255,255,255,0.82) !important;
    letter-spacing: 0.01em;
}
div[data-baseweb="tab-list"] button[aria-selected="true"] [data-testid="stMarkdownContainer"] p {
    color: TOKEN_WARM !important;
    font-weight: 800 !important;
}
div[data-baseweb="tab-highlight"] {
    background: TOKEN_WARM !important;
    height: 2.5px !important;
    border-radius: 2px;
}

.stSelectbox [data-baseweb="select"] > div,
.stMultiSelect [data-baseweb="select"] > div,
.stTextInput input {
    background: rgba(255,255,255,0.14) !important;
    border: 1px solid rgba(255,255,255,0.28) !important;
    border-radius: 12px !important;
    color: white !important;
    font-family: 'Manrope', sans-serif !important;
    box-shadow: none !important;
}
.stSelectbox [data-baseweb="select"] > div > div,
.stMultiSelect [data-baseweb="select"] > div > div {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
.stSelectbox [data-baseweb="select"] span,
.stMultiSelect [data-baseweb="select"] span,
.stTextInput input::placeholder {
    color: rgba(255,255,255,0.96) !important;
    font-weight: 600 !important;
}
.stRadio label,
.stCheckbox label {
    color: rgba(255,255,255,0.96) !important;
    font-size: 0.92rem !important;
    font-weight: 600 !important;
}
.stSlider [data-baseweb="slider"] [role="slider"] {
    background: TOKEN_WARM !important;
    border: 2px solid white !important;
    box-shadow: 0 0 0 3px rgba(249,168,77,0.35) !important;
}
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.16) !important;
    border: 1px solid rgba(255,255,255,0.26) !important;
    border-radius: 12px !important;
    color: white !important;
    font-weight: 700 !important;
}
[data-testid="stDataFrame"] {
    background: rgba(255,255,255,0.94) !important;
    border: 1px solid rgba(30,50,90,0.10) !important;
    border-radius: 14px !important;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(5,30,100,0.10);
    --gdg-bg-cell: rgba(255,255,255,0.98);
    --gdg-bg-cell-medium: rgba(248,250,253,0.98);
    --gdg-bg-header: rgba(243,246,251,0.98);
    --gdg-bg-header-hovered: rgba(235,240,247,0.98);
    --gdg-border-color: rgba(30,50,90,0.12);
    --gdg-horizontal-border-color: rgba(30,50,90,0.08);
    --gdg-text-dark: #1a2847;
    --gdg-text-medium: #5a6b85;
    --gdg-accent-color: #1e6fa8;
    --gdg-font-family: 'Manrope', sans-serif;
}
.stPlotlyChart > div {
    border-radius: 18px;
    overflow: hidden;
    background: rgba(10,50,120,0.22);
    border: 1px solid rgba(255,255,255,0.24);
    box-shadow: 0 8px 32px rgba(5,30,100,0.18);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
}
.stAlert {
    background: rgba(255,255,255,0.18) !important;
    border: 1px solid rgba(255,255,255,0.30) !important;
    border-radius: 12px !important;
    color: white !important;
}

.light-table-wrap {
    background: rgba(255,255,255,0.92);
    border: 1px solid rgba(30,50,90,0.12);
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 8px 24px rgba(5,30,100,0.12);
}
.light-table-scroll {
    max-height: 420px;
    overflow-y: auto;
    overflow-x: auto;
}

.light-table {
    width: 100%;
    min-width: 980px;
    border-collapse: collapse;
    font-family: 'Manrope', sans-serif;
    color: #1a2847;
    background: rgba(255,255,255,0.98);
    table-layout: auto;
}
.light-table th {
    position: sticky;
    top: 0;
    z-index: 2;
    background: #eef3f9;
    color: #44556f;
    text-align: left;
    font-size: 0.84rem;
    font-weight: 700;
    padding: 12px 14px;
    border-bottom: 1px solid rgba(30,50,90,0.10);
}
.light-table td {
    padding: 11px 14px;
    border-bottom: 1px solid rgba(30,50,90,0.08);
    font-size: 0.92rem;
    vertical-align: top;
}
.light-table tbody tr:nth-child(even) td {
    background: rgba(247,250,253,0.88);
}
.light-table tbody tr:hover td {
    background: rgba(227,238,249,0.70);
}
.light-table .num {
    text-align: right;
    white-space: nowrap;
}
.light-table .mono {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.86rem;
}
.light-table .region-col {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.86rem;
    white-space: nowrap;
    min-width: 230px;
}
"""


def _inject_css() -> None:
    css = _CSS.replace("TOKEN_WARM", ACCENT_WARM)
    st.markdown(
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
        '<link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,500;0,9..144,600;0,9..144,700;1,9..144,400&family=Manrope:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">',
        unsafe_allow_html=True,
    )
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# ─── Plotly shared style ─────────────────────────────────────────────────────
PLOTLY_FONT = dict(family="Manrope, sans-serif", color=TX0, size=12)
PLOTLY_AXIS = dict(
    gridcolor="rgba(255,255,255,0.16)",
    zerolinecolor="rgba(255,255,255,0.24)",
    linecolor="rgba(255,255,255,0.24)",
    tickfont=dict(family="Manrope, sans-serif", color=TX1, size=11),
    title_font=dict(family="Manrope, sans-serif", color=TX1, size=12),
)
HOVER_STYLE = dict(
    bgcolor="rgba(8,40,100,0.94)",
    bordercolor=ACCENT_WARM,
    font=dict(family="Manrope, sans-serif", color="white", size=12),
)
_CBAR = dict(
    thickness=14,
    len=0.75,
    outlinewidth=0,
    tickfont=dict(family="Manrope, sans-serif", color="rgba(255,255,255,0.96)", size=10),
    title_font=dict(family="Manrope, sans-serif", color="white", size=11),
    bgcolor="rgba(0,0,0,0)",
    borderwidth=0,
)
_MLAY = dict(
    mapbox_style="carto-positron",
    mapbox_zoom=9.1,
    margin=dict(l=0, r=0, t=0, b=0),
    uirevision="stay",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=PLOTLY_FONT,
)


# ─── Data loaders ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_predictions() -> tuple[pd.DataFrame, dict]:
    if not PRED_PATH.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {PRED_PATH}\nRun: python deployment/make_hk_app_predictions.py"
        )
    df = pd.read_parquet(PRED_PATH)
    df["time"] = pd.to_datetime(df["time"])
    df["target_date"] = pd.to_datetime(df["target_date"])
    df["region_id"] = df["region_id"].astype(str)
    meta = json.loads(META_PATH.read_text(encoding="utf-8")) if META_PATH.exists() else {}
    return df, meta


@st.cache_data(show_spinner=False)
def load_boundary() -> gpd.GeoDataFrame | None:
    for p in BOUNDARY_CANDIDATES:
        if not p.exists():
            continue
        b = gpd.read_file(p).to_crs("EPSG:4326")
        b = b[b.geometry.notna() & ~b.geometry.is_empty & b.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
        if b.empty:
            continue
        try:
            mg = b.union_all()
        except AttributeError:
            mg = b.unary_union
        try:
            mg = mg.buffer(0)
        except Exception:
            pass
        return gpd.GeoDataFrame({"name": ["HK land"]}, geometry=[mg], crs="EPSG:4326")
    return None


@st.cache_data(show_spinner=False)
def load_grid(pred_df: pd.DataFrame, clip_to_boundary: bool = True) -> gpd.GeoDataFrame:
    chosen = None
    gdf = None
    for p in GRID_CANDIDATES:
        if p.exists():
            gdf = gpd.read_file(p).to_crs("EPSG:4326")
            chosen = p
            break
    if gdf is None and {"lat", "lon"}.issubset(pred_df.columns):
        centers = pred_df[["region_id", "lat", "lon"]].dropna().drop_duplicates("region_id")
        gdf = gpd.GeoDataFrame(
            centers,
            geometry=[
                box(lo - GRID_SIZE_DEG / 2, la - GRID_SIZE_DEG / 2, lo + GRID_SIZE_DEG / 2, la + GRID_SIZE_DEG / 2)
                for la, lo in zip(centers["lat"], centers["lon"])
            ],
            crs="EPSG:4326",
        )
    if gdf is None:
        raise FileNotFoundError("No HK grid found.")
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    gdf["region_id"] = gdf["region_id"].astype(str)
    if clip_to_boundary and chosen and "land_clipped" not in chosen.name:
        bnd = load_boundary()
        if bnd is not None:
            try:
                c = gpd.overlay(gdf, bnd[["geometry"]], how="intersection")
            except Exception:
                c = gpd.clip(gdf, bnd[["geometry"]])
            try:
                c = c.explode(index_parts=False).reset_index(drop=True)
            except TypeError:
                c = c.explode().reset_index(drop=True)
            c = c[c.geometry.notna() & ~c.geometry.is_empty].copy()
            if not c.empty:
                gdf = c
    return gdf


def add_centroids(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = gdf.copy()
    reps = out.geometry.representative_point()
    out["centroid_lon"] = reps.x
    out["centroid_lat"] = reps.y
    return out


# ─── Numeric helpers ─────────────────────────────────────────────────────────
def robust_range(s: pd.Series, zero_floor: bool = False, clip_q_low: float = 0.05, clip_q_high: float = 0.95) -> tuple[float, float]:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return (0.0, 1.0)
    lo = float(s.quantile(clip_q_low))
    hi = float(s.quantile(clip_q_high))
    if zero_floor:
        lo = 0.0
    if math.isclose(lo, hi):
        hi = lo + 1e-6
    return (lo, hi)


def symmetric_range(s: pd.Series, q: float = 0.95) -> tuple[float, float]:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return (-1.0, 1.0)
    m = max(float(np.nanquantile(np.abs(s), q)), 1e-6)
    return (-m, m)


def make_step_colorscale(cl: list[str]) -> list[list[float | str]]:
    n = len(cl)
    sc: list[list[float | str]] = []
    for i, c in enumerate(cl):
        sc.append([i / n, c])
        sc.append([(i + 1) / n, c])
    sc[0][0] = 0.0
    sc[-1][0] = 1.0
    return sc


def extract_color_list(scale) -> list[str]:
    return [x[1] if isinstance(x, (list, tuple)) else x for x in scale]


def compute_band_info(s: pd.Series, bins: int = 7, fmt: str = "{:.1f}", unit: str = ""):
    raw = pd.to_numeric(s, errors="coerce")
    valid = raw.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=raw.index, dtype=float), [], 0, None
    nb = int(min(bins, max(2, valid.nunique())))
    edges = np.unique(valid.quantile(np.linspace(0, 1, nb + 1)).to_numpy())
    if len(edges) < 2:
        return (
            pd.Series(0.0, index=raw.index, dtype=float),
            [f"{fmt.format(valid.min())}–{fmt.format(valid.max())}{unit}"],
            1,
            np.array([valid.min(), valid.max()]),
        )
    codes = pd.Series(np.nan, index=raw.index, dtype=float)
    codes.loc[valid.index] = pd.cut(valid, bins=edges, labels=False, include_lowest=True, duplicates="drop").astype(float)
    labels = [f"{fmt.format(edges[i])}–{fmt.format(edges[i+1])}{unit}" for i in range(len(edges) - 1)]
    return codes, labels, len(labels), edges


def band_codes_from_edges(values: pd.Series, edges: np.ndarray | None) -> pd.Series:
    raw = pd.to_numeric(values, errors="coerce")
    out = pd.Series(np.nan, index=raw.index, dtype=float)
    if edges is None or len(edges) < 2:
        return out
    out.loc[raw.notna()] = pd.cut(raw.dropna(), bins=edges, labels=False, include_lowest=True, duplicates="drop").astype(float)
    return out


def merge_map_frame(grid: gpd.GeoDataFrame, day_df: pd.DataFrame) -> gpd.GeoDataFrame:
    cols = [
        c for c in [
            "region_id", "target_date",
            "pred_t2m_mean_tplus1", "pred_rh2m_mean_tplus1", "pred_tp_sum_mm_tplus1",
            "actual_t2m_mean_tplus1", "actual_rh2m_mean_tplus1", "actual_tp_sum_mm_tplus1",
            "err_t2m_mean_tplus1", "err_rh2m_mean_tplus1", "err_tp_sum_mm_tplus1",
        ] if c in day_df.columns
    ]
    return grid.merge(day_df[cols], on="region_id", how="left")


def idw_interpolate(qxy: np.ndarray, sxy: np.ndarray, sv: np.ndarray, power: float = 2.0, k: int = 6) -> np.ndarray:
    out = np.empty(len(qxy), dtype=float)
    for i, (xq, yq) in enumerate(qxy):
        d = np.sqrt((sxy[:, 0] - xq) ** 2 + (sxy[:, 1] - yq) ** 2)
        j = np.argmin(d)
        if d[j] < 1e-12:
            out[i] = sv[j]
            continue
        if k and k < len(d):
            idx = np.argsort(d)[:k]
            d = d[idx]
            v = sv[idx]
        else:
            v = sv
        w = 1.0 / np.maximum(d, 1e-6) ** power
        out[i] = np.sum(w * v) / np.sum(w)
    return out


def build_smoothed_surface(
    boundary_gdf: gpd.GeoDataFrame | None,
    coarse_gdf: gpd.GeoDataFrame,
    value_col: str,
    step_deg: float = SMOOTH_CELL_STEP_DEG,
    power: float = SMOOTH_IDW_POWER,
    k: int = SMOOTH_K_NEAREST,
) -> gpd.GeoDataFrame | None:
    if boundary_gdf is None or boundary_gdf.empty:
        return None
    src = coarse_gdf.dropna(subset=["centroid_lon", "centroid_lat", value_col]).copy()
    if src.empty:
        return None
    try:
        land_geom = boundary_gdf.union_all()
    except AttributeError:
        land_geom = boundary_gdf.unary_union
    prepared = prep(land_geom)
    minx, miny, maxx, maxy = boundary_gdf.total_bounds
    xs = np.arange(minx, maxx + step_deg, step_deg)
    ys = np.arange(miny, maxy + step_deg, step_deg)
    centers, geoms = [], []
    half = step_deg / 2.0
    for y in ys:
        for x in xs:
            if not prepared.covers(Point(x, y)):
                continue
            centers.append((x, y))
            geoms.append(box(x - half, y - half, x + half, y + half))
    if not geoms:
        return None
    surface = gpd.GeoDataFrame(
        {
            "smooth_id": [f"s_{i}" for i in range(len(geoms))],
            "centroid_lon": [c[0] for c in centers],
            "centroid_lat": [c[1] for c in centers],
        },
        geometry=geoms,
        crs="EPSG:4326",
    )
    source_xy = src[["centroid_lon", "centroid_lat"]].to_numpy(dtype=float)
    source_val = pd.to_numeric(src[value_col], errors="coerce").to_numpy(dtype=float)
    source_region = src["region_id"].astype(str).to_numpy()
    query_xy = surface[["centroid_lon", "centroid_lat"]].to_numpy(dtype=float)
    surface["smooth_value"] = idw_interpolate(query_xy, source_xy, source_val, power, k)
    nearest_region_ids = []
    for xq, yq in query_xy:
        d = np.sqrt((source_xy[:, 0] - xq) ** 2 + (source_xy[:, 1] - yq) ** 2)
        nearest_region_ids.append(source_region[np.argmin(d)])
    surface["region_id"] = nearest_region_ids
    try:
        surface = gpd.overlay(surface, boundary_gdf[["geometry"]].copy(), how="intersection")
    except Exception:
        surface = gpd.clip(surface, boundary_gdf[["geometry"]].copy())
    surface = surface[surface.geometry.notna() & ~surface.geometry.is_empty].copy()
    try:
        surface = surface.explode(index_parts=False).reset_index(drop=True)
    except TypeError:
        surface = surface.explode().reset_index(drop=True)
    if surface.empty:
        return None
    metric = surface.to_crs(3857)
    keep = metric.geometry.area > 1
    surface = surface.loc[keep.values].copy().to_crs("EPSG:4326")
    surface["smooth_id"] = [f"s_{i}" for i in range(len(surface))]
    reps = surface.geometry.representative_point()
    surface["centroid_lon"] = reps.x
    surface["centroid_lat"] = reps.y
    return surface


# ─── Map builders ────────────────────────────────────────────────────────────
def make_single_target_map(
    gdf: gpd.GeoDataFrame,
    plot_col: str,
    raw_col: str,
    colorscale,
    range_color: tuple[float, float],
    label: str,
    target_cfg: dict,
    boundary_gdf: gpd.GeoDataFrame | None = None,
    band_mode: bool = False,
    band_labels: list[str] | None = None,
    band_edges: np.ndarray | None = None,
    render_mode: str = "Smoothed Land Surface",
    show_region_edges: bool = False,
):
    center_lat = float(gdf["centroid_lat"].mean())
    center_lon = float(gdf["centroid_lon"].mean())
    fig = go.Figure()

    render_mode_norm = render_mode.lower().strip()
    use_smoothed = render_mode_norm == "smoothed land surface"

    if use_smoothed and boundary_gdf is not None:
        surface = build_smoothed_surface(boundary_gdf, gdf, raw_col)
        if surface is not None and not surface.empty:
            if band_mode:
                surface["__plot"] = band_codes_from_edges(surface["smooth_value"], band_edges)
                zcol = "__plot"
                zmin, zmax = 0, max(len(band_labels or []) - 1, 1)
                colorbar = {
                    **_CBAR,
                    "title": label,
                    "tickvals": list(range(len(band_labels or []))),
                    "ticktext": band_labels or [],
                }
            else:
                zcol = "smooth_value"
                zmin, zmax = range_color
                colorbar = {**_CBAR, "title": label}

            geojson = json.loads(surface[["smooth_id", "geometry"]].to_json())
            fig.add_trace(
                go.Choroplethmapbox(
                    geojson=geojson,
                    locations=surface["smooth_id"],
                    z=surface[zcol],
                    featureidkey="properties.smooth_id",
                    colorscale=colorscale,
                    zmin=zmin,
                    zmax=zmax,
                    marker_opacity=SMOOTH_OPACITY,
                    marker_line_width=0,
                    marker_line_color="rgba(0,0,0,0)",
                    colorbar=colorbar,
                    customdata=np.stack([surface["region_id"].astype(str).to_numpy(), surface["smooth_value"].to_numpy()], axis=-1),
                    hovertemplate=f"<b>%{{customdata[0]}}</b><br>Predicted: %{{customdata[1]:.2f}} {target_cfg['unit']}<extra></extra>",
                )
            )
            if show_region_edges:
                ej = json.loads(gdf[["region_id", "geometry"]].to_json())
                fig.add_trace(
                    go.Choroplethmapbox(
                        geojson=ej,
                        locations=gdf["region_id"],
                        z=np.zeros(len(gdf)),
                        featureidkey="properties.region_id",
                        colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
                        showscale=False,
                        marker_opacity=0.01,
                        marker_line_width=0.08,
                        marker_line_color="rgba(30,50,90,0.20)",
                        hoverinfo="skip",
                    )
                )
            fig.update_layout(**_MLAY, mapbox_center={"lat": center_lat, "lon": center_lon}, height=680)
            return fig

    geojson = json.loads(gdf[["region_id", "geometry"]].to_json())
    custom_cols = ["region_id", target_cfg["pred_col"], target_cfg["actual_col"], target_cfg["err_col"]]
    custom = np.stack(
        [gdf[c].to_numpy() if c in gdf.columns else np.full(len(gdf), np.nan) for c in custom_cols],
        axis=-1,
    )
    colorbar = {**_CBAR, "title": label}
    if band_mode and band_labels:
        colorbar["tickvals"] = list(range(len(band_labels)))
        colorbar["ticktext"] = band_labels
    fig.add_trace(
        go.Choroplethmapbox(
            geojson=geojson,
            locations=gdf["region_id"],
            z=gdf[plot_col],
            featureidkey="properties.region_id",
            colorscale=colorscale,
            zmin=range_color[0],
            zmax=range_color[1],
            marker_opacity=COARSE_OPACITY,
            marker_line_width=0.12,
            marker_line_color="rgba(30,50,90,0.18)",
            colorbar=colorbar,
            customdata=custom,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Predicted: %{customdata[1]:.2f}<br>"
                "Actual: %{customdata[2]:.2f}<br>"
                "Error: %{customdata[3]:.2f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(**_MLAY, mapbox_center={"lat": center_lat, "lon": center_lon}, height=680)
    return fig


def make_combined_map(gdf: gpd.GeoDataFrame, boundary_gdf: gpd.GeoDataFrame | None):
    center_lat = float(gdf["centroid_lat"].mean())
    center_lon = float(gdf["centroid_lon"].mean())
    fig = go.Figure()

    temp_col = TARGET_CFG["Temperature (°C)"]["pred_col"]
    rh_col = TARGET_CFG["Relative Humidity (%)"]["pred_col"]
    rain_col = TARGET_CFG["Rainfall (mm)"]["pred_col"]

    temp_range = robust_range(gdf[temp_col], clip_q_low=0.03, clip_q_high=0.97)
    surface = build_smoothed_surface(boundary_gdf, gdf, temp_col) if boundary_gdf is not None else None
    if surface is not None and not surface.empty:
        geojson = json.loads(surface[["smooth_id", "geometry"]].to_json())
        fig.add_trace(
            go.Choroplethmapbox(
                geojson=geojson,
                locations=surface["smooth_id"],
                z=surface["smooth_value"],
                featureidkey="properties.smooth_id",
                colorscale=TEMP_COLORS,
                zmin=temp_range[0],
                zmax=temp_range[1],
                marker_opacity=0.88,
                marker_line_width=0,
                marker_line_color="rgba(0,0,0,0)",
                colorbar={**_CBAR, "title": "Temperature (°C)"},
                customdata=np.stack([surface["region_id"].astype(str).to_numpy(), surface["smooth_value"].to_numpy()], axis=-1),
                hovertemplate="<b>%{customdata[0]}</b><br>Temperature: %{customdata[1]:.1f}°C<extra></extra>",
            )
        )

    ov = gdf.copy()
    ov["rh"] = pd.to_numeric(ov[rh_col], errors="coerce")
    ov["rain"] = pd.to_numeric(ov[rain_col], errors="coerce")
    ov["temp"] = pd.to_numeric(ov[temp_col], errors="coerce")
    ov["humid_lat"] = ov["centroid_lat"] + 0.010
    ov["rain_lat"] = ov["centroid_lat"] - 0.010

    fig.add_trace(
        go.Scattermapbox(
            lat=ov["humid_lat"], lon=ov["centroid_lon"], mode="markers",
            marker={"size": 38, "color": "rgba(255,255,255,0.92)", "opacity": 1.0},
            hoverinfo="skip", showlegend=False,
        )
    )
    fig.add_trace(
        go.Scattermapbox(
            lat=ov["rain_lat"], lon=ov["centroid_lon"], mode="markers",
            marker={"size": 38, "color": "rgba(255,255,255,0.92)", "opacity": 1.0},
            hoverinfo="skip", showlegend=False,
        )
    )

    fig.add_trace(
        go.Scattermapbox(
            lat=ov["humid_lat"],
            lon=ov["centroid_lon"],
            mode="markers+text",
            text=[f"{v:.0f}%" if pd.notna(v) else "-" for v in ov["rh"]],
            textposition="middle center",
            textfont={"size": 12, "color": "#ffffff", "family": "Manrope, sans-serif"},
            marker={"size": 30, "color": "rgba(93,154,99,0.98)", "opacity": 0.98},
            name="Humidity",
            hovertemplate="<b>%{customdata[0]}</b><br>Temperature: %{customdata[1]:.1f}°C<br>Humidity: %{customdata[2]:.0f}%<br>Rainfall: %{customdata[3]:.1f} mm<extra></extra>",
            customdata=np.stack([ov["region_id"].astype(str), ov["temp"], ov["rh"], ov["rain"]], axis=-1),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scattermapbox(
            lat=ov["rain_lat"],
            lon=ov["centroid_lon"],
            mode="markers+text",
            text=[f"{v:.1f}" if pd.notna(v) else "-" for v in ov["rain"]],
            textposition="middle center",
            textfont={"size": 12, "color": "#ffffff", "family": "Manrope, sans-serif"},
            marker={"size": 30, "color": "rgba(54,95,150,0.98)", "opacity": 0.98},
            name="Rainfall",
            hovertemplate="<b>%{customdata[0]}</b><br>Temperature: %{customdata[1]:.1f}°C<br>Humidity: %{customdata[2]:.0f}%<br>Rainfall: %{customdata[3]:.1f} mm<extra></extra>",
            customdata=np.stack([ov["region_id"].astype(str), ov["temp"], ov["rh"], ov["rain"]], axis=-1),
            showlegend=False,
        )
    )
    fig.update_layout(**_MLAY, mapbox_center={"lat": center_lat, "lon": center_lon}, height=720)
    return fig


# ─── Chart builders ─────────────────────────────────────────────────────────
def make_region_timeseries(df: pd.DataFrame, region_id: str, pred_col: str, actual_col: str, title: str) -> go.Figure:
    d = df[df["region_id"] == str(region_id)].copy().sort_values("target_date")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["target_date"], y=d[actual_col], mode="lines", name="Actual", line=dict(width=2.4, color=ACCENT_WARM), opacity=0.95))
    fig.add_trace(go.Scatter(x=d["target_date"], y=d[pred_col], mode="lines", name="Predicted", line=dict(width=2.2, color="#9ed9ff", dash="dot"), opacity=0.95))
    fig.update_layout(
        title=dict(
            text=title,
            x=0.03,
            xanchor="left",
            font=dict(family="Fraunces, serif", color="white", size=16),
        ),
        xaxis=dict(title="Target Date", **PLOTLY_AXIS),
        yaxis=dict(title=title, **PLOTLY_AXIS),
        height=320,
        margin=dict(l=34, r=24, t=64, b=22),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=PLOTLY_FONT,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)", font=dict(family="Manrope, sans-serif", color=TX1, size=11)),
        hoverlabel=HOVER_STYLE,
    )
    return fig


def make_single_metric_history(df: pd.DataFrame, region_id: str, col: str, title: str, y_title: str, color: str) -> go.Figure:
    d = df[df["region_id"] == str(region_id)].copy().sort_values("target_date").tail(30)
    is_rain = y_title.lower() == "mm"
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=d["target_date"],
            y=d[col],
            mode="lines+markers",
            line={"width": 2.8, "color": color, "shape": "spline", "smoothing": 0.55},
            marker={"size": 7, "color": color, "line": {"width": 1.6, "color": "rgba(255,255,255,0.75)"}},
            fill="tozeroy" if is_rain else None,
            fillcolor=hex_to_rgba(color, 0.22) if is_rain else None,
        )
    )
    fig.update_layout(
        title=dict(text=title, font=dict(family="Fraunces, serif", color="white", size=16)),
        xaxis=dict(title="Date", **PLOTLY_AXIS),
        yaxis=dict(title=y_title, **PLOTLY_AXIS),
        height=280,
        margin=dict(l=24, r=24, t=56, b=22),
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=PLOTLY_FONT,
        hoverlabel=dict(bgcolor="rgba(8,40,100,0.94)", bordercolor=color, font=dict(family="Manrope, sans-serif", color="white", size=12)),
    )
    return fig


# ─── HTML helpers ────────────────────────────────────────────────────────────
def render_metric_card(title: str, value: str, subtitle: str, bg: str = "", compact: bool = False):
    sty = f' style="background:{bg};"' if bg else ""
    cls = "metric-value compact" if compact else "metric-value"
    value_html = str(value).replace("_", "_<wbr>")
    st.markdown(
        f'<div class="metric-card"{sty}><div class="metric-topline">{title}</div><div class="{cls}">{value_html}</div><div class="metric-sub">{subtitle}</div></div>',
        unsafe_allow_html=True,
    )


def render_light_table(df: pd.DataFrame, numeric_cols: list[str] | None = None, max_height: int = 420):
    if df.empty:
        st.info("No rows available.")
        return

    numeric_cols = set(numeric_cols or [])

    headers = []
    for col in df.columns:
        th_class = "region-col" if str(col).lower() == "region" else ""
        headers.append(f'<th class="{th_class}">{col}</th>')
    headers_html = "".join(headers)

    rows = []
    for _, row in df.iterrows():
        tds = []
        for col, val in row.items():
            if str(col).lower() == "region":
                cls = "region-col"
            elif col in numeric_cols:
                cls = "num"
            else:
                cls = ""

            if pd.isna(val):
                text = "—"
            elif col in numeric_cols:
                text = f"{val:,.4f}".rstrip("0").rstrip(".")
            else:
                text = str(val)

            tds.append(f'<td class="{cls}">{text}</td>')

        rows.append("<tr>" + "".join(tds) + "</tr>")

    html = (
        f'<div class="light-table-wrap"><div class="light-table-scroll" style="max-height:{max_height}px;">'
        f'<table class="light-table"><thead><tr>{headers_html}</tr></thead><tbody>{"".join(rows)}</tbody></table>'
        f'</div></div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def get_region_row(day_df: pd.DataFrame, region_id: str) -> pd.Series | None:
    d = day_df[day_df["region_id"] == str(region_id)]
    return d.iloc[0] if not d.empty else None


def rain_emoji(v: float) -> str:
    if pd.isna(v):
        return "☁️"
    if v >= 80:
        return "⛈️"
    if v >= 40:
        return "🌧️"
    if v >= 15:
        return "☔"
    if v > 0:
        return "🌦️"
    return "🌤️"


def render_forecast_strip(pred_df: pd.DataFrame, region_id: str, start_date: pd.Timestamp, horizon: int = 7):
    d = pred_df[(pred_df["region_id"] == str(region_id)) & (pred_df["target_date"] >= pd.Timestamp(start_date))].sort_values("target_date").head(horizon).copy()
    if d.empty:
        st.info("No forward dates available.")
        return
    cols = st.columns(len(d), gap="small")
    for col, (_, row) in zip(cols, d.iterrows()):
        col.markdown(
            f'''
            <div class="forecast-card">
                <div class="forecast-date">{pd.Timestamp(row["target_date"]).strftime("%d %b")}</div>
                <div style="font-size:1.42rem; margin-bottom:8px; line-height:1;">
                    {rain_emoji(row["pred_tp_sum_mm_tplus1"])}
                </div>
                <div class="forecast-main">{row["pred_t2m_mean_tplus1"]:.1f}°</div>
                <div class="forecast-sub">
                    <div class="forecast-line"><strong>{row["pred_rh2m_mean_tplus1"]:.0f}%</strong> RH</div>
                    <div class="forecast-line"><strong>{row["pred_tp_sum_mm_tplus1"]:.1f}</strong> mm</div>
                </div>
            </div>
            ''',
            unsafe_allow_html=True,
        )


def build_advisories(region_row: pd.Series | None, all_df: pd.DataFrame) -> list[tuple[str, str]]:
    if region_row is None:
        return []
    advisories = []
    temp = pd.to_numeric(region_row.get("pred_t2m_mean_tplus1"), errors="coerce")
    rain = pd.to_numeric(region_row.get("pred_tp_sum_mm_tplus1"), errors="coerce")
    rh = pd.to_numeric(region_row.get("pred_rh2m_mean_tplus1"), errors="coerce")
    if pd.notna(temp) and temp >= VERY_HOT_TEMP_C:
        advisories.append(("Heat Caution", "Very hot conditions predicted. Reduce sun exposure, hydrate early, and avoid strenuous outdoor activity."))
    elif pd.notna(temp) and temp >= HOT_TEMP_C:
        advisories.append(("Warm-Day Caution", "A hot day is expected. Bring water, use sunscreen, and plan outdoor activity carefully."))
    r95 = float(pd.to_numeric(all_df["pred_tp_sum_mm_tplus1"], errors="coerce").quantile(0.95))
    if pd.notna(rain) and rain >= r95:
        advisories.append(("Heavy-Rain Watch", "Rainfall is high relative to the forecast distribution. Consider extra travel time."))
    if pd.notna(temp) and pd.notna(rh) and temp >= 30 and rh >= 85:
        advisories.append(("Humid Heat Discomfort", "Temperature and humidity together suggest muggy conditions. Indoor cooling recommended."))
    if not advisories:
        advisories.append(("No Major Caution Flags", "Conditions look manageable for this date, though normal weather awareness is still recommended."))
    return advisories


def territory_summary(day_df: pd.DataFrame) -> dict:
    t = pd.to_numeric(day_df["pred_t2m_mean_tplus1"], errors="coerce").mean()
    rh = pd.to_numeric(day_df["pred_rh2m_mean_tplus1"], errors="coerce").mean()
    return {
        "temp": t,
        "rh": rh,
        "rain": pd.to_numeric(day_df["pred_tp_sum_mm_tplus1"], errors="coerce").mean(),
        "feels_like": compute_heat_index(t, rh),
    }


# ─── App ─────────────────────────────────────────────────────────────────────
_inject_css()
pred_df, meta = load_predictions()

with st.sidebar:
    st.markdown("## Controls")
    target_label = st.selectbox("Target", list(TARGET_CFG.keys()))
    view_mode = st.radio("Map Layer", ["Best-Model Forecast", "Actual Next-Day Value", "Forecast Error"])
    color_mode = st.selectbox("Colour Style", ["Continuous", "High-Contrast Bands"], index=0)
    clip_q = st.slider("Colour Contrast", min_value=0.80, max_value=0.99, value=0.96, step=0.01)
    clip_to_boundary = st.toggle("Clip Cells To Hong Kong Boundary", value=True)
    render_mode = st.radio("Map Rendering", ["Smoothed Land Surface", "Original Model Regions"], index=0)
    show_region_edges = st.toggle("Show Original Region Edges", value=False)

boundary_gdf = load_boundary()
grid = add_centroids(load_grid(pred_df, clip_to_boundary=clip_to_boundary))
if boundary_gdf is None:
    st.info("Land mask not found — add hk_land_boundary.geojson to data/interim/")

with st.sidebar:
    all_dates = sorted(pd.to_datetime(pred_df["target_date"].dropna().unique()))
    default_date = all_dates[min(len(all_dates) - 1, int(len(all_dates) * 0.8))]
    typed_date_str = st.text_input("Select Date", value="", placeholder="YYYY-MM-DD", help="e.g. 2024-03-14")
    slider_date = st.select_slider("Target Date", options=all_dates, value=default_date, format_func=lambda d: pd.Timestamp(d).strftime("%Y-%m-%d"))
    selected_date = slider_date
    if typed_date_str.strip():
        try:
            td = pd.to_datetime(typed_date_str.strip(), format="%Y-%m-%d").normalize()
            avail = {pd.Timestamp(d).normalize(): pd.Timestamp(d) for d in all_dates}
            if td in avail:
                selected_date = avail[td]
            else:
                st.warning("Date not in forecast data.")
        except ValueError:
            st.warning("Invalid format — use YYYY-MM-DD.")
    region_options = sorted(grid["region_id"].astype(str).unique())
    selected_region = st.selectbox("Location Focus", region_options)
    st.markdown("---")
    st.markdown("## Forecast Model")
    if meta.get("targets"):
        key = "t2m_mean" if target_label == "Temperature (°C)" else "rh2m_mean" if target_label == "Relative Humidity (%)" else "tp_sum_mm"
        info = meta["targets"].get(key, {})
        for lbl, k in [("Family", "family"), ("Tag", "model_tag"), ("RMSE", "report_rmse"), ("R²", "report_r2")]:
            st.markdown(f"<span class='mini-pill'>{lbl}: {info.get(k, '-')}</span>", unsafe_allow_html=True)

hero_date = pd.Timestamp(selected_date).strftime("%A, %d %B %Y")
st.markdown(
    f'<div class="hero-wrap"><div class="hero-kicker">Hong Kong Weather Center</div><div class="hero-title">Next-Day Forecast For <em>{hero_date}</em></div><div class="hero-sub">A regional meteorological dashboard for Hong Kong. Temperature forms the primary land surface, with humidity and rainfall layered as per-region indicators.</div></div>',
    unsafe_allow_html=True,
)

cfg = TARGET_CFG[target_label]
day_df = pred_df[pred_df["target_date"] == pd.Timestamp(selected_date)].copy()
map_gdf = merge_map_frame(grid, day_df)
summary = territory_summary(day_df)
region_row = get_region_row(day_df, selected_region)

view_mode_norm = view_mode.lower().strip()
if view_mode_norm == "best-model forecast":
    raw_col = cfg["pred_col"]
    plot_label = target_label
    base_colors = cfg["colors"]
    zero_floor = "Rainfall" in target_label
elif view_mode_norm == "actual next-day value":
    raw_col = cfg["actual_col"]
    plot_label = f"Actual {target_label}"
    base_colors = cfg["colors"]
    zero_floor = "Rainfall" in target_label
else:
    raw_col = cfg["err_col"]
    plot_label = f"Error ({target_label})"
    base_colors = cfg["diff_colors"]
    zero_floor = False

if color_mode == "High-Contrast Bands" and view_mode_norm != "forecast error":
    band_codes, band_labels, n_bands, band_edges = compute_band_info(map_gdf[raw_col], bins=7, fmt=cfg["fmt"], unit=cfg["unit"])
    map_gdf["__plot"] = band_codes
    color_list = extract_color_list(base_colors)
    colorscale = make_step_colorscale(color_list[:n_bands] if n_bands else color_list[:7])
    range_color = (0, max(n_bands - 1, 1))
    band_mode = True
else:
    band_labels = None
    band_edges = None
    map_gdf["__plot"] = pd.to_numeric(map_gdf[raw_col], errors="coerce")
    colorscale = base_colors
    range_color = symmetric_range(pred_df[raw_col], q=clip_q) if view_mode_norm == "forecast error" else robust_range(pred_df[raw_col], zero_floor=zero_floor, clip_q_low=1 - clip_q, clip_q_high=clip_q)
    band_mode = False

single_fig = make_single_target_map(
    map_gdf,
    plot_col="__plot",
    raw_col=raw_col,
    colorscale=colorscale,
    range_color=range_color,
    label=plot_label,
    target_cfg=cfg,
    boundary_gdf=boundary_gdf,
    band_mode=band_mode,
    band_labels=band_labels,
    band_edges=band_edges,
    render_mode=render_mode,
    show_region_edges=show_region_edges,
)
combined_fig = make_combined_map(map_gdf, boundary_gdf)

# top cards
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    render_metric_card("Territory Temperature", f"{summary['temp']:.1f}°C", "Mean Predicted Temperature", "linear-gradient(135deg,rgba(239,100,60,0.30),rgba(255,255,255,0.12))")
with c2:
    fl = summary.get("feels_like")
    fl_str = f"{fl:.1f}°C" if fl is not None else "—"
    diff = fl - summary['temp'] if fl is not None else 0
    diff_note = f"+{diff:.1f}° vs Actual" if diff > 0.5 else f"{diff:.1f}° vs Actual" if diff < -0.5 else "Same As Actual"
    render_metric_card("Feels Like", fl_str, diff_note, "linear-gradient(135deg,rgba(249,168,77,0.30),rgba(255,255,255,0.12))")
with c3:
    render_metric_card("Territory Humidity", f"{summary['rh']:.0f}%", "Mean Predicted RH", "linear-gradient(135deg,rgba(60,180,110,0.26),rgba(255,255,255,0.12))")
with c4:
    render_metric_card("Territory Rainfall", f"{summary['rain']:.1f} mm", "Mean Predicted Rainfall", "linear-gradient(135deg,rgba(70,120,210,0.30),rgba(255,255,255,0.12))")
with c5:
    top_region = day_df.assign(_metric=pd.to_numeric(day_df[TARGET_CFG["Rainfall (mm)"]["pred_col"]], errors="coerce")).sort_values("_metric", ascending=False).iloc[0]["region_id"] if not day_df.empty else "-"
    render_metric_card("Wettest Region", str(top_region), "Highest Predicted Rainfall", "linear-gradient(135deg,rgba(30,100,200,0.28),rgba(255,255,255,0.12))", compact=True)

tab_ov, tab_co, tab_ex, tab_re = st.tabs(["Dashboard", "Combined Map", "Variable Explorer", "Location Detail"])

with tab_ov:
    left, right = st.columns([1.45, 1.0])
    with left:
        st.markdown("<div class='section-label'>Weather Overview</div>", unsafe_allow_html=True)
        st.plotly_chart(combined_fig, use_container_width=True, config=PLOTLY_CONFIG, key=f"ov_{selected_date}_{selected_region}")
        st.markdown("<div class='small-note'>Temperature is the primary smoothed land surface. Green badges show humidity values and blue badges show rainfall values directly on the map.</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='section-label'>Location Snapshot</div>", unsafe_allow_html=True)
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            render_metric_card("Temperature", f"{region_row['pred_t2m_mean_tplus1']:.1f}°C" if region_row is not None else "-", "Predicted Temperature", "linear-gradient(135deg,rgba(239,100,60,0.30),rgba(255,255,255,0.12))")
        with s2:
            if region_row is not None:
                rfl = compute_heat_index(float(region_row['pred_t2m_mean_tplus1']), float(region_row['pred_rh2m_mean_tplus1']))
                rfl_lbl, _ = heat_comfort_label(float(region_row['pred_t2m_mean_tplus1']), rfl, float(region_row['pred_rh2m_mean_tplus1']))
                render_metric_card("Feels Like", f"{rfl:.1f}°C" if rfl is not None else "-", rfl_lbl, "linear-gradient(135deg,rgba(249,168,77,0.30),rgba(255,255,255,0.12))")
            else:
                render_metric_card("Feels Like", "-", "Heat Index", "linear-gradient(135deg,rgba(249,168,77,0.30),rgba(255,255,255,0.12))")
        with s3:
            render_metric_card("Humidity", f"{region_row['pred_rh2m_mean_tplus1']:.0f}%" if region_row is not None else "-", "Predicted RH", "linear-gradient(135deg,rgba(60,180,110,0.26),rgba(255,255,255,0.12))")
        with s4:
            render_metric_card("Rainfall", f"{region_row['pred_tp_sum_mm_tplus1']:.1f} mm" if region_row is not None else "-", "Predicted Rainfall", "linear-gradient(135deg,rgba(70,120,210,0.30),rgba(255,255,255,0.12))")

        st.markdown("<div class='section-label'>7-Day Outlook</div>", unsafe_allow_html=True)
        render_forecast_strip(pred_df, selected_region, pd.Timestamp(selected_date), horizon=7)
        st.markdown("<div class='section-label'>Advisories</div>", unsafe_allow_html=True)
        if region_row is not None:
            _fl = compute_heat_index(float(region_row['pred_t2m_mean_tplus1']), float(region_row['pred_rh2m_mean_tplus1']))
            _, _fl_desc = heat_comfort_label(float(region_row['pred_t2m_mean_tplus1']), _fl, float(region_row['pred_rh2m_mean_tplus1']))
            st.markdown(
                f'<div class="advice-card" style="border-left-color:#f9a84d;background:linear-gradient(145deg,rgba(249,168,77,0.24),rgba(20,60,120,0.20));"><div class="advice-title">🌡️ Feels Like {_fl:.1f}°C</div><div class="advice-body">{_fl_desc} <em>Actual: {float(region_row["pred_t2m_mean_tplus1"]):.1f}°C · Humidity: {float(region_row["pred_rh2m_mean_tplus1"]):.0f}% · Heat Index Formula (Rothfusz NWS)</em></div></div>',
                unsafe_allow_html=True,
            )
        for title, body in build_advisories(region_row, pred_df):
            st.markdown(f'<div class="advice-card"><div class="advice-title">{title}</div><div class="advice-body">{body}</div></div>', unsafe_allow_html=True)

    st.markdown("<div class='dashboard-bottom-gap'></div>", unsafe_allow_html=True)
    o1, o2 = st.columns(2)
    with o1:
        st.plotly_chart(make_region_timeseries(pred_df, selected_region, cfg["pred_col"], cfg["actual_col"], target_label), use_container_width=True, config=PLOTLY_CONFIG, key=f"trend_{selected_region}_{target_label}")
    with o2:
        rank_data = day_df.assign(_m=pd.to_numeric(day_df[cfg["pred_col"]], errors="coerce")).sort_values("_m", ascending=False)
        vals = rank_data["_m"].to_numpy()
        if len(vals) > 0:
            vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
            rng = max(vmax - vmin, 1e-6)
            bar_colors = [f"rgba({int(91 + (239 - 91) * n)},{int(155 + (168 - 155) * n)},{int(196 + (77 - 196) * n)},0.88)" for n in [(v - vmin) / rng for v in vals]]
        else:
            bar_colors = ACCENT_WARM
        rf = go.Figure()
        rf.add_trace(go.Bar(x=rank_data["region_id"], y=rank_data["_m"], marker=dict(color=bar_colors, line=dict(color="rgba(255,255,255,0.15)", width=0.5))))
        rf.update_layout(
            title=dict(
                text=f"Regional Ranking · {hero_date}",
                x=0.03,
                xanchor="left",
                font=dict(family="Fraunces, serif", color="white", size=16),
            ),
            xaxis=dict(title="Region", **PLOTLY_AXIS),
            yaxis=dict(title=target_label, **PLOTLY_AXIS),
            height=320,
            margin=dict(l=34, r=24, t=64, b=22),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=PLOTLY_FONT,
            hoverlabel=HOVER_STYLE,
        )
        st.plotly_chart(rf, use_container_width=True, config=PLOTLY_CONFIG, key=f"rank_{selected_date}_{target_label}")

with tab_co:
    st.markdown("<div class='section-label'>Temperature Surface · Humidity And Rainfall Overlays</div>", unsafe_allow_html=True)
    left, right = st.columns([1.55, 0.95])
    with left:
        st.plotly_chart(combined_fig, use_container_width=True, config=PLOTLY_CONFIG, key=f"tc_{selected_date}_{selected_region}")
    with right:
        st.markdown(
            """<div class=\"glass-card\">
                <h3>Map Legend</h3>
                <div class=\"legend-list\">
                    <div class=\"legend-item\"><span class=\"legend-emoji\">🌡️</span><span class=\"legend-text\">Temperature renders as a smoothed land surface.</span></div>
                    <div class=\"legend-item\"><span class=\"legend-emoji\">🟢</span><span class=\"legend-text\">Green badges show each region’s relative humidity value.</span></div>
                    <div class=\"legend-item\"><span class=\"legend-emoji\">🔵</span><span class=\"legend-text\">Blue badges show each region’s rainfall value.</span></div>
                    <div class=\"legend-item\"><span class=\"legend-emoji\">✨</span><span class=\"legend-text\">Labels are shown directly on the map for quick scanning.</span></div>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        tbl = day_df[["region_id", "pred_t2m_mean_tplus1", "pred_rh2m_mean_tplus1", "pred_tp_sum_mm_tplus1"]].copy()
        tbl.columns = ["Region", "Temp (°C)", "RH (%)", "Rain (mm)"]
        render_light_table(tbl.sort_values("Rain (mm)", ascending=False), numeric_cols=["Temp (°C)", "RH (%)", "Rain (mm)"], max_height=370)

with tab_ex:
    st.markdown("<div class='section-label'>Variable Explorer</div>", unsafe_allow_html=True)
    left, right = st.columns([1.25, 1.25])
    with left:
        st.plotly_chart(single_fig, use_container_width=True, config=PLOTLY_CONFIG, key=f"ex_{target_label}_{view_mode}_{color_mode}_{selected_date}_{render_mode}")
    with right:
        specs = [
            ("region_id", "Region"),
            (cfg["pred_col"], f"{target_label} Pred"),
            (cfg["actual_col"], f"{target_label} Actual"),
            (cfg["err_col"], f"{target_label} Error"),
            ("pred_t2m_mean_tplus1", "Temp (°C)"),
            ("pred_rh2m_mean_tplus1", "RH (%)"),
            ("pred_tp_sum_mm_tplus1", "Rain (mm)"),
        ]
        ss, so, fc, rm = set(), set(), [], {}
        for src, out in specs:
            if src not in day_df.columns or src in ss or out in so:
                continue
            ss.add(src)
            so.add(out)
            fc.append(src)
            rm[src] = out
        table = day_df[fc].copy().rename(columns=rm)
        sort_name = f"{target_label} Pred" if view_mode_norm == "best-model forecast" else f"{target_label} Actual" if view_mode_norm == "actual next-day value" else f"{target_label} Error"
        render_light_table(table.sort_values(sort_name, ascending=False), numeric_cols=[c for c in table.columns if c != "Region"], max_height=560)

with tab_re:
    st.markdown(f"<div class='section-label'>Location Detail · {selected_region}</div>", unsafe_allow_html=True)
    r1, r2, r3 = st.columns(3)
    with r1:
        st.plotly_chart(make_single_metric_history(pred_df, selected_region, "pred_t2m_mean_tplus1", "Temperature", "°C", ACCENT_WARM), use_container_width=True, config=PLOTLY_CONFIG, key=f"rt_{selected_region}")
    with r2:
        st.plotly_chart(make_single_metric_history(pred_df, selected_region, "pred_rh2m_mean_tplus1", "Humidity", "%", ACCENT_GREEN), use_container_width=True, config=PLOTLY_CONFIG, key=f"rh_{selected_region}")
    with r3:
        st.plotly_chart(make_single_metric_history(pred_df, selected_region, "pred_tp_sum_mm_tplus1", "Rainfall", "mm", ACCENT_RAIN), use_container_width=True, config=PLOTLY_CONFIG, key=f"rr_{selected_region}")
    st.markdown("<div class='small-note'>All values are sourced directly from the forecast output tables.</div>", unsafe_allow_html=True)
