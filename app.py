# -*- coding: utf-8 -*-
"""
RoadShield AI — Road Accident Severity Predictor
Dark-mode single-page Streamlit app with UX-optimised input grouping
"""

import os, warnings
import streamlit as st
import pandas as pd
import numpy as np
import joblib, shap
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RoadShield AI | Severity Predictor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────
# CSS — consistent dark mode (no white leaks)
# ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');

/* ─ Global reset ─ */
html, body, [class*="css"],
.stApp, .main, .main .block-container,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
section[data-testid="stSidebar"],
div[data-testid="stDecoration"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #080c18 !important;
    color: #cbd5e1 !important;
}
.main .block-container {
    padding: 1rem 2rem 3rem !important;
    max-width: 1350px !important;
}
#MainMenu, footer, header { visibility: hidden !important; }
.stDeployButton { display: none !important; }

/* ─ Sidebar ─ */
[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #1e293b !important;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

/* ─ ALL widget backgrounds ─ */
.stSelectbox > div > div,
.stSelectbox div[data-baseweb="select"],
.stSelectbox div[data-baseweb="select"] > div,
.stMultiSelect div[data-baseweb="select"],
.stMultiSelect div[data-baseweb="select"] > div,
.stTextInput input, .stNumberInput input,
.stTextArea textarea,
div[data-baseweb="popover"],
div[data-baseweb="menu"],
div[data-baseweb="popover"] ul {
    background-color: #111827 !important;
    border-color: #1e293b !important;
    color: #f1f5f9 !important;
}
.stSelectbox label, .stSlider label, .stCheckbox label,
.stNumberInput label, .stTextInput label {
    color: #94a3b8 !important;
    font-size: .83rem !important;
    font-weight: 600 !important;
}
/* Slider track */
.stSlider div[data-baseweb="slider"] div { background: #1e293b !important; }
/* Checkbox */
.stCheckbox span[data-baseweb="checkbox"] { background: #111827 !important; border-color: #334155 !important; }

/* ─ Tabs / expander / info boxes ─ */
.stExpander, .streamlit-expanderContent,
.stAlert, div[data-testid="stExpander"],
div[role="alert"] {
    background-color: #0d1117 !important;
    border-color: #1e293b !important;
    color: #cbd5e1 !important;
}
.stSpinner > div { color: #a5b4fc !important; }

/* ─ Hero ─ */
.hero {
    background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
    padding: 2rem 2.5rem; border-radius: 18px; color: #f1f5f9;
    margin-bottom: 1.8rem;
    border: 1px solid #312e81;
    box-shadow: 0 10px 40px rgba(99,102,241,.15);
}
.hero h1 { font-size: 2.1rem; font-weight: 800; margin: 0 0 .3rem;
           letter-spacing: -.5px; color: #f1f5f9 !important; }
.hero p  { font-size: 1rem; opacity: .7; margin: 0; font-weight: 300;
           color: #94a3b8 !important; }
.badge { display:inline-flex; align-items:center; gap:.5rem;
         background:rgba(99,102,241,.2); border:1px solid #4338ca;
         padding:.35rem 1rem; border-radius:50px; font-size:.8rem;
         font-weight:600; margin-top:.8rem; color:#a5b4fc !important; }

/* ─ Section headers ─ */
.sec-head { font-size:1.05rem; font-weight:700; color:#f1f5f9 !important;
            margin-bottom:.8rem; padding-bottom:.45rem;
            border-bottom:2px solid #4338ca; }

/* ─ Input cards ─ */
.card { background:#0d1117; border-radius:14px; padding:1.4rem;
        margin-bottom:1.1rem; border:1px solid #1e293b; }

/* ─ Group label (pre/post accident) ─ */
.group-label { font-size:.68rem; font-weight:700; color:#475569 !important;
               text-transform:uppercase; letter-spacing:2px;
               margin: 1.2rem 0 .5rem; padding-left:.2rem; }

/* ─ Result card ─ */
.res-card { border-radius:16px; padding:2rem 1.6rem; text-align:center;
            color:#fff; margin:1rem 0; border:2px solid; }
.res-slight  { background:#052e16; border-color:#22c55e; }
.res-serious { background:#1c1003; border-color:#f59e0b; }
.res-fatal   { background:#1f0000; border-color:#ef4444; }
.res-lbl  { font-size:.72rem; font-weight:700; text-transform:uppercase;
            letter-spacing:2.5px; color:#94a3b8 !important; margin-bottom:.5rem; }
.res-sev  { font-size:2rem; font-weight:900; letter-spacing:-.3px; }
.res-conf { font-size:.9rem; color:#94a3b8 !important; margin-top:.3rem; }
.res-dot  { width:14px; height:14px; border-radius:50%;
            display:inline-block; margin-bottom:.7rem; }

/* ─ Alerts ─ */
.alert-fatal   { background:#1f0000; border:1px solid #7f1d1d; color:#fca5a5 !important;
                 border-radius:10px; padding:.85rem 1.2rem; margin:.6rem 0;
                 font-size:.86rem; font-weight:600; }
.alert-serious { background:#1c1003; border:1px solid #92400e; color:#fcd34d !important;
                 border-radius:10px; padding:.85rem 1.2rem; margin:.6rem 0;
                 font-size:.86rem; font-weight:600; }

/* ─ Probability bars ─ */
.prob-wrap { background:#0d1117; border-radius:12px; padding:1.2rem;
             margin:.7rem 0; border:1px solid #1e293b; }
.prob-bar  { margin:.6rem 0; }
.prob-lbl  { font-size:.82rem; font-weight:600; color:#94a3b8 !important; margin-bottom:.2rem; }
.prob-track{ background:#1e293b; border-radius:8px; height:24px; overflow:hidden; }
.prob-fill { height:100%; border-radius:8px; display:flex; align-items:center;
             justify-content:flex-end; padding-right:8px; font-size:.72rem;
             font-weight:700; color:#fff; min-width:36px; }
.pf-slight  { background:linear-gradient(90deg,#166534,#22c55e); }
.pf-serious { background:linear-gradient(90deg,#92400e,#f59e0b); }
.pf-fatal   { background:linear-gradient(90deg,#991b1b,#ef4444); }

/* ─ Summary table ─ */
.sum-wrap { background:#0d1117; border-radius:14px; padding:1.3rem;
            border:1px solid #1e293b; }
.sum-row  { display:flex; justify-content:space-between; align-items:center;
            padding:.4rem 0; border-bottom:1px solid #1e293b; }
.sum-row:last-child { border-bottom:none; }
.sum-k { font-weight:500; color:#64748b !important; font-size:.83rem; }
.sum-v { font-weight:700; color:#f1f5f9 !important; font-size:.85rem; }

/* ─ Divider ─ */
.divider { height:1px; border:none; margin:1.4rem 0; background:#1e293b; }

/* ─ SHAP ─ */
.chart-title { font-size:.72rem; font-weight:700; color:#475569 !important;
               text-transform:uppercase; letter-spacing:2px; margin-bottom:.6rem; }
.shap-summary { color:#94a3b8 !important; font-size:.9rem; line-height:1.75;
                padding:1rem 1.2rem; background:#0a0f1e;
                border-radius:10px; border:1px solid #1e293b; margin-bottom:1.2rem; }
.shap-card { background:#0a0f1e; border:1px solid #1e293b;
             border-radius:10px; padding:.85rem 1rem; margin-bottom:.5rem; }
.shap-feat { font-size:.9rem; font-weight:700; color:#f1f5f9 !important; margin-bottom:.4rem; }
.shap-bar-bg { background:#1e293b; border-radius:4px; height:5px; margin:.35rem 0; }
.shap-val  { font-size:.78rem; font-family:'JetBrains Mono',monospace; }

/* ─ Button ─ */
.stButton > button {
    background: linear-gradient(135deg,#4f46e5,#7c3aed) !important;
    color:#fff !important; border:none !important;
    font-weight:700 !important; font-size:1rem !important;
    border-radius:12px !important; padding:.7rem 2rem !important;
    box-shadow:0 6px 22px rgba(99,102,241,.35) !important;
    width:100% !important;
}
.stButton > button:hover { box-shadow:0 10px 30px rgba(99,102,241,.5) !important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
# PATHS & CONSTANTS
# ──────────────────────────────────────────────────────────
MODEL_PATH    = os.path.join("model", "best_model(1).pkl")
ENCODERS_PATH = os.path.join("model", "label_encoders(1).pkl")
SCALER_PATH   = os.path.join("model", "scaler(1).pkl")

SEVERITY_MAP = {0: "Fatal Injury", 1: "Serious Injury", 2: "Slight Injury"}
SEV_COLOR    = {"Fatal Injury":"#ef4444","Serious Injury":"#f59e0b","Slight Injury":"#22c55e"}
SEV_CSS      = {"Fatal Injury":"fatal","Serious Injury":"serious","Slight Injury":"slight"}
SEV_EMOJI    = {"Fatal Injury":"🔴","Serious Injury":"🟠","Slight Injury":"🟢"}
SEV_RISK     = {"Fatal Injury":"Critical","Serious Injury":"High","Slight Injury":"Low"}
SEV_IDX      = {"Fatal Injury":0,"Serious Injury":1,"Slight Injury":2}

# ── UX-optimised feature groups ──
# PRE-ACCIDENT: User realistically knows these before/during driving
# These are the conditions that LEAD to an accident
DISPLAY = {
    "Day_of_week":"📅 Day of Week","Age_band_of_driver":"👤 Driver Age Group",
    "Sex_of_driver":"⚧ Sex of Driver","Vehicle_driver_relation":"🔗 Driver-Vehicle Relation",
    "Driving_experience":"🎓 Driving Experience","Type_of_vehicle":"🚗 Vehicle Type",
    "Owner_of_vehicle":"📋 Vehicle Ownership","Defect_of_vehicle":"🔧 Known Vehicle Defect",
    "Area_accident_occured":"📍 Area Type","Lanes_or_Medians":"🛣️ Road Lane Type",
    "Road_allignment":"🔀 Road Alignment","Types_of_Junction":"⚡ Junction Type",
    "Road_surface_type":"🛤️ Road Surface Material","Road_surface_conditions":"💧 Road Surface Condition",
    "Light_conditions":"💡 Lighting","Weather_conditions":"🌦️ Weather",
    "Type_of_collision":"💥 Collision Type","Number_of_vehicles_involved":"🚙 Vehicles Involved",
    "Vehicle_movement":"➡️ Vehicle Movement","Fitness_of_casuality":"🏥 Casualty Fitness",
    "Pedestrian_movement":"🚶 Pedestrian Movement","Cause_of_accident":"⚠️ Cause of Accident",
}

DEFAULTS = {
    "Day_of_week":"Monday","Age_band_of_driver":"18-30","Sex_of_driver":"Male",
    "Vehicle_driver_relation":"Owner","Driving_experience":"2-5yr",
    "Type_of_vehicle":"Automobile","Owner_of_vehicle":"Owner",
    "Defect_of_vehicle":"No defect","Area_accident_occured":"Other",
    "Lanes_or_Medians":"Undivided Two way","Road_allignment":"Tangent road with flat terrain",
    "Types_of_Junction":"No junction","Road_surface_type":"Asphalt roads",
    "Road_surface_conditions":"Dry","Light_conditions":"Daylight",
    "Weather_conditions":"Normal","Type_of_collision":"Vehicle with vehicle collision",
    "Number_of_vehicles_involved":2,"Vehicle_movement":"Going straight",
    "Fitness_of_casuality":"Normal","Pedestrian_movement":"Not a Pedestrian",
    "Cause_of_accident":"Driving carelessly",
}

# ══ UX-OPTIMISED FEATURE GROUPS ══
# BASIC MODE — 8 questions a user realistically knows before an accident
BASIC_DRIVER  = ["Age_band_of_driver", "Driving_experience"]           # 2
BASIC_VEHICLE = ["Type_of_vehicle", "Defect_of_vehicle"]               # 2
BASIC_ENV     = ["Light_conditions", "Weather_conditions",
                 "Road_surface_conditions", "Road_allignment"]          # 4
# Total basic = 8 questions

# ADVANCED — remaining features, shown in an extra section when toggled
ADVANCED_FEATS = [
    # Driver extras
    "Sex_of_driver", "Vehicle_driver_relation", "Fitness_of_casuality",
    # Vehicle extras
    "Owner_of_vehicle", "Number_of_vehicles_involved",
    # Environment extras
    "Day_of_week", "Road_surface_type", "Lanes_or_Medians",
    "Types_of_Junction", "Area_accident_occured",
    # Post-incident details
    "Cause_of_accident", "Type_of_collision",
    "Vehicle_movement", "Pedestrian_movement",
]

SCENARIOS = {
    "🎯 Custom (Manual Input)": {},
    "🏙️ Urban Daytime Commute": {
        "Light_conditions":"Daylight","Weather_conditions":"Normal",
        "Area_accident_occured":"Residential areas","Road_surface_conditions":"Dry",
        "Lanes_or_Medians":"Two-way (divided with solid lines road marking)",
        "Types_of_Junction":"Crossing","Driving_experience":"5-10yr",
        "Type_of_vehicle":"Automobile","Road_surface_type":"Asphalt roads",
        "Road_allignment":"Tangent road with flat terrain",
    },
    "🌙 Highway Night Drive": {
        "Light_conditions":"Darkness - no lighting","Weather_conditions":"Normal",
        "Area_accident_occured":"Outside rural areas","Road_surface_conditions":"Dry",
        "Road_allignment":"Tangent road with flat terrain","Lanes_or_Medians":"Undivided Two way",
        "Types_of_Junction":"No junction","Driving_experience":"Above 10yr",
        "Type_of_vehicle":"Automobile",
    },
    "🌧️ Rainy Evening": {
        "Weather_conditions":"Raining","Road_surface_conditions":"Wet or damp",
        "Light_conditions":"Darkness - lights lit","Road_surface_type":"Asphalt roads with some distress",
        "Road_allignment":"Gentle horizontal curve","Driving_experience":"1-2yr",
        "Area_accident_occured":"Other",
    },
    "⛰️ Mountain / Foggy Road": {
        "Road_allignment":"Steep grade downward with mountainous terrain",
        "Light_conditions":"Daylight","Weather_conditions":"Fog or mist",
        "Road_surface_type":"Asphalt roads with some distress",
        "Road_surface_conditions":"Wet or damp","Lanes_or_Medians":"Undivided Two way",
        "Driving_experience":"5-10yr","Area_accident_occured":"Outside rural areas",
        "Types_of_Junction":"No junction",
    },
    "🚶 Pedestrian Zone": {
        "Area_accident_occured":"Office areas","Types_of_Junction":"Crossing",
        "Light_conditions":"Daylight","Weather_conditions":"Normal",
        "Road_surface_conditions":"Dry","Lanes_or_Medians":"Two-way (divided with solid lines road marking)",
        "Pedestrian_movement":"Crossing from nearside - making a loss of loss",
        "Driving_experience":"2-5yr","Type_of_vehicle":"Automobile",
    },
}

# ──────────────────────────────────────────────────────────
# CACHED LOADERS
# ──────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_encoders():
    return joblib.load(ENCODERS_PATH)

@st.cache_resource
def load_scaler():
    try:    return joblib.load(SCALER_PATH)
    except: return None

@st.cache_resource
def get_explainer(_model):
    try:    return shap.TreeExplainer(_model)
    except: return None

# ──────────────────────────────────────────────────────────
# ML CORE
# ──────────────────────────────────────────────────────────
def preprocess(user_dict, encoders, scaler, feature_order):
    row = {}
    for f in feature_order:
        v = user_dict.get(f)
        if f in encoders:
            try:    row[f] = int(encoders[f].transform([v])[0])
            except: row[f] = 0
        else:
            row[f] = int(v) if v is not None else 2
    df = pd.DataFrame([row], columns=feature_order)
    if scaler:
        try:    df = pd.DataFrame(scaler.transform(df), columns=feature_order)
        except: pass
    return df

def predict(model, X):
    pred  = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    if len(proba) < 3:
        fp = np.zeros(3); fp[:len(proba)] = proba; proba = fp
    return SEVERITY_MAP.get(pred, str(pred)), proba

def compute_shap(explainer, X_raw, pred_idx):
    """Returns (list[float], float) or (None, error_str)."""
    try:
        sv = explainer.shap_values(X_raw)
        ev = explainer.expected_value
        if isinstance(sv, list):
            row  = np.array(sv[pred_idx], dtype=float).ravel()
            base = float(ev[pred_idx]) if hasattr(ev,"__len__") else float(ev)
        else:
            sv_np = np.array(sv, dtype=float)
            if sv_np.ndim == 3:
                # SHAP 0.51+: (n_samples, n_features, n_classes)
                row  = sv_np[0, :, pred_idx]
                base = float(ev[pred_idx]) if hasattr(ev,"__len__") else float(ev)
            else:
                row  = sv_np[0]
                base = float(ev) if not hasattr(ev,"__len__") else float(ev[0])
        return [float(x) for x in row], float(base)
    except Exception as e:
        return None, str(e)

# ──────────────────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────────────────
def init_state():
    if "ci" not in st.session_state:
        st.session_state.ci = dict(DEFAULTS)
    if "active_sc" not in st.session_state:
        st.session_state.active_sc = "🎯 Custom (Manual Input)"

def apply_scenario(name):
    merged = dict(DEFAULTS)
    merged.update(SCENARIOS.get(name, {}))
    st.session_state.ci = merged
    st.session_state.active_sc = name

# ══════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        st.markdown(
            "<h2 style='color:#f1f5f9!important;font-size:1.25rem;margin:0 0 .2rem;'>"
            "🎛️ Control Panel</h2>",
            unsafe_allow_html=True,
        )
        st.markdown("<hr style='border:none;border-top:1px solid #1e293b;margin:.5rem 0 1rem;'>",
                    unsafe_allow_html=True)

        # Scenario
        st.markdown(
            "<p style='color:#64748b!important;font-size:.68rem;font-weight:700;"
            "text-transform:uppercase;letter-spacing:1.5px;margin-bottom:.4rem;'>📋 Quick Scenarios</p>",
            unsafe_allow_html=True,
        )
        sc_names = list(SCENARIOS.keys())
        chosen = st.selectbox(
            "Scenario", sc_names,
            index=sc_names.index(st.session_state.active_sc) if st.session_state.active_sc in sc_names else 0,
            label_visibility="collapsed",
        )
        if chosen != st.session_state.active_sc:
            apply_scenario(chosen)
            st.rerun()

        if chosen != "🎯 Custom (Manual Input)":
            label = chosen.split(" ", 1)[1] if " " in chosen else chosen
            st.markdown(
                f"<div style='background:rgba(99,102,241,.1);border-radius:8px;"
                f"padding:.5rem .9rem;margin:.4rem 0;border:1px solid rgba(99,102,241,.2);'>"
                f"<small style='color:#64748b!important;'>Auto-filled for "
                f"<b style='color:#a5b4fc!important;'>{label}</b></small></div>",
                unsafe_allow_html=True,
            )

        st.markdown("<hr style='border:none;border-top:1px solid #1e293b;margin:.9rem 0;'>",
                    unsafe_allow_html=True)

        # Advanced toggle
        st.markdown(
            "<p style='color:#64748b!important;font-size:.68rem;font-weight:700;"
            "text-transform:uppercase;letter-spacing:1.5px;margin-bottom:.4rem;'>⚙️ Input Detail Level</p>",
            unsafe_allow_html=True,
        )
        advanced = st.checkbox(
            "Include Incident Details",
            value=False,
            help="Adds post-accident fields (cause, collision type, movements) — "
                 "useful if you're analysing an incident that already occurred.",
        )

        st.markdown("<hr style='border:none;border-top:1px solid #1e293b;margin:.9rem 0;'>",
                    unsafe_allow_html=True)

        # Model info
        st.markdown(
            "<p style='color:#64748b!important;font-size:.68rem;font-weight:700;"
            "text-transform:uppercase;letter-spacing:1.5px;margin-bottom:.5rem;'>📊 Model Info</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div style='background:#0a0f1e;border-radius:10px;padding:.85rem 1rem;"
            "border:1px solid #1e293b;'>"
            "<table style='width:100%;border-collapse:collapse;'>"
            "<tr><td style='color:#475569!important;font-size:.78rem;padding:.2rem 0;'>Algorithm</td>"
            "<td style='color:#a5b4fc!important;font-size:.78rem;font-weight:600;text-align:right;'>Random Forest</td></tr>"
            "<tr><td style='color:#475569!important;font-size:.78rem;padding:.2rem 0;'>Features</td>"
            "<td style='color:#a5b4fc!important;font-size:.78rem;font-weight:600;text-align:right;'>22</td></tr>"
            "<tr><td style='color:#475569!important;font-size:.78rem;padding:.2rem 0;'>Accuracy</td>"
            "<td style='color:#a5b4fc!important;font-size:.78rem;font-weight:600;text-align:right;'>82.63%</td></tr>"
            "<tr><td style='color:#475569!important;font-size:.78rem;padding:.2rem 0;'>F1 Weighted</td>"
            "<td style='color:#a5b4fc!important;font-size:.78rem;font-weight:600;text-align:right;'>0.7931</td></tr>"
            "<tr><td style='color:#475569!important;font-size:.78rem;padding:.2rem 0;'>ROC AUC</td>"
            "<td style='color:#a5b4fc!important;font-size:.78rem;font-weight:600;text-align:right;'>0.6387</td></tr>"
            "</table></div>",
            unsafe_allow_html=True,
        )

        st.markdown("<hr style='border:none;border-top:1px solid #1e293b;margin:.9rem 0;'>",
                    unsafe_allow_html=True)
        st.markdown(
            "<div style='text-align:center;padding:.3rem 0;'>"
            "<span style='color:#334155!important;font-size:.72rem;'>Group 3 · Fr. CRIT · 2025-26</span>"
            "</div>",
            unsafe_allow_html=True,
        )

    return chosen, advanced

# ══════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════
def render_header():
    st.markdown("""
    <div class="hero">
        <h1>🛡️ RoadShield AI</h1>
        <p>AI-powered road accident severity prediction · Machine Learning + Explainable AI</p>
        <div class="badge">🌲 Random Forest &nbsp;·&nbsp; Accuracy 82.6% &nbsp;·&nbsp; F1 0.793</div>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# INPUT WIDGETS
# ══════════════════════════════════════════════════════════
def make_widget(feat, encoders, ci):
    """Render a selectbox/slider. Returns current value."""
    label = DISPLAY.get(feat, feat)
    cur   = ci.get(feat, DEFAULTS.get(feat))
    if feat in encoders:
        opts = list(encoders[feat].classes_)
        idx  = opts.index(cur) if cur in opts else 0
        return st.selectbox(label, opts, index=idx)
    return st.slider(label, 1, 7, int(cur) if cur else 2)

def render_inputs(encoders, advanced):
    ci  = st.session_state.ci
    new = {}

    # ── BASIC: 8 questions across 3 columns ──
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="sec-head">🧑‍✈️ Driver</div>', unsafe_allow_html=True)
        for f in BASIC_DRIVER:
            new[f] = make_widget(f, encoders, ci)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="sec-head">🚗 Vehicle</div>', unsafe_allow_html=True)
        for f in BASIC_VEHICLE:
            new[f] = make_widget(f, encoders, ci)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="sec-head">🌍 Conditions</div>', unsafe_allow_html=True)
        for f in BASIC_ENV:
            new[f] = make_widget(f, encoders, ci)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── ADVANCED: remaining 14 features ──
    if advanced:
        st.markdown('<div class="group-label">📋 Additional Details (Driver · Vehicle · Road · Incident)</div>',
                    unsafe_allow_html=True)
        st.caption("Extra fields for detailed analysis — includes post-accident details like cause and collision type.")
        adv_cols = st.columns(3)
        for i, f in enumerate(ADVANCED_FEATS):
            with adv_cols[i % 3]:
                new[f] = make_widget(f, encoders, ci)
    else:
        for f in ADVANCED_FEATS:
            new[f] = ci.get(f, DEFAULTS.get(f))

    # Persist
    for f, v in new.items():
        ci[f] = v
    return ci

# ══════════════════════════════════════════════════════════
# PREDICTION RESULT
# ══════════════════════════════════════════════════════════
def render_result(severity, proba, ci):
    css   = SEV_CSS[severity]
    color = SEV_COLOR[severity]
    risk  = SEV_RISK[severity]
    conf  = max(proba) * 100

    # ── Result card ──
    st.markdown(f"""
    <div class="res-card res-{css}">
        <div class="res-dot" style="background:{color};box-shadow:0 0 14px {color};"></div>
        <div class="res-lbl">Predicted Severity</div>
        <div class="res-sev" style="color:{color};">{severity}</div>
        <div class="res-conf">Risk Level: {risk} &nbsp;·&nbsp; Confidence: {conf:.1f}%</div>
    </div>""", unsafe_allow_html=True)

    if severity == "Fatal Injury":
        st.markdown('<div class="alert-fatal">⚠️ FATAL RISK — Conditions indicate extremely high severity. Immediate intervention recommended.</div>', unsafe_allow_html=True)
    elif severity == "Serious Injury":
        st.markdown('<div class="alert-serious">⚠️ SERIOUS RISK — Emergency response may be required for these conditions.</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Probability bars + Severity gauge + Summary ──
    pc1, pc2 = st.columns(2)

    with pc1:
        st.markdown('<div class="chart-title">Class Probability</div>', unsafe_allow_html=True)
        labels_p = ["Fatal Injury","Serious Injury","Slight Injury"]
        colors_p = ["#ef4444","#f59e0b","#22c55e"]
        vals_p   = [proba[0], proba[1], proba[2]]
        fig_p = go.Figure(go.Bar(
            x=[f"{v*100:.1f}%" for v in vals_p],
            y=labels_p,
            orientation="h",
            marker_color=colors_p,
            text=[f"{v*100:.1f}%" for v in vals_p],
            textposition="outside",
            textfont=dict(color="#f1f5f9"),
        ))
        fig_p.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1", family="Inter"),
            height=200, margin=dict(l=10,r=50,t=10,b=10),
            xaxis=dict(showgrid=False, showticklabels=False, range=[0,115]),
            yaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig_p, use_container_width=True, config={"displayModeBar":False})

    with pc2:
        st.markdown('<div class="chart-title">Severity Gauge</div>', unsafe_allow_html=True)
        # Weighted gauge: Fatal=100, Serious=66, Slight=33
        gauge_val = proba[0]*100 + proba[1]*66 + proba[2]*33
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(gauge_val, 1),
            domain={"x":[0,1],"y":[0,1]},
            gauge={
                "axis":{"range":[0,100],"tickcolor":"#64748b"},
                "bar":{"color":color},
                "steps":[
                    {"range":[0,33],"color":"#052e16"},
                    {"range":[33,66],"color":"#1c1003"},
                    {"range":[66,100],"color":"#1f0000"},
                ],
                "threshold":{"line":{"color":color,"width":3},"value":gauge_val},
            },
            number={"font":{"color":color,"size":30}},
        ))
        fig_g.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1"), height=200,
            margin=dict(l=20,r=20,t=20,b=10),
        )
        st.plotly_chart(fig_g, use_container_width=True, config={"displayModeBar":False})

    # ── Summary table ──
    st.markdown("#### 📋 Prediction Summary")
    rows = {
        "🎯 Severity":   severity,
        "⚡ Risk Level": risk,
        "📊 Confidence": f"{conf:.1f}%",
        "👤 Driver Age": ci.get("Age_band_of_driver","—"),
        "🎓 Experience": ci.get("Driving_experience","—"),
        "💡 Lighting":   ci.get("Light_conditions","—"),
        "🌦️ Weather":   ci.get("Weather_conditions","—"),
        "💧 Road":       ci.get("Road_surface_conditions","—"),
    }
    html = '<div class="sum-wrap">'
    for k, v in rows.items():
        html += f'<div class="sum-row"><span class="sum-k">{k}</span><span class="sum-v">{v}</span></div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# SHAP SECTION (reference-style: Plotly bar + risk/mitigating cards)
# ══════════════════════════════════════════════════════════
def render_shap(sv_list, bv, feature_order, severity):
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">🧠 Why This Prediction?</div>', unsafe_allow_html=True)

    feat_labels  = [DISPLAY.get(f, f.replace("_"," ").title()).lstrip("📅👤⚧🔗🎓🚗📋🔧📍🛣️🔀⚡🛤️💧💡🌦️💥🚙➡️🏥🚶⚠️ ") for f in feature_order]
    pairs        = list(zip(feat_labels, sv_list))
    pairs_sorted = sorted(pairs, key=lambda x: x[1])

    top_risk = [(k,v) for k,v in reversed(pairs_sorted) if v > 0][:4]
    top_safe = [(k,v) for k,v in pairs_sorted if v < 0][:4]

    # Summary text
    parts = []
    if top_risk:
        parts.append(f"<b>{top_risk[0][0]}</b> was the strongest factor pushing toward {severity}.")
    if top_safe:
        parts.append(f"<b>{top_safe[0][0]}</b> helped reduce the severity risk.")
    txt = " ".join(parts) if parts else "All features contributed roughly equally."
    st.markdown(f'<div class="shap-summary">{txt}</div>', unsafe_allow_html=True)

    # Risk + Mitigating factor cards
    rc1, rc2 = st.columns(2)
    with rc1:
        st.markdown('<div class="chart-title">🔴 Top Risk Factors</div>', unsafe_allow_html=True)
        if top_risk:
            for k, v in top_risk:
                bar_pct = min(int(abs(v) * 400), 100)
                st.markdown(f"""
                <div class="shap-card" style="border-left:3px solid #ef4444;">
                    <div class="shap-feat">{k}</div>
                    <div class="shap-bar-bg"><div style="background:#ef4444;width:{bar_pct}%;height:5px;border-radius:4px;"></div></div>
                    <span class="shap-val" style="color:#ef4444;">+{v:.4f}</span>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown('<p style="color:#475569;font-size:.85rem;">No significant risk factors detected.</p>',
                        unsafe_allow_html=True)

    with rc2:
        st.markdown('<div class="chart-title">🔵 Mitigating Factors</div>', unsafe_allow_html=True)
        if top_safe:
            for k, v in top_safe:
                bar_pct = min(int(abs(v) * 400), 100)
                st.markdown(f"""
                <div class="shap-card" style="border-left:3px solid #3b82f6;">
                    <div class="shap-feat">{k}</div>
                    <div class="shap-bar-bg"><div style="background:#3b82f6;width:{bar_pct}%;height:5px;border-radius:4px;"></div></div>
                    <span class="shap-val" style="color:#3b82f6;">{v:.4f}</span>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown('<p style="color:#475569;font-size:.85rem;">No strong mitigating factors found.</p>',
                        unsafe_allow_html=True)

    # Full SHAP horizontal bar (Plotly)
    st.markdown("<br>", unsafe_allow_html=True)
    all_names = [k for k, v in pairs_sorted]
    all_vals  = [v for k, v in pairs_sorted]
    bar_cols  = ["#ef4444" if v > 0 else "#3b82f6" for v in all_vals]

    fig = go.Figure(go.Bar(
        x=all_vals, y=all_names, orientation="h",
        marker_color=bar_cols,
        text=[f"{v:+.4f}" for v in all_vals],
        textposition="outside",
        textfont=dict(size=9, color="#94a3b8"),
    ))
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1", family="Inter"),
        height=520,
        title=dict(text=f"SHAP Values — {severity} Class",
                   font=dict(color="#f1f5f9", size=12)),
        margin=dict(l=10, r=70, t=40, b=10),
        xaxis=dict(showgrid=True, gridcolor="#1e293b",
                   zeroline=True, zerolinecolor="#334155", zerolinewidth=1),
        yaxis=dict(showgrid=False),
    )
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
def main():
    model    = load_model()
    encoders = load_encoders()
    scaler   = load_scaler()
    explainer = get_explainer(model)

    if model is None or encoders is None:
        st.error("❌ Model or encoder files missing from `model/` folder.")
        st.stop()

    feature_order = list(model.feature_names_in_)
    init_state()

    render_header()
    scenario, advanced = render_sidebar()
    user_inputs = render_inputs(encoders, advanced)

    # Predict button
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        go_btn = st.button("🛡️  Run RoadShield Prediction", use_container_width=True)

    if go_btn:
        with st.spinner("🔄 Analysing parameters…"):
            X = preprocess(user_inputs, encoders, scaler, feature_order)
            severity, proba = predict(model, X)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        render_result(severity, proba, user_inputs)

        # SHAP
        if explainer is not None:
            pred_idx  = SEV_IDX.get(severity, 0)
            X_raw     = preprocess(user_inputs, encoders, None, feature_order)
            sv_list, bv = compute_shap(explainer, X_raw, pred_idx)
            if sv_list is not None:
                render_shap(sv_list, bv, feature_order, severity)
            elif isinstance(bv, str):
                st.warning(f"SHAP error: {bv}")


if __name__ == "__main__":
    main()
