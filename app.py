"""
VetConnect AI — Professional Animal Welfare Platform
Market-ready Streamlit frontend | Local LLM (llama3.2 via Ollama)
"""

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from math import radians, cos, sin, sqrt, atan2
from datetime import datetime, date

from llm import ask_vet_ai, triage_symptoms, ask_medication_info, generate_health_summary, ask_first_aid
from data import get_vets, get_ngos, get_breed_risks, get_first_aid_quickref, get_vaccination_schedule
from emergency import get_all_emergencies, get_emergency_guide, get_critical_emergencies

# PAGE CONFIG

st.set_page_config(
    page_title="VetConnect AI",
    page_icon="v",
    layout="wide",
    initial_sidebar_state="expanded",
)

# THEME - Full custom CSS for a market-ready look

st.markdown("""
<style>
  /* Brand colours */
  :root {
    --green-dark:   #1a3d2b;
    --green-mid:    #2d6a4f;
    --green-light:  #52b788;
    --green-pale:   #d8f3dc;
    --blue-dark:    #1e40af;
    --blue-mid:     #2563eb;
    --blue-light:   #60a5fa;
    --blue-pale:    #dbeafe;
    --cream:        #fefae0;
    --amber:        #e9c46a;
    --amber-dark:   #c9993a;
    --red-soft:     #fde8e8;
    --red-strong:   #c0392b;
    --text-dark:    #0f172a;
    --text-mid:     #334155;
    --text-light:   #64748b;
    --border:       #e2e8f0;
    --card-shadow:  0 2px 12px rgba(0,0,0,0.07);
  }

  /* Animations */
  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
  }
  @keyframes slideInLeft {
    from { opacity: 0; transform: translateX(-12px); }
    to { opacity: 1; transform: translateX(0); }
  }
  @keyframes scaleUp {
    from { transform: scale(0.98); opacity: 0.8; }
    to { transform: scale(1); opacity: 1; }
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.85; }
  }

  /* Global */
  html, body, [class*="css"] {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background-color: #f0f9ff !important;
  }

  /* Hide Streamlit chrome */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 1.5rem; padding-bottom: 3rem; background-color: #f0f9ff; animation: fadeIn 0.5s ease-out; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #d4f1f9 0%, #4f9b7f 45%, #2d6a4f 100%) !important;
    border-right: none;
    animation: slideInLeft 0.6s ease-out;
  }
  [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
  [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] div { color: #ffffff !important; }
  [data-testid="stSidebar"] .stButton > button {
    background: #2d3748 !important;
    border: 2px solid #4a5568 !important;
    color: #ffffff !important;
    text-align: left !important;
    justify-content: flex-start !important;
    border-radius: 10px !important;
    margin-bottom: 6px;
    font-size: 0.88rem;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    font-weight: 600;
    box-shadow: 0 2px 4px rgba(0,0,0,0.3) !important;
    animation: fadeIn 0.5s ease-out backwards;
    padding: 0.6rem 1rem !important;
  }
  [data-testid="stSidebar"] .stButton > button:nth-child(1) { animation-delay: 0.1s; }
  [data-testid="stSidebar"] .stButton > button:nth-child(2) { animation-delay: 0.15s; }
  [data-testid="stSidebar"] .stButton > button:nth-child(3) { animation-delay: 0.2s; }
  [data-testid="stSidebar"] .stButton > button:nth-child(4) { animation-delay: 0.25s; }
  [data-testid="stSidebar"] .stButton > button:nth-child(n+5) { animation-delay: 0.3s; }
  [data-testid="stSidebar"] .stButton > button:hover {
    background: #4a5568 !important;
    border-color: #718096 !important;
    color: #ffffff !important;
    box-shadow: 0 6px 16px rgba(0,0,0,0.4) !important;
    transform: translateY(-2px);
  }
  [data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: #2563eb !important;
    border: 2px solid #1e40af !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    box-shadow: 0 4px 12px rgba(37,99,235,0.35) !important;
  }
  [data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
    background: #1e40af !important;
    box-shadow: 0 8px 20px rgba(30,64,175,0.4) !important;
    transform: translateY(-2px);
  }

  /* Sidebar Navigation Sections */
  .sidebar-section {
    margin-bottom: 1.2rem;
  }
  .sidebar-category {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 800;
    color: #cbd5e1;
    margin: 1rem 0 0.6rem 0;
    padding: 0.4rem 0.8rem;
    opacity: 0.9;
  }
  .sidebar-nav-group {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  /* Sidebar Selectbox Styling */
  [data-testid="stSidebar"] .stSelectbox > div > div > select {
    background: #2d3748 !important;
    color: #ffffff !important;
    border: 2px solid #4a5568 !important;
    border-radius: 10px !important;
    padding: 0.6rem 1rem !important;
    font-weight: 600;
  }

  /* Primary buttons */
  .stButton > button[kind="primary"] {
    background: var(--blue-mid) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600;
    letter-spacing: 0.02em;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 12px rgba(37,99,235,0.3) !important;
    animation: scaleUp 0.5s ease-out;
  }
  .stButton > button[kind="primary"]:hover {
    background: var(--blue-dark) !important;
    transform: translateY(-3px) scale(1.02);
    box-shadow: 0 8px 24px rgba(37,99,235,0.4) !important;
  }
  [data-testid="stSidebar"] .stButton > button[kind="secondary"] {
    background: #2d3748 !important;
    border: 2px solid #4a5568 !important;
    color: #ffffff !important;
    font-weight: 600 !important;
  }
  [data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
    background: #4a5568 !important;
    border-color: #718096 !important;
    color: #ffffff !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4) !important;
  }

  .stButton > button[kind="secondary"] {
    border-radius: 8px !important;
    border: 1.5px solid var(--border) !important;
    font-weight: 500;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    background: #ffffff !important;
    color: #334155 !important;
    animation: fadeIn 0.4s ease-out;
  }
  .stButton > button[kind="secondary"]:hover {
    border-color: var(--blue-light) !important;
    color: var(--blue-mid) !important;
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(37,99,235,0.15) !important;
  }

  /* Metric cards */
  [data-testid="stMetric"] {
    background: linear-gradient(135deg, #ffffff 0%, #f0f9ff 100%);
    border-radius: 12px;
    padding: 1.1rem 1.2rem;
    border: 1px solid #bfdbfe;
    box-shadow: 0 2px 8px rgba(37,99,235,0.08);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    animation: scaleUp 0.5s ease-out;
  }
  [data-testid="stMetric"]:hover {
    box-shadow: 0 6px 16px rgba(37,99,235,0.12);
    transform: translateY(-2px);
  }
  [data-testid="stMetricLabel"] { color: var(--text-light) !important; font-size: 0.82rem !important; font-weight: 500 !important; text-transform: uppercase; letter-spacing: 0.05em; }
  [data-testid="stMetricValue"] { color: var(--blue-dark) !important; font-size: 1.9rem !important; font-weight: 700 !important; }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 2px solid var(--border); }
  .stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; font-weight: 500; color: var(--text-mid); transition: all 0.25s ease-out; }
  .stTabs [aria-selected="true"] { background: var(--blue-pale) !important; color: var(--blue-dark) !important; border-bottom: 2px solid var(--blue-mid) !important; animation: fadeIn 0.3s ease-out; }

  /* Dataframes */
  [data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; border: 1px solid var(--border); animation: fadeIn 0.5s ease-out; }

  /* Inputs */
  .stTextInput > div > div > input,
  .stTextArea > div > div > textarea,
  .stSelectbox > div > div { 
    border-radius: 8px !important;
    transition: all 0.25s ease-out !important;
  }
  .stTextInput > div > div > input:focus,
  .stTextArea > div > div > textarea:focus,
  .stSelectbox > div > div:focus {
    box-shadow: 0 0 0 3px rgba(37,99,235,0.1) !important;
  }

  /* Custom components */

  .hero-badge {
    display: inline-block;
    background: var(--blue-pale);
    color: var(--blue-dark);
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 999px;
    border: 1px solid rgba(37,99,235,0.2);
    margin-bottom: 0.8rem;
  }

  .hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    color: var(--blue-dark);
    line-height: 1.15;
    margin-bottom: 0.5rem;
  }

  .hero-sub {
    font-size: 1.1rem;
    color: var(--text-mid);
    line-height: 1.6;
    margin-bottom: 1.5rem;
    max-width: 560px;
  }

  .feature-card {
    background: white;
    border-radius: 14px;
    padding: 1.4rem;
    border: 1.5px solid #bfdbfe;
    box-shadow: 0 2px 12px rgba(37,99,235,0.08);
    height: 100%;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    animation: scaleUp 0.5s ease-out;
  }
  .feature-card:hover {
    border-color: var(--blue-light);
    box-shadow: 0 12px 28px rgba(37,99,235,0.16);
    transform: translateY(-4px) scale(1.01);
  }
  .feature-card .icon { font-size: 1.8rem; margin-bottom: 0.6rem; }
  .feature-card h4 { font-size: 1rem; font-weight: 700; color: var(--blue-dark); margin: 0 0 0.3rem; }
  .feature-card p  { font-size: 0.85rem; color: var(--text-mid); margin: 0; line-height: 1.5; }

  .emergency-card {
    background: white;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    border: 1.5px solid #bfdbfe;
    margin-bottom: 8px;
    display: flex;
    align-items: flex-start;
    gap: 12px;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    animation: fadeIn 0.4s ease-out;
  }
  .emergency-card:hover { 
    border-color: var(--blue-light);
    box-shadow: 0 4px 12px rgba(37,99,235,0.12);
    transform: translateX(4px);
  }
  .emergency-card.critical { border-left: 4px solid #c0392b; }
  .emergency-card.high     { border-left: 4px solid #e9c46a; }

  .do-item {
    background: #f0faf4;
    border: 1px solid #bfdbfe;
    border-radius: 8px;
    padding: 9px 13px;
    margin: 5px 0;
    font-size: 0.9rem;
    color: #1a3d2b;
    line-height: 1.5;
  }
  .dont-item {
    background: #fff5f5;
    border: 1px solid #fed7d7;
    border-radius: 8px;
    padding: 9px 13px;
    margin: 5px 0;
    font-size: 0.9rem;
    color: #742a2a;
    line-height: 1.5;
  }
  .step-item {
    background: white;
    border: 1.5px solid #bfdbfe;
    border-left: 4px solid var(--blue-mid);
    border-radius: 0 10px 10px 0;
    padding: 11px 15px;
    margin: 6px 0;
    font-size: 0.91rem;
    color: var(--text-dark);
    line-height: 1.55;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    animation: slideInLeft 0.4s ease-out;
  }
  .step-item:hover {
    border-left-color: var(--blue-dark);
    box-shadow: 0 2px 8px rgba(37,99,235,0.1);
    transform: translateX(4px);
  }
  .call-vet-item {
    background: #fff5f5;
    border: 1px solid #fed7d7;
    border-radius: 8px;
    padding: 9px 13px;
    margin: 5px 0;
    font-size: 0.9rem;
    color: #c53030;
    line-height: 1.5;
  }

  .chat-user {
    background: var(--blue-pale);
    border-radius: 14px 14px 4px 14px;
    padding: 11px 15px;
    margin: 8px 0 8px 15%;
    font-size: 0.9rem;
    color: var(--blue-dark);
    line-height: 1.6;
    border: 1px solid #93c5fd;
  }
  .chat-ai {
    background: #ffffff;
    border-radius: 14px 14px 14px 4px;
    padding: 11px 15px;
    margin: 8px 15% 8px 0;
    font-size: 0.9rem;
    color: var(--text-dark);
    line-height: 1.6;
    border: 1px solid var(--border);
  }
  .chat-ai-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--blue-mid);
    margin-bottom: 5px;
  }

  .vet-card {
    background: white;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    border: 1.5px solid #bfdbfe;
    margin-bottom: 10px;
    box-shadow: 0 2px 8px rgba(37,99,235,0.08);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    animation: fadeIn 0.4s ease-out;
  }
  .vet-card:hover { 
    border-color: var(--blue-light);
    box-shadow: 0 6px 20px rgba(37,99,235,0.15);
    transform: translateY(-2px);
  }
  .vet-card .vet-name { font-size: 1rem; font-weight: 700; color: var(--blue-dark); }
  .vet-card .vet-meta { font-size: 0.83rem; color: var(--text-mid); margin-top: 3px; }
  .vet-card .vet-dist { font-size: 1.1rem; font-weight: 700; color: var(--blue-dark); text-align: right; }

  .badge {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 2px 9px;
    border-radius: 999px;
    letter-spacing: 0.04em;
    margin-right: 4px;
  }
  .badge-emergency { background: #fde8e8; color: #c0392b; border: 1px solid #f5c6cb; }
  .badge-gov       { background: #e8f5e9; color: #2e7d32; border: 1px solid #c8e6c9; }
  .badge-private   { background: var(--blue-soft); color: var(--blue-strong); border: 1px solid #bee3f8; }
  .badge-ngo       { background: #fff8e1; color: #f57f17; border: 1px solid #ffe082; }
  .badge-rating    { background: #fffde7; color: #f9a825; border: 1px solid #fff176; }

  .pet-profile-card {
    background: white;
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    border: 1.5px solid #bfdbfe;
    box-shadow: 0 2px 8px rgba(37,99,235,0.08);
    margin-bottom: 10px;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    animation: fadeIn 0.4s ease-out;
  }
  .pet-profile-card:hover {
    box-shadow: 0 6px 16px rgba(37,99,235,0.15);
    transform: translateY(-2px);
    border-color: var(--blue-light);
  }
  .pet-avatar {
    width: 48px; height: 48px;
    background: var(--blue-pale);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.4rem;
  }

  .reminder-item {
    background: white;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    border: 1.5px solid #bfdbfe;
    border-left: 4px solid var(--blue-mid);
    margin-bottom: 8px;
    box-shadow: 0 2px 8px rgba(37,99,235,0.08);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    animation: fadeIn 0.4s ease-out;
  }
  .reminder-item:hover {
    box-shadow: 0 6px 16px rgba(37,99,235,0.12);
    border-left-color: var(--blue-dark);
    transform: translateX(4px);
  }

  .triage-green  { background: #f0faf4; border: 2px solid #52b788; border-radius: 12px; padding: 1.2rem; }
  .triage-amber  { background: #fffbeb; border: 2px solid #e9c46a; border-radius: 12px; padding: 1.2rem; }
  .triage-red    { background: #fff5f5; border: 2px solid #c0392b; border-radius: 12px; padding: 1.2rem; }

  .section-header {
    font-size: 1.5rem;
    font-weight: 800;
    color: var(--blue-dark);
    margin-bottom: 0.2rem;
  }
  .section-sub {
    font-size: 0.92rem;
    color: var(--text-mid);
    margin-bottom: 1.4rem;
  }

  .window-banner {
    background: linear-gradient(135deg, #fff5f5, #fff0f0);
    border: 1px solid #fed7d7;
    border-radius: 10px;
    padding: 10px 16px;
    font-size: 0.88rem;
    color: #c0392b;
    font-weight: 500;
    margin: 12px 0;
  }

  .reassurance-box {
    background: var(--blue-pale);
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 0.9rem;
    color: var(--blue-dark);
    border: 1px solid #93c5fd;
    margin-top: 12px;
    font-style: italic;
  }

  .stats-row {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin: 1rem 0;
  }

  .page-header {
    padding-bottom: 1rem;
    margin-bottom: 1.5rem;
    border-bottom: 2px solid #bfdbfe;
    animation: fadeIn 0.5s ease-out;
  }

  .quick-action-btn > button {
    background: var(--blue-pale) !important;
    color: var(--blue-dark) !important;
    border: 1px solid #93c5fd !important;
    border-radius: 999px !important;
    font-size: 0.82rem !important;
    padding: 4px 14px !important;
    font-weight: 600 !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    animation: scaleUp 0.5s ease-out;
  }
  .quick-action-btn > button:hover {
    background: var(--blue-light) !important;
    color: white !important;
    border-color: var(--blue-light) !important;
    transform: scale(1.05);
    box-shadow: 0 4px 12px rgba(37,99,235,0.25) !important;
  }

  .pet-profile-card:hover {
    box-shadow: 0 6px 16px rgba(37,99,235,0.15);
    transform: translateY(-2px);
    border-color: var(--blue-light);
  }
    background: white;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    border: 1.5px solid #bfdbfe;
    box-shadow: 0 2px 8px rgba(37,99,235,0.08);
    margin-bottom: 10px;
  }

  .related-chip {
    display: inline-block;
    background: var(--blue-pale);
    color: var(--blue-dark);
    font-size: 0.78rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 999px;
    border: 1px solid #93c5fd;
    margin: 3px 3px 0 0;
    cursor: pointer;
  }
</style>
""", unsafe_allow_html=True)


# SESSION STATE

defaults = {
    "page": "home",
    "chat_history": [],
    "triage_result": None,
    "pet_profiles": [],
    "active_pet": None,
    "reminders": [],
    "animals_assisted": 2847,
    "emergency_cases": 341,
    "health_report": None,
    "selected_emergency": None,
    "abuse_reports": [],
    "breeder_reports": [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def nav(page: str):
    st.session_state.page = page
    st.rerun()


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return round(R * 2 * atan2(sqrt(a), sqrt(1 - a)), 2)


def get_active_pet() -> dict | None:
    if not st.session_state.active_pet:
        return None
    return next((p for p in st.session_state.pet_profiles if p["name"] == st.session_state.active_pet), None)


SPECIES_EMOJI = {"Dog": "D", "Cat": "C", "Rabbit": "R", "Bird": "B", "Guinea Pig": "G", "Other": "O"}
TYPE_BADGE = {"Government": "badge-gov", "Private": "badge-private", "NGO": "badge-ngo"}


# SIDEBAR

with st.sidebar:
    # App Header
    st.markdown("""
    <div style="padding: 1.2rem 0.5rem 1rem; text-align: center; animation: fadeIn 0.5s ease-out;">
      <div style="font-size:1.4rem; font-weight:800; color:#fff; letter-spacing:-0.02em;">🐾 VetConnect AI</div>
      <div style="font-size:0.65rem; color:#7dba98; text-transform:uppercase; letter-spacing:0.1em; margin-top:3px;">Smart Pet Healthcare</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='margin: 0.8rem 0; border: none; border-top: 2px solid rgba(255,255,255,0.1);'>", unsafe_allow_html=True)

    # Initialize state for section expansion
    if "sidebar_sections" not in st.session_state:
        st.session_state.sidebar_sections = {
            "core": True,
            "care": True,
            "manage": True,
            "support": True
        }

    # CORE SECTION
    col1, col2 = st.columns([4, 0.8])
    with col1:
        st.markdown('<div style="font-size:0.7rem; text-transform:uppercase; letter-spacing:0.12em; font-weight:800; color:#cbd5e1; margin-bottom:0.6rem;">Core</div>', unsafe_allow_html=True)
    
    core_items = [
        ("🏠 Home", "home"),
        ("💬 AI Assistant", "ai"),
        ("🚨 Emergency Mode", "emergency"),
    ]
    
    for label, key in core_items:
        is_active = st.session_state.page == key
        if st.button(label, key=f"nav_{key}", use_container_width=True,
                     type="primary" if is_active else "secondary"):
            nav(key)

    st.markdown("<hr style='margin: 0.8rem 0; border: none; border-top: 1px solid rgba(255,255,255,0.08);'>", unsafe_allow_html=True)

    # DIAGNOSIS & CARE SECTION
    st.markdown('<div style="font-size:0.7rem; text-transform:uppercase; letter-spacing:0.12em; font-weight:800; color:#cbd5e1; margin-bottom:0.6rem;">Diagnosis & Care</div>', unsafe_allow_html=True)
    
    care_items = [
        ("🩺 Symptom Triage", "triage"),
        ("💊 Medication Checker", "meds"),
        ("📋 Health Reports", "health"),
        ("💉 Vaccination Guide", "vaccines"),
    ]
    
    for label, key in care_items:
        is_active = st.session_state.page == key
        if st.button(label, key=f"nav_{key}", use_container_width=True,
                     type="primary" if is_active else "secondary"):
            nav(key)

    st.markdown("<hr style='margin: 0.8rem 0; border: none; border-top: 1px solid rgba(255,255,255,0.08);'>", unsafe_allow_html=True)

    # PET MANAGEMENT SECTION
    st.markdown('<div style="font-size:0.7rem; text-transform:uppercase; letter-spacing:0.12em; font-weight:800; color:#cbd5e1; margin-bottom:0.6rem;">Your Pets</div>', unsafe_allow_html=True)
    
    manage_items = [
        ("📝 Pet Profiles", "profiles"),
        ("⏰ Reminders", "reminders"),
    ]
    
    for label, key in manage_items:
        is_active = st.session_state.page == key
        if st.button(label, key=f"nav_{key}", use_container_width=True,
                     type="primary" if is_active else "secondary"):
            nav(key)

    # Active Pet Selector
    if st.session_state.pet_profiles:
        names = ["🔓 No active pet"] + [f"🐾 {p['name']}" for p in st.session_state.pet_profiles]
        selected_name = st.session_state.active_pet
        display_names = ["🔓 No active pet"] + [f"🐾 {n}" for n in [p['name'] for p in st.session_state.pet_profiles]]
        idx = 0
        if selected_name:
            for i, p in enumerate(st.session_state.pet_profiles):
                if p['name'] == selected_name:
                    idx = i + 1
                    break
        
        selected = st.selectbox("Active Pet", display_names, index=idx, 
                               key="active_pet_select",
                               label_visibility="collapsed")
        st.session_state.active_pet = None if selected == "🔓 No active pet" else selected.replace("🐾 ", "")

    st.markdown("<hr style='margin: 0.8rem 0; border: none; border-top: 1px solid rgba(255,255,255,0.08);'>", unsafe_allow_html=True)

    # SUPPORT & RESOURCES SECTION
    st.markdown('<div style="font-size:0.7rem; text-transform:uppercase; letter-spacing:0.12em; font-weight:800; color:#cbd5e1; margin-bottom:0.6rem;">Support & Resources</div>', unsafe_allow_html=True)
    
    support_items = [
        ("🗺️ Find Clinics", "vets"),
        ("🐕 Street Animals", "stray"),
        ("🚫 Report Abuse", "abuse"),
        ("⚠️ Report Breeder", "breeder"),
    ]
    
    for label, key in support_items:
        is_active = st.session_state.page == key
        if st.button(label, key=f"nav_{key}", use_container_width=True,
                     type="primary" if is_active else "secondary"):
            nav(key)

    st.markdown("<hr style='margin: 0.8rem 0; border: none; border-top: 2px solid rgba(255,255,255,0.1);'>", unsafe_allow_html=True)

    # Footer Info
    st.markdown("""
    <div style='margin-top:1rem; font-size:0.65rem; color:#5a8a70; text-align:center; line-height:1.6; opacity:0.85;'>
      <div style='margin-bottom: 0.5rem;'>🔒 <strong>Privacy First</strong></div>
      <div>All AI runs locally.<br>Your data never leaves this device.</div>
    </div>
    """, unsafe_allow_html=True)


# HOME

if st.session_state.page == "home":

    # Hero section
    col_hero, col_img = st.columns([3, 2])
    with col_hero:
        st.markdown('<div class="hero-badge">Local AI - No internet required</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-title">Veterinary care,<br>wherever you are.</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-sub">VetConnect AI brings expert animal health guidance to your fingertips — powered by a local AI model that runs entirely on your device. No subscription. No internet. No data sharing.</div>', unsafe_allow_html=True)

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if st.button("Emergency", use_container_width=True, type="primary"):
                nav("emergency")
        with col_b:
            if st.button("Ask the AI", use_container_width=True, type="primary"):
                nav("ai")
        with col_c:
            if st.button("Triage", use_container_width=True, type="primary"):
                nav("triage")

    st.markdown("---")

    # Stats
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Animals Helped", f"{st.session_state.animals_assisted:,}", "+12 today")
    c2.metric("Emergency Cases", f"{st.session_state.emergency_cases:,}", "+3 today")
    c3.metric("Clinics Mapped", len(get_vets()), "Bangalore")
    c4.metric("Rescue NGOs", len(get_ngos()), "Active")

    st.markdown("---")
    st.markdown("### Everything your pet needs")

    features = [
        ("AI Vet Assistant",     "Describe any symptom and get structured, expert-level advice — all offline.",        "ai"),
        ("Symptom Triage",       "Green / Amber / Red triage in seconds. Know how urgent it really is.",               "triage"),
        ("Emergency Mode",       "11 fully guided emergencies: step-by-step DOs, DON'Ts, and timing windows.",         "emergency"),
        ("Clinic Finder",        "Distance-sorted clinics with 24h emergency status, ratings, and specialties.",       "vets"),
        ("Medication Checker",   "Is it safe? AI-powered safety check for any medication, by species.",                "meds"),
        ("Pet Profiles",         "One profile per pet. Every AI query is automatically personalised.",                 "profiles"),
        ("Health Reports",       "AI-generated personalised health report cards for each of your animals.",            "health"),
        ("Vaccination Guides",   "Species-specific vaccination schedules and what's due next.",                       "vaccines"),
        ("Street Animal Support",        "Rescue NGOs, step-by-step street animal handling, and safety guidance.",                    "stray"),
        ("Report Animal Abuse",  "Submit confidential abuse reports to the right authorities. Every report matters.", "abuse"),
        ("Report Illegal Breeder","Flag illegal or unethical breeders. Help shut down puppy mills and exploitation.", "breeder"),
    ]

    for row_start in range(0, len(features), 3):
        cols = st.columns(3)
        for col, feat in zip(cols, features[row_start:row_start+3]):
            title, desc, page = feat
            with col:
                st.markdown(f"""
                <div class="feature-card">
                  <h4>{title}</h4>
                  <p>{desc}</p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown('<div class="quick-action-btn">', unsafe_allow_html=True)
                if st.button(f"Open ->", key=f"feat_{page}", use_container_width=True):
                    nav(page)
                st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div style="text-align:center; font-size:0.78rem; color:#a0aec0; padding:0.5rem 0;">VetConnect AI is informational only and does not replace professional veterinary diagnosis or treatment.</div>', unsafe_allow_html=True)


# AI VET ASSISTANT

elif st.session_state.page == "ai":

    st.markdown('<div class="page-header"><div class="section-header">AI Vet Assistant</div><div class="section-sub">Ask anything about your pet\'s health. The AI maintains context across your conversation.</div></div>', unsafe_allow_html=True)

    pet = get_active_pet()
    if pet:
        st.markdown(f"""
        <div style="background:var(--blue-pale); border:1px solid #bfdbfe; border-radius:10px; padding:10px 16px; margin-bottom:1rem; font-size:0.88rem; color:var(--blue-dark);">
          Active pet: {pet['name']} - {pet['species']}, {pet['age']}, {pet['weight']} kg
          {f"<br>Medical notes: <em>{pet['notes']}</em>" if pet.get('notes') else ""}
        </div>
        """, unsafe_allow_html=True)

    # Chat history
    chat_container = st.container()
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown("""
            <div style="text-align:center; padding:2rem; color:#a0aec0;">
              <div style="font-size:1.3rem; margin-bottom:0.5rem;">Animal Care Assistant</div>
              <div style="font-size:0.95rem;">Hello! I'm VetConnect AI. Describe your pet's symptoms or ask me anything about their health.</div>
              <div style="font-size:0.82rem; margin-top:0.5rem; color:#cbd5e0;">Try: My dog has been vomiting since this morning or Is my cat's limp serious?</div>
            </div>
            """, unsafe_allow_html=True)
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"<div class='chat-user'>You: {msg['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='chat-ai'><div class='chat-ai-label'>VetConnect AI</div>{msg['content']}</div>", unsafe_allow_html=True)

    st.markdown("---")

    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_area(
            "Your question or symptom description",
            height=100,
            placeholder="e.g. My Labrador hasn't eaten today and keeps licking his paws. He's 4 years old and 32kg. No vomiting...",
            label_visibility="collapsed",
        )
        col_send, col_clear = st.columns([4, 1])
        with col_send:
            send = st.form_submit_button("Send", use_container_width=True, type="primary")
        with col_clear:
            clear = st.form_submit_button("Clear chat", use_container_width=True)

    if clear:
        st.session_state.chat_history = []
        st.rerun()

    if send and user_input.strip():
        full_input = user_input
        if pet:
            full_input = (
                f"Patient: {pet['name']}, {pet['species']}, age {pet['age']}, weight {pet['weight']} kg.\n"
                f"Medical notes: {pet.get('notes', 'None')}\n"
                f"Breed: {pet.get('breed', 'Unknown')}\n\n"
                f"Owner: {user_input}"
            )
        with st.spinner("VetConnect AI is thinking..."):
            response = ask_vet_ai(full_input, st.session_state.chat_history[-10:])
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.session_state.animals_assisted += 1
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# SYMPTOM TRIAGE
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.page == "triage":

    st.markdown('<div class="page-header"><div class="section-header">🔍 Symptom Triage</div><div class="section-sub">Get a structured Green / Amber / Red assessment in seconds. Know how serious it really is.</div></div>', unsafe_allow_html=True)

    pet = get_active_pet()
    SPECIES = ["Dog", "Cat", "Rabbit", "Bird", "Guinea Pig", "Other"]

    with st.form("triage_form"):
        c1, c2 = st.columns(2)
        with c1:
            def_species = SPECIES.index(pet["species"]) if pet and pet.get("species") in SPECIES else 0
            species  = st.selectbox("Species *", SPECIES, index=def_species)
            age      = st.text_input("Age *", value=pet["age"] if pet else "", placeholder="e.g. 3 years 2 months")
        with c2:
            weight   = st.text_input("Weight (kg)", value=str(pet["weight"]) if pet else "", placeholder="e.g. 12")
            duration = st.text_input("How long have symptoms been present? *", placeholder="e.g. 6 hours, since yesterday morning")

        symptoms = st.text_area(
            "Describe all symptoms in as much detail as possible *",
            height=110,
            placeholder="e.g. Vomiting twice in the last 3 hours, not eating, hunched posture, gums look slightly pale. Ate something from the garden earlier.",
        )
        submitted = st.form_submit_button("Run Triage →", use_container_width=True, type="primary")

    if submitted and symptoms.strip():
        with st.spinner("Analysing symptoms..."):
            result = triage_symptoms(species, age, weight, symptoms, duration)
        st.session_state.triage_result = result
        st.session_state.emergency_cases += 1

    if st.session_state.triage_result:
        result_text = st.session_state.triage_result
        st.markdown("---")

        level_class = "triage-red" if "Red" in result_text else "triage-amber" if "Amber" in result_text else "triage-green"
        level_icon  = "🔴 Emergency — Act Now" if "Red" in result_text else "🟡 Vet Visit Recommended" if "Amber" in result_text else "🟢 Monitor at Home"

        st.markdown(f"<div class='{level_class}'><strong>{level_icon}</strong></div>", unsafe_allow_html=True)
        st.markdown(result_text)

        if "Red" in result_text:
            st.markdown("---")
            col_e, col_v = st.columns(2)
            with col_e:
                if st.button("🚨 Open Emergency Mode →", type="primary", use_container_width=True):
                    nav("emergency")
            with col_v:
                if st.button("🗺️ Find Nearest Emergency Clinic →", use_container_width=True):
                    nav("vets")


# ══════════════════════════════════════════════════════════════════════════════
# EMERGENCY MODE
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.page == "emergency":

    # Top alert banner
    st.markdown("""
    <div style="background:#c0392b; color:white; border-radius:10px; padding:12px 18px; margin-bottom:1.2rem; font-weight:700; font-size:1rem; display:flex; align-items:center; gap:10px;">
      🚨 EMERGENCY MODE — Read the steps carefully. Contact a vet as soon as you can.
    </div>
    """, unsafe_allow_html=True)

    emergency_names = get_all_emergencies()
    critical_names  = get_critical_emergencies()

    # Pre-select if navigated here with a suggestion
    default_idx = 0
    if st.session_state.get("selected_emergency") in emergency_names:
        default_idx = emergency_names.index(st.session_state.selected_emergency)

    selected = st.selectbox(
        "Select the emergency situation",
        emergency_names,
        index=default_idx,
        format_func=lambda x: f"{'🔴 ' if x in critical_names else '🟡 '}{x}",
    )

    guide = get_emergency_guide(selected)
    if not guide:
        st.error("Guide not found.")
        st.stop()

    # Severity + summary
    sev_colour = "#c0392b" if guide.severity == "Critical" else "#d4850a"
    st.markdown(f"""
    <div style="background:white; border:1px solid {sev_colour}; border-left:5px solid {sev_colour}; border-radius:10px; padding:1rem 1.2rem; margin-bottom:0.8rem;">
      <div style="font-size:1.1rem; font-weight:800; color:{sev_colour};">{guide.icon} {guide.severity} Emergency</div>
      <div style="font-size:0.9rem; color:#4a5568; margin-top:4px; line-height:1.55;">{guide.summary}</div>
    </div>
    """, unsafe_allow_html=True)

    # Time window
    st.markdown(f'<div class="window-banner">⏱️ {guide.estimated_window}</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["✅  DOs & DON'Ts", "📋  Step-by-Step", "🏥  Call Vet If...", "📞  Contacts & Info"])

    # ── Tab 1: DOs & DON'Ts
    with tab1:
        col_do, col_dont = st.columns(2)
        with col_do:
            st.markdown("#### ✅ DO these things")
            for item in guide.immediate_dos:
                st.markdown(f"<div class='do-item'>✔ {item}</div>", unsafe_allow_html=True)
        with col_dont:
            st.markdown("#### ❌ DON'T do these")
            for item in guide.immediate_donts:
                st.markdown(f"<div class='dont-item'>✖ {item}</div>", unsafe_allow_html=True)
        st.markdown(f'<div class="reassurance-box">💬 {guide.reassurance}</div>', unsafe_allow_html=True)

    # ── Tab 2: Step-by-step
    with tab2:
        st.markdown("#### Follow these steps in order — do not skip")
        for i, step in enumerate(guide.step_by_step, 1):
            st.markdown(f"<div class='step-item'><strong style='color:var(--blue-mid)'>Step {i}</strong> — {step}</div>", unsafe_allow_html=True)

    # ── Tab 3: Call vet triggers
    with tab3:
        st.markdown("#### Go to a vet immediately if you see ANY of the following")
        for trigger in guide.call_vet_if:
            st.markdown(f"<div class='call-vet-item'>🔴 {trigger}</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### 🏥 Nearest Emergency Clinics")
        vets = get_vets()
        emerg_vets = vets[vets["Emergency"] == True][["Name", "Area", "Phone", "Hours", "Rating"]]
        st.dataframe(emerg_vets, hide_index=True, use_container_width=True)

    # ── Tab 4: Contacts & Info
    with tab4:
        st.markdown("#### 📞 Animal Rescue Organisations")
        ngos = get_ngos()
        for _, ngo in ngos.iterrows():
            wa = "✅ WhatsApp" if ngo.get("WhatsApp") else ""
            st.markdown(f"""
            <div class="ngo-card">
              <strong>{ngo['Name']}</strong> — {ngo['Area']}<br>
              <span style="font-size:0.85rem; color:#4a5568;">📞 {ngo['Phone']} · {ngo['Service']} {wa}</span>
            </div>
            """, unsafe_allow_html=True)

        if guide.related:
            st.markdown("---")
            st.markdown("#### Related emergencies")
            for rel in guide.related:
                st.markdown(f"<span class='related-chip'>{rel}</span>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# FIND CLINICS
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.page == "vets":

    st.markdown('<div class="page-header"><div class="section-header">🗺️ Find Veterinary Clinics</div><div class="section-sub">Enter your location to find the nearest clinics ranked by distance.</div></div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: user_lat = st.number_input("Latitude",  value=12.9716, format="%.4f")
    with c2: user_lon = st.number_input("Longitude", value=77.5946, format="%.4f")
    with c3: emerg_only = st.checkbox("24h Emergency only")
    with c4: type_filter = st.selectbox("Type", ["All", "Government", "Private", "NGO"])

    vets = get_vets()
    if emerg_only:   vets = vets[vets["Emergency"] == True]
    if type_filter != "All": vets = vets[vets["Type"] == type_filter]
    vets = vets.copy()
    vets["Distance"] = vets.apply(lambda r: haversine(user_lat, user_lon, r["Latitude"], r["Longitude"]), axis=1)
    nearest = vets.sort_values("Distance").head(8)

    st.markdown("---")

    col_list, col_map = st.columns([2, 3])

    with col_list:
        st.markdown("#### Nearest Clinics")
        for _, row in nearest.iterrows():
            type_class = TYPE_BADGE.get(row["Type"], "badge-private")
            emerg_badge = "<span class='badge badge-emergency'>24h Emergency</span>" if row["Emergency"] else ""
            rating_str  = f"<span class='badge badge-rating'>★ {row['Rating']}</span>"
            st.markdown(f"""
            <div class="vet-card">
              <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                <div>
                  <div class="vet-name">{row['Name']}</div>
                  <div class="vet-meta">📍 {row['Area']} · 📞 {row['Phone']}</div>
                  <div class="vet-meta" style="margin-top:3px;">🕐 {row['Hours']} · 🔬 {row['Specialties']}</div>
                  <div style="margin-top:6px;">{emerg_badge} <span class="badge {type_class}">{row['Type']}</span> {rating_str}</div>
                </div>
                <div class="vet-dist">{row['Distance']} km</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    with col_map:
        st.markdown("#### Map")
        # Create map with better styling
        m = folium.Map(
            location=[user_lat, user_lon], 
            zoom_start=13, 
            tiles="OpenStreetMap",
            prefer_canvas=True,
            max_zoom=19,
            min_zoom=8
        )
        
        # Add custom user location marker
        folium.Marker(
            [user_lat, user_lon],
            popup=folium.Popup(
                '<div style="font-family:Arial; font-weight:bold; color:#1e40af;">📍 Your Location</div>',
                max_width=200
            ),
            tooltip="You are here",
            icon=folium.Icon(color="blue", icon="home", prefix="fa", icon_color="white"),
            z_index=1000
        ).add_to(m)
        
        # Add circle around user location for reference
        folium.Circle(
            [user_lat, user_lon],
            radius=5000,
            color="#2563eb",
            fill=True,
            fillColor="#93c5fd",
            fillOpacity=0.1,
            weight=2,
            opacity=0.6,
            popup="5 km radius"
        ).add_to(m)
        
        # Color map with better blue theme
        color_map = {"Government": "#2563eb", "NGO": "#f59e0b", "Private": "#dc2626"}
        icon_map = {"Government": "hospital", "NGO": "heart", "Private": "plus"}
        
        for idx, (_, row) in enumerate(nearest.iterrows()):
            clinictype = row["Type"]
            distance_km = row['Distance']
            
            # Marker size based on rating
            marker_size = int(30 + (row['Rating'] * 5))
            
            # Custom styled popup with professional layout
            popup_html = f"""
            <div style="font-family: Arial, sans-serif; width: 250px; padding: 12px;">
                <div style="font-size: 16px; font-weight: bold; color: #1e40af; margin-bottom: 8px;">
                    {row['Name']}
                </div>
                <div style="border-bottom: 2px solid #93c5fd; margin-bottom: 8px;"></div>
                <div style="font-size: 13px; color: #334155; line-height: 1.8;">
                    <div>📞 <strong>{row['Phone']}</strong></div>
                    <div>📍 {row['Area']}</div>
                    <div>🕐 {row['Hours']}</div>
                    <div>🔬 {row['Specialties']}</div>
                    <div style="margin-top: 6px; padding-top: 6px; border-top: 1px solid #e2e8f0;">
                        <span style="background: #e0f2fe; color: #1e40af; padding: 2px 8px; border-radius: 4px; font-weight: bold;">
                            ★ {row['Rating']}
                        </span>
                        <span style="background: #dbeafe; color: #1e40af; padding: 2px 8px; border-radius: 4px; margin-left: 4px;">
                            {distance_km:.1f} km
                        </span>
                        {'<span style="background: #fecaca; color: #991b1b; padding: 2px 8px; border-radius: 4px; margin-left: 4px; font-weight: bold;">Emergency 24h</span>' if row['Emergency'] else ''}
                    </div>
                </div>
            </div>
            """
            
            folium.Marker(
                [row["Latitude"], row["Longitude"]],
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"<b>{row['Name']}</b><br>{distance_km:.1f} km away",
                icon=folium.Icon(
                    color=color_map.get(clinictype, "#6b7280"), 
                    icon=icon_map.get(clinictype, "plus-sign"),
                    prefix="fa",
                    icon_color="white"
                ),
                z_index=500 - idx  # Closer ones on top
            ).add_to(m)
        
        # Add layer control
        folium.LayerControl(position='topright', collapsed=False).add_to(m)
        
        st_folium(m, width=None, height=520, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# STREET ANIMAL SUPPORT
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.page == "stray":

    st.markdown('<div class="page-header"><div class="section-header">🐕 Street Animal Support</div><div class="section-sub">How to help a street or injured animal safely — and who to call.</div></div>', unsafe_allow_html=True)

    col_ngo, col_steps = st.columns([3, 2])

    with col_ngo:
        st.markdown("#### Rescue Organisations")
        ngos = get_ngos()
        for _, row in ngos.iterrows():
            wa_tag = "<span class='badge' style='background:#e0f2fe; color:#1e40af; border:1px solid #bfdbfe;'>WhatsApp</span>" if row.get("WhatsApp") else ""
            street_animal_tag = "<span class='badge' style='background:#e8f4fd; color:#2980b9; border:1px solid #bee3f8;'>Accepts Street Animals</span>" if row.get("Accepts_Street_Animals") else ""
            st.markdown(f"""
            <div class="ngo-card">
              <div style="font-weight:700; color:var(--blue-dark); font-size:0.95rem;">{row['Name']}</div>
              <div style="font-size:0.83rem; color:#4a5568; margin:3px 0;">📍 {row['Area']} · 📞 {row['Phone']}</div>
              <div style="font-size:0.83rem; color:#4a5568;">{row['Service']}</div>
              <div style="margin-top:6px;">{wa_tag} {street_animal_tag}</div>
            </div>
            """, unsafe_allow_html=True)

    with col_steps:
        st.markdown("#### What to do right now")
        steps = [
            ("🛡️", "Ensure your own safety first — injured animals may bite"),
            ("📍", "Note exact location to share with rescuers"),
            ("🚧", "Create a safe barrier around the animal using bags or a jacket"),
            ("💧", "Place water nearby if the animal is conscious"),
            ("📷", "Take a photo to share with the rescue team"),
            ("📞", "Call the nearest organisation (left column)"),
            ("🕐", "Stay nearby if safe — animals are calmer with a human presence"),
        ]
        for icon, text in steps:
            st.markdown(f"""
            <div style="display:flex; gap:10px; align-items:flex-start; margin:8px 0; background:white; border-radius:8px; padding:10px; border:1px solid var(--border);">
              <div style="font-size:1.2rem;">{icon}</div>
              <div style="font-size:0.88rem; color:#2d3748; line-height:1.5;">{text}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div style="background:#fff5f5; border:1px solid #fed7d7; border-radius:8px; padding:12px 14px; font-size:0.85rem; color:#742a2a; line-height:1.55;">
          ⚠️ <strong>Do NOT</strong> attempt to pick up injured street animals without guidance. Bites and scratches from stressed animals can transmit rabies and other infections.
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MEDICATION CHECKER
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.page == "meds":

    st.markdown('<div class="page-header"><div class="section-header">💊 Medication Safety Checker</div><div class="section-sub">Check if any medication is safe for your pet\'s species. Powered by local AI.</div></div>', unsafe_allow_html=True)

    pet = get_active_pet()
    SPECIES = ["Dog", "Cat", "Rabbit", "Bird", "Guinea Pig", "Other"]
    def_species = SPECIES.index(pet["species"]) if pet and pet.get("species") in SPECIES else 0

    col1, col2 = st.columns(2)
    with col1:
        medication = st.text_input("Medication or substance name", placeholder="e.g. Metronidazole, Benadryl, Ibuprofen, Xylitol")
    with col2:
        species = st.selectbox("Pet species", SPECIES, index=def_species)

    if st.button("Check Safety →", type="primary") and medication.strip():
        with st.spinner("Checking medication safety..."):
            result = ask_medication_info(medication, species)
        st.markdown("---")
        is_danger = any(w in result.upper() for w in ["DANGEROUS", "TOXIC", "FATAL", "DO NOT", "NEVER", "☠️", "🚫"])
        if is_danger:
            st.error(f"⚠️ Safety concern detected for **{medication}** in **{species}**")
        else:
            st.success(f"Medication information for **{medication}** in **{species}**")
        st.markdown(result)
        st.caption("⚠️ This is informational guidance from a local AI model — not a prescription or veterinary advice.")

    st.markdown("---")
    st.markdown("#### ⚠️ Known toxic substances — always emergency vet")
    danger_data = pd.DataFrame({
        "Substance":      ["Ibuprofen / Advil", "Paracetamol / Tylenol", "Xylitol (sweetener)", "Grapes & Raisins", "Aspirin", "Dark Chocolate", "Lilies", "Rat Poison (anticoagulant)"],
        "Toxic to":       ["Dogs & Cats", "Cats (severe)", "Dogs (severe)", "Dogs", "Cats especially", "Dogs & Cats", "Cats (fatal)", "Dogs & Cats"],
        "Primary Risk":   ["Kidney & GI failure", "Liver failure", "Severe hypoglycaemia", "Kidney failure", "GI bleeding", "Heart arrhythmia", "Acute kidney failure", "Internal bleeding"],
    })
    st.dataframe(danger_data, hide_index=True, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PET PROFILES
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.page == "profiles":

    st.markdown('<div class="page-header"><div class="section-header">📋 Pet Profiles</div><div class="section-sub">Save your pets\' details. Every AI query is automatically personalised when a pet is active.</div></div>', unsafe_allow_html=True)

    with st.expander("➕ Add New Pet Profile", expanded=not st.session_state.pet_profiles):
        with st.form("add_pet"):
            c1, c2 = st.columns(2)
            with c1:
                p_name    = st.text_input("Pet Name *")
                p_species = st.selectbox("Species *", ["Dog", "Cat", "Rabbit", "Bird", "Guinea Pig", "Other"])
                p_age     = st.text_input("Age *", placeholder="e.g. 2 years 4 months")
            with c2:
                p_weight  = st.number_input("Weight (kg) *", min_value=0.1, max_value=200.0, value=5.0, step=0.5)
                p_breed   = st.text_input("Breed (optional)", placeholder="e.g. Labrador, Persian, Holland Lop")
                p_sex     = st.selectbox("Sex", ["Unknown", "Male (neutered)", "Male (intact)", "Female (spayed)", "Female (intact)"])
            p_notes = st.text_area("Medical conditions, allergies, or special notes", height=80,
                                   placeholder="e.g. Has epilepsy, on phenobarbitone 30mg twice daily. Allergic to amoxicillin.")
            if st.form_submit_button("Save Profile →", type="primary"):
                if p_name.strip():
                    st.session_state.pet_profiles.append({
                        "name": p_name, "species": p_species, "age": p_age,
                        "weight": p_weight, "breed": p_breed, "sex": p_sex, "notes": p_notes,
                    })
                    st.success(f"✅ Profile saved for **{p_name}**!")
                    st.rerun()
                else:
                    st.error("Pet name is required.")

    if st.session_state.pet_profiles:
        st.markdown("#### Your Pets")
        for i, pet in enumerate(st.session_state.pet_profiles):
            emoji = SPECIES_EMOJI.get(pet["species"], "🐾")
            breed_risks = get_breed_risks(pet.get("breed", ""))
            col_card, col_del = st.columns([5, 1])
            with col_card:
                risk_html = ""
                if breed_risks:
                    risk_tags = " ".join([f"<span class='badge' style='background:#fff8e1; color:#7b5300; border:1px solid #ffe082;'>{r}</span>" for r in breed_risks])
                    risk_html = f"<div style='margin-top:8px; font-size:0.78rem; color:#7b5300;'><strong>Breed watch:</strong> {risk_tags}</div>"
                st.markdown(f"""
                <div class="pet-profile-card">
                  <div style="display:flex; gap:12px; align-items:flex-start;">
                    <div class="pet-avatar">{emoji}</div>
                    <div>
                      <div style="font-weight:800; font-size:1rem; color:var(--blue-dark);">{pet['name']}</div>
                      <div style="font-size:0.85rem; color:#4a5568;">{pet['species']} · {pet.get('breed','') or 'Breed not specified'} · {pet['age']} · {pet['weight']} kg · {pet.get('sex','')}</div>
                      {f"<div style='font-size:0.83rem; color:#718096; margin-top:3px; font-style:italic;'>📝 {pet['notes']}</div>" if pet.get('notes') else ""}
                      {risk_html}
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
            with col_del:
                st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
                if st.button("Delete", key=f"del_{i}", type="secondary"):
                    st.session_state.pet_profiles.pop(i)
                    if st.session_state.active_pet == pet["name"]:
                        st.session_state.active_pet = None
                    st.rerun()
    else:
        st.info("No pets added yet. Use the form above to add your first pet.")


# ══════════════════════════════════════════════════════════════════════════════
# HEALTH REPORTS
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.page == "health":

    st.markdown('<div class="page-header"><div class="section-header">🏥 Health Report Card</div><div class="section-sub">AI-generated personalised health summary for your pet based on their profile.</div></div>', unsafe_allow_html=True)

    if not st.session_state.pet_profiles:
        st.info("Add a pet profile first to generate a health report.")
        if st.button("Go to Pet Profiles →", type="primary"):
            nav("profiles")
        st.stop()

    pet_names = [p["name"] for p in st.session_state.pet_profiles]
    selected_name = st.selectbox("Select pet", pet_names)
    pet = next(p for p in st.session_state.pet_profiles if p["name"] == selected_name)

    col_pet, col_btn = st.columns([3, 1])
    with col_pet:
        emoji = SPECIES_EMOJI.get(pet["species"], "🐾")
        st.markdown(f"""
        <div style="background:var(--blue-pale); border-radius:12px; padding:1rem 1.4rem; border:1px solid #bfdbfe;">
          <span style="font-size:1.8rem;">{emoji}</span>
          <strong style="font-size:1.05rem; color:var(--blue-dark);">{pet['name']}</strong>
          <span style="color:#4a5568; font-size:0.88rem;"> · {pet['species']} · {pet.get('breed','?')} · {pet['age']} · {pet['weight']} kg</span>
        </div>
        """, unsafe_allow_html=True)
    with col_btn:
        generate = st.button("Generate Report →", type="primary", use_container_width=True)

    if generate:
        with st.spinner(f"Generating personalised health report for {pet['name']}..."):
            report = generate_health_summary(pet)
        st.session_state.health_report = (selected_name, report)

    if st.session_state.health_report and st.session_state.health_report[0] == selected_name:
        _, report_text = st.session_state.health_report
        st.markdown("---")
        st.markdown(f"""
        <div style="background:white; border:1px solid var(--border); border-radius:14px; padding:1.5rem; box-shadow:var(--card-shadow);">
          <div style="font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; color:var(--blue-mid); margin-bottom:0.8rem;">🏥 Health Report — {pet['name']} — {datetime.now().strftime('%d %b %Y')}</div>
          {report_text.replace(chr(10), '<br>')}
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# VACCINATION GUIDE
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.page == "vaccines":

    st.markdown('<div class="page-header"><div class="section-header">📅 Vaccination Guide</div><div class="section-sub">Species-specific vaccination schedules and recommended timings.</div></div>', unsafe_allow_html=True)

    species = st.radio("Species", ["Dog", "Cat"], horizontal=True)
    schedule_df = get_vaccination_schedule(species)

    if schedule_df.empty:
        st.info(f"No schedule available for {species} yet.")
    else:
        st.dataframe(
            schedule_df.style.map(
                lambda v: "background-color:#fff3cd; color:#7d5a00;" if v == "Booster"
                else "background-color:#fde8e8; color:#7d1515;" if v == "Core"
                else "",
                subset=["Type"]
            ),
            hide_index=True,
            use_container_width=True,
        )

    st.markdown("---")
    st.markdown("""
    <div style="background:#e8f4fd; border:1px solid #bee3f8; border-radius:10px; padding:12px 16px; font-size:0.87rem; color:#1a5276; line-height:1.6;">
      <strong>📌 Important notes:</strong><br>
      • These are general guidelines — your vet may recommend different timing based on local disease prevalence.<br>
      • Puppies and kittens should not go outdoors until 1–2 weeks after their final puppy/kitten vaccinations.<br>
      • Keep vaccination records — many boarding facilities, groomers, and airlines require proof of vaccination.<br>
      • Rabies vaccination is required by law in many states in India.
    </div>
    """, unsafe_allow_html=True)

    # AI first aid quick-ref section
    st.markdown("---")
    st.markdown('<div class="section-header" style="font-size:1.2rem;">🩺 First Aid Quick Reference</div>', unsafe_allow_html=True)
    st.caption("Common situations and what to do at home vs when to call a vet.")
    first_aid_df = get_first_aid_quickref()
    for _, row in first_aid_df.iterrows():
        with st.expander(f"🩹 {row['Situation']}"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**What to do:** {row['Steps']}")
            with col_b:
                st.markdown(f"**Go to vet if:** {row['Go to Vet']}")


# ══════════════════════════════════════════════════════════════════════════════
# REMINDERS
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.page == "reminders":

    st.markdown('<div class="page-header"><div class="section-header">⏰ Reminders</div><div class="section-sub">Track medications, treatments, and vet appointments for all your pets.</div></div>', unsafe_allow_html=True)

    with st.expander("➕ Add Reminder", expanded=not st.session_state.reminders):
        with st.form("add_reminder"):
            c1, c2 = st.columns(2)
            with c1:
                r_pet  = st.text_input("Pet name", value=st.session_state.active_pet or "")
                r_type = st.selectbox("Type", ["Medication", "Vet Appointment", "Vaccination", "Treatment", "Grooming", "Other"])
                r_med  = st.text_input("Medication / Treatment / Note", placeholder="e.g. Metronidazole, Annual checkup")
            with c2:
                r_dose = st.text_input("Dose (if medication)", placeholder="e.g. 1 tablet, 5ml")
                r_time = st.time_input("Time")
                r_freq = st.selectbox("Frequency", ["Once", "Daily", "Twice Daily", "Every 8 Hours", "Weekly", "Monthly", "As needed"])
            r_notes = st.text_input("Additional notes", placeholder="e.g. Give with food")
            if st.form_submit_button("Add Reminder →", type="primary"):
                st.session_state.reminders.append({
                    "pet": r_pet, "type": r_type, "medication": r_med,
                    "dose": r_dose, "time": str(r_time), "frequency": r_freq,
                    "notes": r_notes, "added": datetime.now().strftime("%d %b %Y"),
                })
                st.success("✅ Reminder added!")
                st.rerun()

    if st.session_state.reminders:
        st.markdown("#### Active Reminders")
        for i, rem in enumerate(st.session_state.reminders):
            type_emoji = {"Medication": "💊", "Vet Appointment": "🏥", "Vaccination": "💉", "Treatment": "🩺", "Grooming": "✂️", "Other": "📌"}.get(rem.get("type",""), "📌")
            c_rem, c_del = st.columns([5, 1])
            with c_rem:
                st.markdown(f"""
                <div class="reminder-item">
                  <div style="font-weight:700; color:var(--blue-dark);">{type_emoji} {rem['pet']} — {rem['medication']} {f"({rem['dose']})" if rem.get('dose') else ""}</div>
                  <div style="font-size:0.83rem; color:#4a5568; margin-top:3px;">🕐 {rem['time']} · {rem['frequency']} · Added {rem['added']}</div>
                  {f"<div style='font-size:0.82rem; color:#718096; margin-top:2px;'>📝 {rem['notes']}</div>" if rem.get('notes') else ""}
                </div>
                """, unsafe_allow_html=True)
            with c_del:
                st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)
                if st.button("Delete", key=f"rem_{i}"):
                    st.session_state.reminders.pop(i)
                    st.rerun()
    else:
        st.info("No reminders yet. Add one above to track medications and appointments.")


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.page == "dashboard":

    st.markdown('<div class="page-header"><div class="section-header">📊 Impact Dashboard</div><div class="section-sub">Platform stats and clinic network overview.</div></div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Animals Helped",     f"{st.session_state.animals_assisted:,}", "+12 today")
    c2.metric("Emergency Cases",    f"{st.session_state.emergency_cases:,}", "+3 today")
    c3.metric("Clinics in Network", len(get_vets()),  "Bangalore")
    c4.metric("Active NGOs",        len(get_ngos()),  "6 on WhatsApp")

    st.markdown("---")

    col_a, col_b = st.columns(2)
    vets = get_vets()

    with col_a:
        st.markdown("#### Clinics by Type")
        st.bar_chart(vets["Type"].value_counts(), color="#2d6a4f")

    with col_b:
        st.markdown("#### Emergency vs Standard Clinics")
        ec = vets["Emergency"].value_counts().rename({True: "24h Emergency", False: "Standard Hours"})
        st.bar_chart(ec, color="#52b788")

    st.markdown("---")
    st.markdown("#### Full Clinic Directory")
    display_vets = vets[["Name", "Area", "Type", "Emergency", "Phone", "Hours", "Rating", "Specialties"]]
    st.dataframe(display_vets, hide_index=True, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# REPORT ANIMAL ABUSE
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.page == "abuse":

    st.markdown("""
    <div style="background:#7b1010; color:white; border-radius:10px; padding:12px 18px; margin-bottom:1.2rem; font-weight:700; font-size:1rem;">
      🆘 ANIMAL ABUSE REPORTING — Your report is confidential. You could save a life.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="page-header"><div class="section-header">🆘 Report Animal Abuse</div><div class="section-sub">Use this form to report suspected animal cruelty, neglect, or abuse. All reports are kept confidential. Authorities and NGOs will be listed below.</div></div>', unsafe_allow_html=True)

    # ── National / State contacts
    st.markdown("### 📞 Who to call right now")
    contacts_col1, contacts_col2 = st.columns(2)
    with contacts_col1:
        st.markdown("""
        <div style="background:white; border:1px solid var(--border); border-left:4px solid #c0392b; border-radius:10px; padding:1rem 1.2rem; margin-bottom:10px; box-shadow:var(--card-shadow);">
          <div style="font-weight:700; color:#c0392b; font-size:0.95rem;">🏛️ Animal Welfare Board of India (AWBI)</div>
          <div style="font-size:0.85rem; color:#4a5568; margin-top:4px;">📞 1800-11-4444 (Toll-Free)</div>
          <div style="font-size:0.82rem; color:#718096; margin-top:2px;">awbi@nic.in · National authority for animal welfare</div>
        </div>
        <div style="background:white; border:1px solid var(--border); border-left:4px solid #e9c46a; border-radius:10px; padding:1rem 1.2rem; margin-bottom:10px; box-shadow:var(--card-shadow);">
          <div style="font-weight:700; color:#7d5a00; font-size:0.95rem;">🚔 Karnataka Police — Animal Cruelty</div>
          <div style="font-size:0.85rem; color:#4a5568; margin-top:4px;">📞 112 (Emergency) · 080-22942222 (Non-emergency)</div>
          <div style="font-size:0.82rem; color:#718096; margin-top:2px;">PCA Act 1960 offences — file an FIR</div>
        </div>
        <div style="background:white; border:1px solid var(--border); border-left:4px solid #2980b9; border-radius:10px; padding:1rem 1.2rem; margin-bottom:10px; box-shadow:var(--card-shadow);">
          <div style="font-weight:700; color:#1a5276; font-size:0.95rem;">🐾 CUPA — Cruelty Investigations</div>
          <div style="font-size:0.85rem; color:#4a5568; margin-top:4px;">📞 080-22947300 · 9483085XXX (WhatsApp)</div>
          <div style="font-size:0.82rem; color:#718096; margin-top:2px;">Bangalore's primary cruelty investigation NGO</div>
        </div>
        """, unsafe_allow_html=True)
    with contacts_col2:
        st.markdown("""
        <div style="background:white; border:1px solid var(--border); border-left:4px solid #52b788; border-radius:10px; padding:1rem 1.2rem; margin-bottom:10px; box-shadow:var(--card-shadow);">
          <div style="font-weight:700; color:#1a3d2b; font-size:0.95rem;">🐕 People For Animals (PFA)</div>
          <div style="font-size:0.85rem; color:#4a5568; margin-top:4px;">📞 080-28602682</div>
          <div style="font-size:0.82rem; color:#718096; margin-top:2px;">National network, Bangalore chapter active</div>
        </div>
        <div style="background:white; border:1px solid var(--border); border-left:4px solid #9b59b6; border-radius:10px; padding:1rem 1.2rem; margin-bottom:10px; box-shadow:var(--card-shadow);">
          <div style="font-weight:700; color:#6c3483; font-size:0.95rem;">⚖️ Prevention of Cruelty to Animals Act, 1960</div>
          <div style="font-size:0.85rem; color:#4a5568; margin-top:4px;">Key Sections: 11 (cruelty), 12 (performing animals), 13 (killing)</div>
          <div style="font-size:0.82rem; color:#718096; margin-top:2px;">Max penalty: ₹10,000 fine and/or 5 years imprisonment</div>
        </div>
        <div style="background:#fff5f5; border:1px solid #fed7d7; border-radius:10px; padding:1rem 1.2rem; margin-bottom:10px;">
          <div style="font-weight:700; color:#c0392b; font-size:0.88rem;">⚠️ If the animal is in immediate danger</div>
          <div style="font-size:0.83rem; color:#742a2a; margin-top:4px; line-height:1.5;">Call 112 first. Document with photos/video if safe to do so. Do NOT confront the abuser alone.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Karnataka State Animal Welfare Board
    st.markdown("### 🏛️ Karnataka State Animal Welfare Board (KSAWB)")
    st.markdown("""
    <div style="background:#e3f2fd; border:2px solid #2563eb; border-radius:12px; padding:1.2rem; margin-bottom:1.2rem;">
      <div style="font-weight:700; color:#1e40af; font-size:0.95rem; margin-bottom:0.8rem;">📍 State Nodal Authority for Animal Welfare</div>
      
      <div style="display:grid; grid-template-columns:1fr 1fr; gap:1rem; font-size:0.85rem; color:#1f2937; line-height:1.6;">
        <div>
          <div style="font-weight:600; color:#1e40af; margin-bottom:0.4rem;">📞 Contact Details</div>
          <div>📞 Phone: 080-2258-6595</div>
          <div>📧 Email: ksawb@gmail.com</div>
          <div>🌐 Website: ksawb.karnataka.gov.in</div>
        </div>
        <div>
          <div style="font-weight:600; color:#1e40af; margin-bottom:0.4rem;">📍 Office Location</div>
          <div>Dr. Rajendra Prasad Rd, Ashokanagar,</div>
          <div>Bengaluru, Karnataka 560001</div>
          <div style="margin-top:0.4rem; font-size:0.8rem; color:#666;">Working Hours: Mon-Fri, 10 AM - 5 PM</div>
        </div>
      </div>
      
      <div style="margin-top:1rem; padding-top:1rem; border-top:1px solid #90caf9; font-size:0.83rem; color:#1f2937;">
        <div style="font-weight:600; color:#1e40af; margin-bottom:0.4rem;">✓ KSAWB Services</div>
        <ul style="margin:0.4rem 0 0 1.2rem; padding:0;">
          <li>Registers and oversees animal welfare organizations</li>
          <li>Provides legal support and investigative assistance for animal cruelty cases</li>
          <li>Issues licenses for animal facilities and veterinary practices</li>
          <li>Conducts welfare inspections and audit of sanctuaries</li>
          <li>Can file cases under Prevention of Cruelty to Animals Act</li>
        </ul>
      </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📝 Submit an Abuse Report")
    st.caption("Your identity is not shared with the abuser. Reports are forwarded to relevant NGOs and authorities.")

    with st.form("abuse_form"):
        c1, c2 = st.columns(2)
        with c1:
            a_type     = st.selectbox("Type of abuse *", [
                "Physical cruelty / beating",
                "Neglect (no food/water/shelter)",
                "Abandonment",
                "Dog fighting / animal baiting",
                "Poisoning",
                "Hoarding (too many animals in poor conditions)",
                "Chaining / confinement cruelty",
                "Vehicular cruelty",
                "Other / Unknown",
            ])
            a_species  = st.selectbox("Animal species *", ["Dog", "Cat", "Cow / Buffalo", "Horse / Donkey", "Bird", "Multiple species", "Unknown / Other"])
            a_count    = st.text_input("Approximate number of animals affected", placeholder="e.g. 1, 3, 10+")
        with c2:
            a_location = st.text_input("Location / Address *", placeholder="e.g. Near Koramangala Water Tank, Bangalore")
            a_date     = st.date_input("When did this happen / when did you witness it?", value=date.today())
            a_urgency  = st.radio("Is the animal still in danger right now?", ["Yes — needs immediate help", "Possibly / Unknown", "No — reporting after the fact"], horizontal=False)

        a_description = st.text_area(
            "Describe what you witnessed *",
            height=120,
            placeholder="Please describe the abuse in as much detail as possible — what you saw, heard, and when. Any identifying features of the abuser (if known) are helpful.",
        )
        a_evidence = st.text_input("Do you have photos or video?", placeholder="e.g. Yes, available on request / Uploaded to Google Drive link / No")
        a_contact  = st.text_input("Your contact (optional — to follow up)", placeholder="Phone or email — kept confidential, never shared with the abuser")
        a_anon     = st.checkbox("Submit anonymously (your contact details will not be recorded)")

        if st.form_submit_button("Submit Report →", type="primary"):
            if a_type and a_location.strip() and a_description.strip():
                report = {
                    "type": a_type, "species": a_species, "count": a_count,
                    "location": a_location, "date": str(a_date), "urgency": a_urgency,
                    "description": a_description, "evidence": a_evidence,
                    "contact": "Anonymous" if a_anon else a_contact,
                    "submitted": datetime.now().strftime("%d %b %Y %H:%M"),
                }
                st.session_state.abuse_reports.append(report)
                st.success("✅ Report submitted. Thank you for speaking up for animals.")
                if "immediate" in a_urgency:
                    st.error("🚨 This animal needs immediate help — please also call 112 or CUPA (080-22947300) right now.")
                st.rerun()
            else:
                st.error("Please fill in all required fields (Type, Location, Description).")

    if st.session_state.abuse_reports:
        st.markdown("---")
        st.markdown(f"#### 📋 Reports Filed This Session ({len(st.session_state.abuse_reports)})")
        for i, rep in enumerate(st.session_state.abuse_reports):
            urgency_color = "#c0392b" if "immediate" in rep["urgency"] else "#e9c46a" if "Possibly" in rep["urgency"] else "#52b788"
            st.markdown(f"""
            <div style="background:white; border:1px solid var(--border); border-left:4px solid {urgency_color}; border-radius:10px; padding:1rem 1.2rem; margin-bottom:8px; box-shadow:var(--card-shadow);">
              <div style="font-weight:700; color:var(--blue-dark);">#{i+1} — {rep['type']} · {rep['species']}</div>
              <div style="font-size:0.85rem; color:#4a5568; margin-top:3px;">📍 {rep['location']} · 📅 {rep['date']} · Submitted {rep['submitted']}</div>
              <div style="font-size:0.83rem; color:#718096; margin-top:3px; font-style:italic;">"{rep['description'][:120]}{'...' if len(rep['description'])>120 else ''}"</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# REPORT ILLEGAL BREEDER
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.page == "breeder":

    st.markdown("""
    <div style="background:#4a1942; color:white; border-radius:10px; padding:12px 18px; margin-bottom:1.2rem; font-weight:700; font-size:1rem;">
      🔎 ILLEGAL BREEDER REPORTING — Help stop puppy mills and unethical breeding operations.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="page-header"><div class="section-header">🔎 Report an Illegal or Unethical Breeder</div><div class="section-sub">Illegal breeders and puppy mills cause immense suffering. Your report helps authorities and NGOs investigate and shut them down.</div></div>', unsafe_allow_html=True)

    tab_report, tab_signs, tab_law = st.tabs(["📝 Submit a Report", "⚠️ Warning Signs", "⚖️ Legal Framework"])

    with tab_report:
        st.markdown("#### 📞 Who to contact")
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            st.markdown("""
            <div style="background:white; border:1px solid var(--border); border-left:4px solid #c0392b; border-radius:10px; padding:0.9rem; box-shadow:var(--card-shadow);">
              <div style="font-weight:700; color:#c0392b; font-size:0.9rem;">🏛️ AWBI</div>
              <div style="font-size:0.83rem; color:#4a5568; margin-top:3px;">1800-11-4444 (Toll-Free)<br>awbi@nic.in</div>
            </div>
            """, unsafe_allow_html=True)
        with col_c2:
            st.markdown("""
            <div style="background:white; border:1px solid var(--border); border-left:4px solid #2980b9; border-radius:10px; padding:0.9rem; box-shadow:var(--card-shadow);">
              <div style="font-weight:700; color:#1a5276; font-size:0.9rem;">🐾 CUPA Investigations</div>
              <div style="font-size:0.83rem; color:#4a5568; margin-top:3px;">080-22947300<br>Bangalore-based</div>
            </div>
            """, unsafe_allow_html=True)
        with col_c3:
            st.markdown("""
            <div style="background:white; border:1px solid var(--border); border-left:4px solid #52b788; border-radius:10px; padding:0.9rem; box-shadow:var(--card-shadow);">
              <div style="font-weight:700; color:#1a3d2b; font-size:0.9rem;">🚔 Police / FIR</div>
              <div style="font-size:0.83rem; color:#4a5568; margin-top:3px;">112 or local station<br>PCA Act Sec 11</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        
        # Karnataka State Animal Welfare Board
        st.markdown("### 🏛️ Karnataka State Animal Welfare Board (KSAWB)")
        st.markdown("""
        <div style="background:#e3f2fd; border:2px solid #2563eb; border-radius:12px; padding:1.2rem; margin-bottom:1.2rem;">
          <div style="font-weight:700; color:#1e40af; font-size:0.95rem; margin-bottom:0.8rem;">📍 State Authority for Illegal Breeder Cases</div>
          
          <div style="display:grid; grid-template-columns:1fr 1fr; gap:1rem; font-size:0.85rem; color:#1f2937; line-height:1.6;">
            <div>
              <div style="font-weight:600; color:#1e40af; margin-bottom:0.4rem;">📞 Contact Details</div>
              <div>📞 Phone: 080-2258-6595</div>
              <div>📧 Email: ksawb@gmail.com</div>
              <div>🌐 Website: ksawb.karnataka.gov.in</div>
            </div>
            <div>
              <div style="font-weight:600; color:#1e40af; margin-bottom:0.4rem;">📍 Office Location</div>
              <div>Dr. Rajendra Prasad Rd, Ashokanagar,</div>
              <div>Bengaluru, Karnataka 560001</div>
              <div style="margin-top:0.4rem; font-size:0.8rem; color:#666;">Working Hours: Mon-Fri, 10 AM - 5 PM</div>
            </div>
          </div>
          
          <div style="margin-top:1rem; padding-top:1rem; border-top:1px solid #90caf9; font-size:0.83rem; color:#1f2937;">
            <div style="font-weight:600; color:#1e40af; margin-bottom:0.4rem;">✓ KSAWB Authority on Breeding</div>
            <ul style="margin:0.4rem 0 0 1.2rem; padding:0;">
              <li>Approves/rejects licenses for breeding facilities and kennels</li>
              <li>Investigates illegal and unethical breeding operations</li>
              <li>Files FIRs under PCA Act & animal cruelty statutes</li>
              <li>Conducts on-site inspections of breeding premises</li>
              <li>Issues notices and orders for closure of illegal operations</li>
              <li>Maintains registry of recognized animal breeders</li>
            </ul>
          </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("#### 📝 Report Form")

        with st.form("breeder_form"):
            c1, c2 = st.columns(2)
            with c1:
                b_type    = st.selectbox("Violation type *", [
                    "Unlicensed commercial breeding (>2 litters/year without AWBI registration)",
                    "Puppy mill / factory farming conditions",
                    "Selling underage puppies (< 8 weeks)",
                    "Falsified pedigree / breed certificates",
                    "Cruel breeding practices (forced breeding, no vet care)",
                    "Breeding banned / exotic species illegally",
                    "Online sale without verification",
                    "Other illegal breeding activity",
                ])
                b_species = st.multiselect("Species involved *", ["Dogs", "Cats", "Exotic birds", "Reptiles", "Primates", "Other exotic animals"])
                b_scale   = st.selectbox("Estimated scale", ["Small (< 10 animals)", "Medium (10–50 animals)", "Large (50+ animals / commercial operation)", "Unknown"])
            with c2:
                b_name    = st.text_input("Breeder / business name (if known)", placeholder="e.g. XYZ Kennels, or a person's name")
                b_location = st.text_input("Location / Address *", placeholder="e.g. Whitefield, Bangalore / specific street address")
                b_online  = st.text_input("Online listings (if any)", placeholder="e.g. OLX ad link, Instagram handle, website URL")

            b_description = st.text_area(
                "Describe what you know *",
                height=120,
                placeholder="Describe conditions, practices, or specific violations you have witnessed or have evidence of. Include dates if possible.",
            )
            b_evidence = st.text_input("Evidence available?", placeholder="e.g. Photos, videos, receipts, veterinary records, chat screenshots")
            b_contact  = st.text_input("Your contact (optional — to follow up)", placeholder="Phone or email — confidential")
            b_anon     = st.checkbox("Submit anonymously")

            if st.form_submit_button("Submit Breeder Report →", type="primary"):
                if b_location.strip() and b_description.strip() and b_species:
                    report = {
                        "type": b_type, "species": ", ".join(b_species), "scale": b_scale,
                        "name": b_name, "location": b_location, "online": b_online,
                        "description": b_description, "evidence": b_evidence,
                        "contact": "Anonymous" if b_anon else b_contact,
                        "submitted": datetime.now().strftime("%d %b %Y %H:%M"),
                    }
                    st.session_state.breeder_reports.append(report)
                    st.success("✅ Breeder report submitted. Thank you for helping protect animals.")
                    st.rerun()
                else:
                    st.error("Please fill in Location, Species, and Description.")

        if st.session_state.breeder_reports:
            st.markdown("---")
            st.markdown(f"#### 📋 Reports Filed This Session ({len(st.session_state.breeder_reports)})")
            for i, rep in enumerate(st.session_state.breeder_reports):
                st.markdown(f"""
                <div style="background:white; border:1px solid var(--border); border-left:4px solid #9b59b6; border-radius:10px; padding:1rem 1.2rem; margin-bottom:8px; box-shadow:var(--card-shadow);">
                  <div style="font-weight:700; color:var(--blue-dark);">#{i+1} — {rep['type'][:60]}{'...' if len(rep['type'])>60 else ''}</div>
                  <div style="font-size:0.85rem; color:#4a5568; margin-top:3px;">📍 {rep['location']} · 🐾 {rep['species']} · Submitted {rep['submitted']}</div>
                  {f"<div style='font-size:0.83rem; color:#718096; margin-top:3px;'>🌐 {rep['online']}</div>" if rep.get('online') else ""}
                </div>
                """, unsafe_allow_html=True)

    with tab_signs:
        st.markdown("### ⚠️ Warning Signs of an Illegal or Unethical Breeder")
        signs = [
            ("🔴", "Always has puppies available", "Reputable breeders have waiting lists — if they always have stock, they're likely overbreeding."),
            ("🔴", "Sells puppies under 8 weeks old", "Legally and developmentally, puppies must stay with their mother for at least 8 weeks."),
            ("🔴", "Refuses to show you the parents or premises", "A legitimate breeder will always let you visit and meet the mother at minimum."),
            ("🟡", "No health certificates or vet records", "Ethical breeders screen for genetic conditions and provide vaccination records."),
            ("🟡", "Extremely low or suspiciously high prices", "Prices far below or above market rate can indicate fraud or exploitation."),
            ("🟡", "Multiple breeds available at once", "Breeding multiple breeds suggests commercial operation, not genuine enthusiasm for a breed."),
            ("🟡", "Pushes for quick sale / 'only one left'", "High-pressure tactics are a red flag — reputable breeders screen buyers carefully."),
            ("🔴", "Animals appear sickly, underweight, or fearful", "Poor condition of animals is a direct indicator of neglect or cruelty."),
            ("🔴", "No AWBI registration (required for > 2 litters/year)", "All commercial breeders must register with AWBI under the Dog Breeding and Marketing Rules, 2017."),
            ("🟡", "Sells via OLX, Quikr, or social media without verification", "Responsible breeders vet buyers — anonymous online sales with no checks are a red flag."),
        ]
        for color, title, detail in signs:
            bg = "#fff5f5" if color == "🔴" else "#fffbeb"
            border = "#fed7d7" if color == "🔴" else "#fde68a"
            text = "#7d1515" if color == "🔴" else "#78350f"
            st.markdown(f"""
            <div style="background:{bg}; border:1px solid {border}; border-radius:8px; padding:11px 14px; margin-bottom:7px;">
              <div style="font-weight:700; color:{text}; font-size:0.91rem;">{color} {title}</div>
              <div style="font-size:0.85rem; color:#4a5568; margin-top:3px; line-height:1.5;">{detail}</div>
            </div>
            """, unsafe_allow_html=True)

    with tab_law:
        st.markdown("### ⚖️ Legal Framework for Breeding Regulation in India")
        laws = [
            ("Prevention of Cruelty to Animals Act, 1960",
             "The principal law. Section 11 covers cruelty, improper confinement, and failure to provide adequate care. Penalties include fines up to ₹50,000 and/or 5 years imprisonment for repeat offences."),
            ("Dog Breeding and Marketing Rules, 2017 (AWBI)",
             "Mandates registration for breeders producing more than 2 litters per year. Requires minimum space, veterinary care, record-keeping, and humane treatment standards."),
            ("Wildlife Protection Act, 1972",
             "Prohibits breeding, trading, or keeping Schedule I and II species without a permit. Violators face up to 7 years imprisonment and substantial fines."),
            ("Foreign Trade (Development & Regulation) Act, 1992",
             "Governs import/export of animals. Illegal trade in exotic animals is also prosecuted under CITES (Convention on International Trade in Endangered Species)."),
            ("Indian Penal Code — Section 428 & 429",
             "Covers killing or maiming animals. Applicable in cases of culling litters, killing sick animals, or any deliberate harm."),
        ]
        for law, detail in laws:
            st.markdown(f"""
            <div style="background:white; border:1px solid var(--border); border-left:4px solid #9b59b6; border-radius:10px; padding:1rem 1.2rem; margin-bottom:10px; box-shadow:var(--card-shadow);">
              <div style="font-weight:700; color:#6c3483; font-size:0.92rem;">⚖️ {law}</div>
              <div style="font-size:0.85rem; color:#4a5568; margin-top:4px; line-height:1.55;">{detail}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background:#e8f4fd; border:1px solid #bee3f8; border-radius:10px; padding:12px 16px; font-size:0.87rem; color:#1a5276; line-height:1.6; margin-top:8px;">
          <strong>📌 How to file a formal complaint:</strong><br>
          1. Collect evidence (photos, video, receipts, chat logs) before reporting.<br>
          2. File an FIR at your local police station citing the relevant Act and section.<br>
          3. Simultaneously report to AWBI (awbi@nic.in) and a local NGO such as CUPA.<br>
          4. Follow up within 7 days — ask for an acknowledgement number.<br>
          5. If police are unresponsive, file a complaint with the District Magistrate or approach the High Court.
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("""
<div style="display:flex; justify-content:space-between; align-items:center; font-size:0.75rem; color:#a0aec0; flex-wrap:wrap; gap:8px;">
  <div>🐾 <strong style="color:#2d6a4f;">VetConnect AI</strong> · Animal Welfare Platform · Microsoft Hackathon 2025</div>
  <div>🔒 All AI runs locally · No data leaves your device · Not a substitute for veterinary care</div>
</div>
""", unsafe_allow_html=True)
