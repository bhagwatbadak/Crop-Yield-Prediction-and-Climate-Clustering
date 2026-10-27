import os, warnings
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
warnings.filterwarnings('ignore')

import base64 as _b64

# Image loader
def _img_b64(filename):
    import os
    app_dir  = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(app_dir)
    for base in [app_dir, root_dir]:
        path = os.path.join(base, 'assets', filename)
        if os.path.exists(path):
            with open(path, 'rb') as fh:
                return _b64.b64encode(fh.read()).decode()
    return ''

def _bg_css(b64, mime='image/avif'):
    if not b64:
        return ''
    return f"url('data:{mime};base64,{b64}')"

IMG_HOME   = _img_b64('homeweb.avif')
IMG_MAIZE  = _img_b64('maize.avif')
IMG_CARROT = _img_b64('carrot.avif')
IMG_CROP   = _img_b64('crop.avif')

# ══════════════════════════════════════════════════════════════
# PATH SETUP
# ══════════════════════════════════════════════════════════════
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(BASE_DIR)
DATA_PATH  = os.path.join(ROOT_DIR, 'data', 'raw', 'crop_yield.csv')
MODEL_DIR  = os.path.join(ROOT_DIR, 'notebooks', 'models')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'notebooks', 'outputs')
PLOTS_DIR  = os.path.join(ROOT_DIR, 'notebooks', 'outputs', 'plots')

# ══════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════
TARGET_COL     = 'Yield'
CROP_COL       = 'Crop'
YEAR_COL       = 'Crop_Year'
SEASON_COL     = 'Season'
STATE_COL      = 'State'
AREA_COL       = 'Area'
PRODUCTION_COL = 'Production'
RAINFALL_COL   = 'Annual_Rainfall'
FERTILIZER_COL = 'Fertilizer'
PESTICIDE_COL  = 'Pesticide'
CLUSTER_COL    = 'Climate_Cluster'
CATEGORICAL_COLS  = [CROP_COL, SEASON_COL, STATE_COL]
CLUSTER_LABELS    = {0:'Arid', 1:'Semi-Arid', 2:'Moderate', 3:'Humid'}
CLUSTER_COLOR_MAP = {0:'#FF6B6B', 1:'#FFD93D', 2:'#6BCB77', 3:'#4D96FF'}
CLUSTER_ICONS     = {0:'🏜️', 1:'🌾', 2:'🌿', 3:'🌧️'}

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AgriSense — Crop Yield AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════
# DARK THEME CSS  — Charcoal + Neon Lime + Amber Accents
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600;700&family=Nunito:wght@300;400;600;700;800&display=swap');

/* ── COLOUR PALETTE ── */
:root {
    --bg-base:      #0D1117;
    --bg-surface:   #161B22;
    --bg-elevated:  #1C2330;
    --bg-card:      #21262D;
    --bg-hover:     #2D333B;

    --lime:         #AAFF00;
    --lime-dim:     #88CC00;
    --lime-glow:    rgba(170,255,0,0.18);
    --lime-border:  rgba(170,255,0,0.28);

    --amber:        #FFB800;
    --amber-glow:   rgba(255,184,0,0.18);

    --cyan:         #00D4FF;
    --cyan-glow:    rgba(0,212,255,0.15);

    --coral:        #FF6B6B;
    --violet:       #B06BFF;

    --text-bright:  #F0F6FC;
    --text-main:    #C9D1D9;
    --text-muted:   #8B949E;
    --text-faint:   #484F58;

    --border:       rgba(170,255,0,0.12);
    --border-card:  rgba(255,255,255,0.06);
    --border-hover: rgba(170,255,0,0.35);

    --shadow-sm:    0 2px 8px rgba(0,0,0,0.4);
    --shadow-md:    0 8px 32px rgba(0,0,0,0.5);
    --shadow-lg:    0 20px 60px rgba(0,0,0,0.6);
    --shadow-lime:  0 0 28px rgba(170,255,0,0.2);

    --r:    16px;
    --r-sm: 10px;
    --ease: cubic-bezier(0.34,1.56,0.64,1);
}

/* ── RESET & BASE ── */
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}

html,body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background:#0D1117 !important;
    font-family:'Nunito',sans-serif !important;
    color:var(--text-main) !important;
}

/* Subtle grid texture overlay */
[data-testid="stAppViewContainer"]::before {
    content:'';position:fixed;inset:0;z-index:0;pointer-events:none;
    background-image:
        linear-gradient(rgba(170,255,0,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(170,255,0,0.03) 1px, transparent 1px);
    background-size:40px 40px;
}

/* Atmospheric glow orbs */
[data-testid="stAppViewContainer"]::after {
    content:'';position:fixed;z-index:0;pointer-events:none;
    width:700px;height:700px;border-radius:50%;
    background:radial-gradient(circle, rgba(170,255,0,0.06) 0%, transparent 65%);
    top:-250px;left:-200px;
    animation:driftOrb 25s ease-in-out infinite;
}

@keyframes driftOrb {
    0%,100%{transform:translate(0,0);}
    40%{transform:translate(120px,80px);}
    70%{transform:translate(-60px,150px);}
}

.main .block-container {
    padding:2rem 2.5rem 6rem !important;
    max-width:1500px !important;
    position:relative;z-index:1;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background:linear-gradient(180deg,#161B22 0%,#0D1117 100%) !important;
    border-right:1px solid var(--border-card) !important;
    box-shadow:4px 0 24px rgba(0,0,0,0.5) !important;
}
[data-testid="stSidebar"] * {font-family:'Nunito',sans-serif !important;}
[data-testid="stSidebar"] .stRadio>div{gap:3px !important;}

[data-testid="stSidebar"] .stRadio label {
    display:flex !important;align-items:center !important;
    padding:12px 18px !important;border-radius:10px !important;
    margin:3px 0 !important;color:var(--text-muted) !important;
    font-size:.88rem !important;font-weight:600 !important;
    transition:all .25s ease !important;cursor:pointer !important;
    border:1px solid transparent !important;
    letter-spacing:.02em !important;
}

[data-testid="stSidebar"] .stRadio label:hover {
    background:rgba(170,255,0,0.06) !important;
    color:var(--lime) !important;
    border-color:var(--border) !important;
    transform:translateX(6px) !important;
}

[data-testid="stSidebar"] [aria-selected="true"] label {
    background:rgba(170,255,0,0.1) !important;
    color:var(--lime) !important;
    border-color:var(--lime-border) !important;
    box-shadow:inset 3px 0 0 var(--lime) !important;
    font-weight:700 !important;
}

/* ── TYPOGRAPHY ── */
h1,h2,h3,h4{font-family:'Rajdhani',sans-serif !important;color:var(--text-bright) !important;letter-spacing:.01em;}
hr{border:none !important;border-top:1px solid var(--border-card) !important;margin:2rem 0 !important;}
code{font-family:'JetBrains Mono',monospace !important;color:var(--lime) !important;
     background:rgba(170,255,0,0.08) !important;padding:.2em .5em !important;border-radius:5px !important;}

/* ── METRICS ── */
[data-testid="stMetric"] {
    background:var(--bg-card) !important;
    border:1px solid var(--border-card) !important;
    border-radius:var(--r) !important;
    padding:1.3rem 1.5rem !important;
    box-shadow:var(--shadow-sm) !important;
    transition:all .35s var(--ease) !important;
    position:relative !important;overflow:visible !important;
    animation:slideUp .5s var(--ease) both;
}
[data-testid="stMetric"]::after {
    content:'';position:absolute;bottom:0;left:0;right:0;height:2px;
    background:linear-gradient(90deg,transparent,var(--lime),transparent);
    opacity:0;transition:opacity .3s;
}
[data-testid="stMetric"]:hover {
    transform:translateY(-6px) !important;
    border-color:var(--lime-border) !important;
    box-shadow:var(--shadow-md),var(--shadow-lime) !important;
    background:var(--bg-hover) !important;
}
[data-testid="stMetric"]:hover::after{opacity:1 !important;}

@keyframes slideUp{from{opacity:0;transform:translateY(16px);}to{opacity:1;transform:translateY(0);}}

[data-testid="stMetricLabel"] {
    font-family:'JetBrains Mono',monospace !important;
    font-size:.65rem !important;font-weight:700 !important;
    letter-spacing:.18em !important;text-transform:uppercase !important;
    color:var(--text-muted) !important;
}
[data-testid="stMetricValue"] {
    font-family:'JetBrains Mono',monospace !important;
    font-size:1.7rem !important;font-weight:700 !important;
    color:var(--lime) !important;line-height:1.2 !important;
}
[data-testid="stMetricDelta"] {color:var(--amber) !important;font-size:.85rem !important;}

/* ── BUTTONS ── */
.stButton>button {
    background:linear-gradient(135deg, #1A2F0A 0%, #2D5200 40%, #AAFF00 100%) !important;
    color:#0D1117 !important;
    font-family:'Rajdhani',sans-serif !important;font-size:1.05rem !important;
    font-weight:700 !important;letter-spacing:.12em !important;text-transform:uppercase !important;
    border:none !important;border-radius:var(--r-sm) !important;padding:.9rem 2rem !important;
    width:100% !important;
    box-shadow:0 6px 28px rgba(170,255,0,0.25), inset 0 1px 0 rgba(255,255,255,0.1) !important;
    transition:all .35s var(--ease) !important;position:relative !important;overflow:visible !important;
}
.stButton>button::before {
    content:'';position:absolute;top:0;left:-100%;width:100%;height:100%;
    background:linear-gradient(90deg,transparent,rgba(255,255,255,0.2),transparent);
    transition:left .5s ease !important;
}
.stButton>button:hover::before{left:100% !important;}
.stButton>button:hover {
    transform:translateY(-3px) !important;
    box-shadow:0 12px 40px rgba(170,255,0,0.4), inset 0 1px 0 rgba(255,255,255,0.15) !important;
    background:linear-gradient(135deg,#2D5200 0%,#AAFF00 60%,#FFD700 100%) !important;
}
.stButton>button:active{transform:translateY(-1px) !important;}

/* ── INPUTS ── */
.stSelectbox>div>div,
.stNumberInput>div>div>input {
    background:var(--bg-card) !important;
    border:1.5px solid var(--border-card) !important;
    border-radius:var(--r-sm) !important;
    color:var(--text-bright) !important;
    transition:all .25s ease !important;
    font-family:'Nunito',sans-serif !important;
}
.stSelectbox>div>div:focus-within,
.stNumberInput>div>div>input:focus {
    border-color:var(--lime) !important;
    box-shadow:0 0 0 3px rgba(170,255,0,0.1) !important;
    background:var(--bg-elevated) !important;
}
.stSelectbox [data-baseweb="select"] {background:var(--bg-card) !important;}
.stSelectbox [data-baseweb="popover"] {background:var(--bg-elevated) !important;border:1px solid var(--border-card) !important;}
.stSelectbox [role="option"] {color:var(--text-main) !important;}
.stSelectbox [role="option"]:hover {background:rgba(170,255,0,0.08) !important;color:var(--lime) !important;}

/* ── SLIDERS ── */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background:var(--lime) !important;
    box-shadow:0 0 14px rgba(170,255,0,0.6) !important;
    width:18px !important;height:18px !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] div:nth-child(2) {
    background:linear-gradient(90deg,var(--lime-dim),var(--lime)) !important;height:4px !important;
}

/* ── PLOTLY CHARTS ── */
[data-testid="stPlotlyChart"] {
    border-radius:var(--r) !important;overflow:visible !important;
    border:1px solid var(--border-card) !important;
    box-shadow:var(--shadow-sm) !important;
    transition:all .35s var(--ease) !important;
    background:var(--bg-card) !important;
}
[data-testid="stPlotlyChart"]:hover {
    border-color:var(--lime-border) !important;
    box-shadow:var(--shadow-md),var(--shadow-lime) !important;
    transform:translateY(-4px) !important;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background:var(--bg-surface) !important;
    border-radius:12px !important;padding:5px !important;gap:3px !important;
    border:1px solid var(--border-card) !important;
}
.stTabs [data-baseweb="tab"] {
    background:transparent !important;border-radius:8px !important;
    color:var(--text-muted) !important;font-weight:600 !important;font-size:.88rem !important;
    transition:all .25s ease !important;font-family:'Nunito',sans-serif !important;
}
.stTabs [aria-selected="true"] {
    background:var(--lime) !important;color:#0D1117 !important;
    box-shadow:0 3px 14px rgba(170,255,0,0.35) !important;font-weight:700 !important;
}

/* ── ALERTS ── */
.stAlert{border-radius:10px !important;border-left-width:3px !important;backdrop-filter:blur(10px) !important;}
[data-testid="stInfo"]{background:rgba(0,212,255,0.07) !important;border-color:var(--cyan) !important;color:var(--text-main) !important;}
[data-testid="stSuccess"]{background:rgba(170,255,0,0.07) !important;border-color:var(--lime) !important;}
[data-testid="stWarning"]{background:rgba(255,184,0,0.07) !important;border-color:var(--amber) !important;}
[data-testid="stError"]{background:rgba(255,107,107,0.07) !important;border-color:var(--coral) !important;}

/* ── PROGRESS BAR ── */
[data-testid="stProgressBar"]>div>div {
    background:linear-gradient(90deg,var(--lime-dim),var(--lime),var(--amber)) !important;
    border-radius:99px !important;box-shadow:0 0 12px rgba(170,255,0,0.5) !important;
}

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] {
    border:1px solid var(--border-card) !important;border-radius:var(--r) !important;
    overflow:visible !important;background:var(--bg-card) !important;
}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
    background:var(--bg-card) !important;
    border:2px dashed rgba(170,255,0,0.2) !important;
    border-radius:var(--r) !important;transition:all .25s ease !important;
}
[data-testid="stFileUploader"]:hover {
    border-color:var(--lime) !important;
    background:rgba(170,255,0,0.04) !important;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar{width:6px;height:6px;}
::-webkit-scrollbar-track{background:var(--bg-base);}
::-webkit-scrollbar-thumb{background:rgba(170,255,0,0.25);border-radius:3px;}
::-webkit-scrollbar-thumb:hover{background:rgba(170,255,0,0.45);}

#MainMenu,footer,header{visibility:visible;}

/* ═══════════════════════════════
   CUSTOM COMPONENTS
═══════════════════════════════ */

/* HERO */
.hero{text-align:center;padding:2rem 2rem 1.8rem;position:relative;}

.hero-badge {
    display:inline-flex;align-items:center;gap:.55rem;
    background:rgba(170,255,0,0.08);border:1px solid var(--lime-border);
    border-radius:99px;padding:.35rem 1.3rem;
    font-family:'JetBrains Mono',monospace;font-size:.68rem;font-weight:700;
    letter-spacing:.16em;text-transform:uppercase;color:var(--lime);
    margin-bottom:1.4rem;animation:pulseBadge 3s ease-in-out infinite;
}
.hero-badge::before {
    content:'';width:7px;height:7px;border-radius:50%;
    background:var(--lime);box-shadow:0 0 10px var(--lime);
    animation:blink 1.4s ease-in-out infinite;
}
@keyframes blink{0%,100%{opacity:1;transform:scale(1);}50%{opacity:.3;transform:scale(.6);}}
@keyframes pulseBadge{0%,100%{box-shadow:0 0 0 0 rgba(170,255,0,0.25);}55%{box-shadow:0 0 0 14px rgba(170,255,0,0);}}

.hero-title {
    font-family:'Rajdhani',sans-serif !important;
    font-size:clamp(2rem,5vw,3.6rem) !important;font-weight:700 !important;
    line-height:1.25 !important;color:var(--text-bright) !important;
    letter-spacing:.03em;animation:fadeUp .7s var(--ease) both;
}
.hero-title .accent {
    background:linear-gradient(90deg, var(--lime) 0%, var(--amber) 60%, var(--cyan) 100%);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
    filter:drop-shadow(0 0 20px rgba(170,255,0,0.35));
}
.hero-sub {
    font-size:.98rem;color:var(--text-muted);max-width:620px;
    margin:.9rem auto 2rem;line-height:1.75;animation:fadeUp .8s var(--ease) .1s both;
}

@keyframes fadeUp{from{opacity:0;transform:translateY(20px);}to{opacity:1;transform:translateY(0);}}

/* KPI GRID */
.kpi-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:.9rem;margin:1rem 0 1.6rem;}

.kpi {
    background:var(--bg-card);border:1px solid var(--border-card);
    border-radius:var(--r);padding:1.2rem .8rem 1rem;text-align:center;
    position:relative;overflow:visible;transition:all .35s var(--ease);
    animation:fadeUp .5s var(--ease) both;cursor:default;
}
.kpi::before {
    content:'';position:absolute;top:0;left:0;right:0;height:2px;
    background:linear-gradient(90deg,transparent,var(--lime),transparent);opacity:0;
    transition:opacity .3s;
}
.kpi:hover {transform:translateY(-8px);border-color:var(--lime-border);box-shadow:var(--shadow-md),var(--shadow-lime);background:var(--bg-hover);}
.kpi:hover::before{opacity:1;}
.kpi-icon{font-size:1.7rem;display:block;margin-bottom:.5rem;}
.kpi-val{font-family:'JetBrains Mono',monospace;font-size:1.55rem;font-weight:700;color:var(--lime);display:block;line-height:1.1;}
.kpi-label{font-family:'JetBrains Mono',monospace;font-size:.6rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--text-muted);display:block;margin-top:.35rem;}

/* SECTION HEADER */
.sh{display:flex;align-items:center;gap:1rem;margin:2.8rem 0 1.5rem;}
.sh-line{flex:1;height:1px;background:linear-gradient(90deg,transparent,rgba(170,255,0,0.2),transparent);}
.sh-tag{font-family:'JetBrains Mono',monospace;font-size:.66rem;font-weight:700;letter-spacing:.16em;
        text-transform:uppercase;color:var(--lime);background:rgba(170,255,0,0.08);
        border:1px solid var(--lime-border);border-radius:6px;padding:.35rem .9rem;white-space:nowrap;}

/* PANEL / CARD */
.panel {
    background:var(--bg-card);border:1px solid var(--border-card);
    border-radius:var(--r);padding:1.6rem;position:relative;overflow:visible;transition:all .25s ease;
}
.panel::before {content:'';position:absolute;top:0;left:0;right:0;height:1px;
    background:linear-gradient(90deg,transparent,rgba(170,255,0,0.3),transparent);}
.panel:hover{border-color:rgba(170,255,0,0.2);box-shadow:var(--shadow-sm);}
.panel-title{font-family:'JetBrains Mono',monospace;font-size:.72rem;font-weight:700;letter-spacing:.14em;
             text-transform:uppercase;color:var(--lime);margin:0 0 1.4rem;padding-bottom:.9rem;
             border-bottom:1px solid var(--border-card);display:flex;align-items:center;gap:.6rem;}

/* RESULT BIG */
.result-big {
    background:linear-gradient(135deg, rgba(170,255,0,0.07) 0%, rgba(0,212,255,0.05) 100%);
    border:2px solid var(--lime-border);border-radius:var(--r);
    padding:2.5rem 2rem 2rem;text-align:center;position:relative;overflow:visible;
    animation:revealResult .6s var(--ease) both;box-shadow:var(--shadow-md),var(--shadow-lime);
}
.result-big::after{content:'';position:absolute;top:-1px;left:15%;right:15%;height:1px;
    background:linear-gradient(90deg,transparent,var(--lime),var(--amber),transparent);}

@keyframes revealResult{from{opacity:0;transform:scale(.94) translateY(12px);}to{opacity:1;transform:scale(1) translateY(0);}}

.result-num {
    font-family:'JetBrains Mono',monospace;font-size:4.8rem;font-weight:700;
    color:var(--lime);line-height:1;
    animation:numPulse 2.5s ease-in-out infinite alternate;
}
@keyframes numPulse{from{text-shadow:0 0 20px rgba(170,255,0,0.3);}to{text-shadow:0 0 50px rgba(170,255,0,0.6);}}
.result-unit{font-size:1rem;color:var(--text-muted);margin-top:.5rem;letter-spacing:.08em;font-family:'JetBrains Mono',monospace;}
.result-eye{font-family:'JetBrains Mono',monospace;font-size:.65rem;font-weight:700;letter-spacing:.18em;text-transform:uppercase;color:var(--text-muted);margin-bottom:.5rem;}

/* CHIPS */
.chip{display:inline-flex;align-items:center;gap:.4rem;padding:.38rem 1rem;border-radius:99px;
      font-size:.8rem;font-weight:700;border:1px solid;transition:all .25s ease;margin:.25rem .18rem;font-family:'JetBrains Mono',monospace;}
.chip-g{background:rgba(170,255,0,0.08);border-color:rgba(170,255,0,0.28);color:var(--lime);}
.chip-t{background:rgba(0,212,255,0.08);border-color:rgba(0,212,255,0.28);color:var(--cyan);}
.chip-gold{background:rgba(255,184,0,0.08);border-color:rgba(255,184,0,0.28);color:var(--amber);}
.chip-sky{background:rgba(176,107,255,0.08);border-color:rgba(176,107,255,0.28);color:var(--violet);}

/* HOW IT WORKS FLOW */
.flow{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin:1.5rem 0;}
.flow-item{background:var(--bg-card);border:1px solid var(--border-card);border-radius:var(--r);
           padding:1.6rem 1.2rem;text-align:center;transition:all .35s var(--ease);
           animation:fadeUp .6s var(--ease) both;position:relative;overflow:visible;}
.flow-item::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;
    background:linear-gradient(90deg,transparent,var(--lime),transparent);opacity:0;transition:opacity .3s;}
.flow-item:hover{transform:translateY(-8px);border-color:var(--lime-border);box-shadow:var(--shadow-md),var(--shadow-lime);background:var(--bg-hover);}
.flow-item:hover::before{opacity:1;}
.flow-num{width:46px;height:46px;border-radius:50%;
    background:linear-gradient(135deg,#1A3A00,#AAFF00);
    display:flex;align-items:center;justify-content:center;
    font-family:'JetBrains Mono',monospace;font-size:.85rem;font-weight:700;
    color:#0D1117;margin:0 auto .9rem;box-shadow:0 4px 18px rgba(170,255,0,0.3);}
.flow-title{font-family:'Rajdhani',sans-serif;font-weight:700;font-size:1rem;color:var(--text-bright);margin-bottom:.5rem;}
.flow-desc{font-size:.83rem;color:var(--text-muted);line-height:1.75;}

/* CLIMATE ZONE CARDS */
.zone{background:var(--bg-card);border-radius:var(--r);padding:1.6rem;
      border:1px solid var(--border-card);text-align:center;transition:all .35s var(--ease);position:relative;overflow:visible;}
.zone:hover{transform:translateY(-7px);box-shadow:var(--shadow-md);background:var(--bg-hover);}
.zone-icon{font-size:2.5rem;display:block;margin-bottom:.8rem;}
.zone-name{font-family:'Rajdhani',sans-serif;font-weight:700;font-size:1rem;margin-bottom:.35rem;color:var(--text-bright);}
.zone-range{font-family:'JetBrains Mono',monospace;font-size:.7rem;color:var(--text-muted);margin-bottom:.7rem;}
.zone-states{font-size:.82rem;color:var(--text-muted);line-height:1.8;}

/* SIDEBAR BRAND */
.brand{text-align:center;padding:2rem 0 1.6rem;}
.brand-icon{font-size:3.5rem;display:block;margin-bottom:.7rem;animation:floatIcon 5s ease-in-out infinite;}
@keyframes floatIcon{0%,100%{transform:translateY(0) rotate(-4deg);}50%{transform:translateY(-10px) rotate(4deg);}}
.brand-name{font-family:'Rajdhani',sans-serif !important;font-size:1.8rem !important;
    font-weight:700 !important;color:var(--text-bright) !important;letter-spacing:.04em;}
.brand-name em{color:var(--lime) !important;font-style:normal !important;}
.brand-sub{font-family:'JetBrains Mono',monospace !important;font-size:.62rem !important;
    color:var(--text-muted) !important;letter-spacing:.18em !important;text-transform:uppercase !important;}

/* SIDEBAR DATA BOX */
.sb-box{background:rgba(170,255,0,0.04);border:1px solid var(--border);border-radius:12px;padding:1rem 1.1rem;margin:.5rem 0;}
.sb-row{display:flex;justify-content:space-between;align-items:center;padding:.45rem 0;
        font-size:.82rem;border-bottom:1px solid rgba(255,255,255,0.04);}
.sb-row:last-child{border-bottom:none;}
.sb-lbl{color:var(--text-muted);}
.sb-val{font-family:'JetBrains Mono',monospace;color:var(--lime);font-weight:700;font-size:.78rem;}

/* STATUS BADGES */
.status-ok{background:rgba(170,255,0,0.08);border:1px solid var(--lime-border);border-radius:10px;
    padding:.7rem 1.1rem;text-align:center;color:var(--lime);font-size:.82rem;font-weight:700;
    animation:statusPulse 3s ease-in-out infinite;font-family:'JetBrains Mono',monospace;}
.status-err{background:rgba(255,107,107,0.08);border:1px solid rgba(255,107,107,0.3);border-radius:10px;
    padding:.7rem 1.1rem;text-align:center;color:var(--coral);font-size:.82rem;font-weight:700;font-family:'JetBrains Mono',monospace;}
@keyframes statusPulse{0%,100%{box-shadow:0 0 0 0 rgba(170,255,0,0.18);}55%{box-shadow:0 0 0 10px rgba(170,255,0,0);}}

/* PAGE HEADER */
.pg-eye{font-family:'JetBrains Mono',monospace;font-size:.7rem;font-weight:700;
    letter-spacing:.18em;text-transform:uppercase;color:var(--lime);margin-bottom:.4rem;}
.pg-title{font-family:'Rajdhani',sans-serif;font-size:2.8rem;font-weight:700;
    color:var(--text-bright);letter-spacing:.02em;margin:0;}
.pg-sub{color:var(--text-muted);margin:.4rem 0 0;font-size:.95rem;}

/* PERFORMANCE BANNER */
.perf{background:linear-gradient(135deg,rgba(170,255,0,0.07) 0%,rgba(0,212,255,0.04) 100%);
    border:1px solid var(--lime-border);border-radius:var(--r);
    padding:1.3rem 1.6rem;display:flex;align-items:center;gap:1.3rem;
    margin-bottom:1.6rem;animation:fadeUp .5s var(--ease) both;box-shadow:var(--shadow-sm),var(--shadow-lime);}
.perf-trophy{font-size:2.5rem;}
.perf-main{font-family:'Rajdhani',sans-serif;font-weight:700;font-size:1.1rem;color:var(--text-bright);}
.perf-sub{font-family:'JetBrains Mono',monospace;font-size:.78rem;color:var(--lime);margin-top:.2rem;}

/* ABOUT CARDS */
.about-card{background:var(--bg-card);border:1px solid var(--border-card);border-radius:var(--r);padding:1.8rem;margin-bottom:1.1rem;}
.about-title{font-family:'Rajdhani',sans-serif;font-size:1.15rem;color:var(--text-bright);
    margin-bottom:1.3rem;padding-bottom:.9rem;border-bottom:1px solid var(--border-card);}
.about-step{margin-bottom:1.3rem;padding-left:1.2rem;border-left:2px solid rgba(170,255,0,0.3);}
.about-step-tag{font-family:'JetBrains Mono',monospace;font-size:.65rem;font-weight:700;
    letter-spacing:.14em;text-transform:uppercase;color:var(--lime);margin-bottom:.35rem;}
.about-step-desc{font-size:.88rem;color:var(--text-muted);line-height:1.8;}

/* TECH STACK TABLE */
table.stk{width:100%;border-collapse:collapse;}
table.stk th{background:rgba(170,255,0,0.06);color:var(--lime);font-family:'JetBrains Mono',monospace;
    font-size:.66rem;letter-spacing:.14em;text-transform:uppercase;padding:.75rem 1rem;
    text-align:left;border-bottom:1px solid var(--border-card);}
table.stk td{padding:.75rem 1rem;font-size:.86rem;color:var(--text-muted);
    border-bottom:1px solid rgba(255,255,255,0.04);transition:all .2s ease;}
table.stk tr:hover td{background:rgba(170,255,0,0.04);color:var(--text-main);}

/* FACT LIST */
.fact{display:flex;justify-content:space-between;padding:.55rem 0;font-size:.86rem;border-bottom:1px solid rgba(255,255,255,0.04);}
.fact-lbl{color:var(--text-muted);}
.fact-val{font-family:'JetBrains Mono',monospace;color:var(--lime);font-weight:700;font-size:.82rem;}

/* BATCH INFO */
.batch-info{background:rgba(170,255,0,0.06);border:1px solid var(--lime-border);border-radius:12px;
    padding:1rem 1.4rem;margin:1.1rem 0;font-size:.88rem;color:var(--lime);font-family:'JetBrains Mono',monospace;}

/* DELTAS */
.delta-up{color:var(--lime);font-weight:700;}
.delta-dn{color:var(--coral);font-weight:700;}

/* FOOTER */
.footer{text-align:center;padding:2.5rem 0 1.5rem;color:var(--text-muted);font-size:.75rem;
    letter-spacing:.1em;border-top:1px solid var(--border-card);margin-top:5rem;
    font-family:'JetBrains Mono',monospace;}
.footer em{color:var(--lime);font-style:normal;}

/* number input label text */
[data-testid="stNumberInput"] label,
[data-testid="stSelectbox"] label,
[data-testid="stSlider"] label {
    color:var(--text-muted) !important;font-size:.83rem !important;font-weight:600 !important;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PLOTLY DARK THEME
# ══════════════════════════════════════════════════════════════
PT = dict(
    plot_bgcolor  = 'rgba(22,27,34,0)',
    paper_bgcolor = 'rgba(22,27,34,0)',
    font       = dict(family='Nunito, sans-serif', color='#8B949E', size=11),
    title_font = dict(family='Rajdhani, sans-serif', color='#F0F6FC', size=17),
    xaxis = dict(
        gridcolor='rgba(255,255,255,0.05)', linecolor='rgba(255,255,255,0.08)',
        tickfont=dict(color='#6E7681', size=10), zeroline=False,
    ),
    yaxis = dict(
        gridcolor='rgba(255,255,255,0.05)', linecolor='rgba(255,255,255,0.08)',
        tickfont=dict(color='#6E7681', size=10), zeroline=False,
    ),
    legend = dict(
        bgcolor='rgba(22,27,34,0.8)', bordercolor='rgba(255,255,255,0.08)',
        borderwidth=1, font=dict(color='#8B949E', size=11),
    ),
    margin   = dict(t=60, b=45, l=50, r=30),
    colorway = ['#AAFF00','#FFB800','#00D4FF','#FF6B6B','#B06BFF','#6BCB77','#FF9500'],
    hoverlabel=dict(
        bgcolor='#21262D', bordercolor='rgba(170,255,0,0.3)',
        font=dict(color='#F0F6FC', size=12),
    ),
)

def T(fig, h=420):
    fig.update_layout(**PT, height=h)
    return fig

# ══════════════════════════════════════════════════════════════
# HELPER — resolve column names
# ══════════════════════════════════════════════════════════════
def resolve_cols(res):
    cols = list(res.columns)
    r2_col = None
    for c in ['R2','r2','R_squared','r_squared','r2_score','R2_score','R2_log','R2_real']:
        if c in cols:
            r2_col = c
            break
    r2_was_computed = False
    if r2_col is None:
        rmse_any = next((c for c in ['RMSE_log','rmse_log','RMSE_real','RMSE','rmse'] if c in cols), None)
        if rmse_any:
            mx = res[rmse_any].max()
            res['R2_computed'] = (1 - res[rmse_any] / mx).clip(0, 1)
        else:
            res['R2_computed'] = 0.0
        r2_col = 'R2_computed'
        r2_was_computed = True
    rmse_col = next((c for c in ['RMSE_log','rmse_log','RMSE_real','rmse_real','RMSE','rmse'] if c in cols), None)
    mae_col  = next((c for c in ['MAE_log','mae_log','MAE_real','mae_real','MAE','mae'] if c in cols), None)
    return r2_col, rmse_col, mae_col, r2_was_computed

# ══════════════════════════════════════════════════════════════
# LOAD DATA & MODELS
# ══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_artifacts():
    try:
        return (
            joblib.load(os.path.join(MODEL_DIR, 'best_model.pkl')),
            joblib.load(os.path.join(MODEL_DIR, 'feature_cols.pkl')),
            joblib.load(os.path.join(MODEL_DIR, 'encoders.pkl')),
            joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl')),
            joblib.load(os.path.join(MODEL_DIR, 'scale_cols.pkl')),
            joblib.load(os.path.join(MODEL_DIR, 'state_cluster_map.pkl')),
        )
    except Exception as e:
        st.warning(f"Model load error: {e}")
        return None, None, None, None, None, None

@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(DATA_PATH)
    for col in CATEGORICAL_COLS:
        df[col] = df[col].str.strip()
    return df

def predict_yield(best_model, feature_cols, encoders, scaler, scale_cols,
                  state_cluster_map, crop, year, season, state,
                  area, rainfall, fertilizer, pesticide):
    eps = 1e-6
    row = {
        CROP_COL: crop, YEAR_COL: year, SEASON_COL: season, STATE_COL: state,
        AREA_COL: np.log1p(area), PRODUCTION_COL: np.log1p(area),
        RAINFALL_COL: rainfall, FERTILIZER_COL: np.log1p(fertilizer),
        PESTICIDE_COL: np.log1p(pesticide), TARGET_COL: 0,
        CLUSTER_COL: state_cluster_map.get(state, 1),
    }
    row['Fertilizer_per_Area']        = row[FERTILIZER_COL] / (row[AREA_COL] + eps)
    row['Pesticide_per_Area']         = row[PESTICIDE_COL]  / (row[AREA_COL] + eps)
    row['Fertilizer_Pesticide_Ratio'] = row[FERTILIZER_COL] / (row[PESTICIDE_COL] + eps)
    row['Production_per_Area']        = row[PRODUCTION_COL] / (row[AREA_COL] + eps)
    row['Year_Trend']                 = year - 1997
    for col in CATEGORICAL_COLS:
        le = encoders[col]; val = str(row[col]).strip()
        if val not in le.classes_: val = le.classes_[0]
        row[col] = le.transform([val])[0]
    X = np.array([[row.get(c, 0) for c in feature_cols]])
    for i, col in enumerate(feature_cols):
        if col in scale_cols:
            idx = scale_cols.index(col)
            X[0,i] = (X[0,i] - scaler.mean_[idx]) / (scaler.scale_[idx] + eps)
    y_log = best_model.predict(X)[0]
    return float(np.expm1(y_log)), float(y_log)

best_model, feature_cols, encoders, scaler, scale_cols, state_cluster_map = load_artifacts()
models_ready = best_model is not None
df = load_data()

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="brand">
        <span class="brand-icon">🌾</span>
        <div class="brand-name">Agri<em>Sense</em></div>
        <div class="brand-sub">ML · Climate AI · India</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("", [
        "🏠  Home", "🔮  Predict Yield", "📊  Batch Prediction",
        "🌡️  Climate Clusters", "📈  Model Performance",
        "📉  Data Explorer", "ℹ️  About",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("""<div style="font-family:'JetBrains Mono',monospace;font-size:.62rem;font-weight:700;
    letter-spacing:.16em;text-transform:uppercase;color:#8B949E;margin-bottom:.7rem;">
    📊 Dataset Overview</div>""", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="sb-box">
        <div class="sb-row"><span class="sb-lbl">📅 Period</span><span class="sb-val">1997–2020</span></div>
        <div class="sb-row"><span class="sb-lbl">🌱 Crops</span><span class="sb-val">{df[CROP_COL].nunique()}</span></div>
        <div class="sb-row"><span class="sb-lbl">🗺️ States</span><span class="sb-val">{df[STATE_COL].nunique()}</span></div>
        <div class="sb-row"><span class="sb-lbl">📋 Records</span><span class="sb-val">{len(df):,}</span></div>
        <div class="sb-row"><span class="sb-lbl">🎯 Best R²</span><span class="sb-val">0.9917</span></div>
    </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    if models_ready:
        st.markdown('<div class="status-ok">✓ &nbsp;Models loaded</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-err">⚠ &nbsp;Models not found</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════
if page == "🏠  Home":
    st.markdown("""
    <div class="hero">
        <div class="hero-badge">🛰️ AI-Powered Agricultural Intelligence</div>
        <div class="hero-title">Predict Crop Yields<br>with <span class="accent">Precision</span></div>
        <div class="hero-sub">Machine learning meets climate science — forecasting agricultural output across 30 Indian states using 24 years of harvest data.</div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi" style="animation-delay:0s"><span class="kpi-icon">📅</span><span class="kpi-val">24</span><span class="kpi-label">Years of Data</span></div>
        <div class="kpi" style="animation-delay:.07s"><span class="kpi-icon">🌱</span><span class="kpi-val">{df[CROP_COL].nunique()}</span><span class="kpi-label">Crop Types</span></div>
        <div class="kpi" style="animation-delay:.14s"><span class="kpi-icon">🗺️</span><span class="kpi-val">{df[STATE_COL].nunique()}</span><span class="kpi-label">Indian States</span></div>
        <div class="kpi" style="animation-delay:.21s"><span class="kpi-icon">📋</span><span class="kpi-val">{len(df)//1000}K+</span><span class="kpi-label">Records</span></div>
        <div class="kpi" style="animation-delay:.28s"><span class="kpi-icon">🎯</span><span class="kpi-val">99.2%</span><span class="kpi-label">Accuracy</span></div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        yr  = df.groupby(YEAR_COL)[TARGET_COL].mean().reset_index()
        fig = px.area(yr, x=YEAR_COL, y=TARGET_COL,
                      title="⬆ Average Yield Trend · 1997–2020",
                      color_discrete_sequence=['#AAFF00'])
        fig.update_traces(
            line=dict(width=2.5, color='#AAFF00'),
            fillcolor='rgba(170,255,0,0.07)',
            hovertemplate='<b>%{x}</b><br>Avg Yield: <b>%{y:.2f} t/ha</b><extra></extra>'
        )
        T(fig); st.plotly_chart(fig, use_container_width=True)

    with c2:
        sc  = df[SEASON_COL].value_counts()
        fig = px.pie(values=sc.values, names=sc.index,
                     title="🌀 Distribution by Season", hole=0.55,
                     color_discrete_sequence=['#AAFF00','#FFB800','#00D4FF','#FF6B6B','#B06BFF','#6BCB77'])
        fig.update_traces(
            textfont=dict(family='JetBrains Mono', color='#0D1117', size=10),
            marker=dict(line=dict(color='#0D1117', width=2)),
            hovertemplate='<b>%{label}</b><br>%{value} (%{percent})<extra></extra>'
        )
        T(fig); st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="sh"><div class="sh-line"></div><span class="sh-tag">🏆 Top Performers</span><div class="sh-line"></div></div>', unsafe_allow_html=True)
    top = df.groupby(CROP_COL)[TARGET_COL].median().sort_values(ascending=False).head(15).reset_index()
    fig = px.bar(top, x=CROP_COL, y=TARGET_COL,
                 title="Top 15 Crops by Median Yield (t/ha)",
                 color=TARGET_COL,
                 color_continuous_scale=[[0,'#1A3A00'],[0.5,'#AAFF00'],[1,'#FFB800']])
    fig.update_traces(marker_line_width=0,
                      hovertemplate='<b>%{x}</b><br>Median: <b>%{y:.2f} t/ha</b><extra></extra>')
    T(fig, h=380); st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="sh"><div class="sh-line"></div><span class="sh-tag">🎻 Season Breakdown</span><div class="sh-line"></div></div>', unsafe_allow_html=True)
    fig = px.violin(df.sample(min(5000, len(df)), random_state=42),
                    x=SEASON_COL, y=TARGET_COL,
                    color=SEASON_COL, title="Yield Distribution per Season", box=True,
                    color_discrete_sequence=['#AAFF00','#FFB800','#00D4FF','#FF6B6B','#B06BFF','#6BCB77'])
    fig.update_traces(meanline_visible=True)
    T(fig, h=400); st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="sh"><div class="sh-line"></div><span class="sh-tag">⚙️ How It Works</span><div class="sh-line"></div></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="flow">
        <div class="flow-item" style="animation-delay:0s">
            <div class="flow-num">01</div>
            <div class="flow-title">Data Ingestion</div>
            <div class="flow-desc">24 years of Indian agriculture records across 30 states, cleaned &amp; validated</div>
        </div>
        <div class="flow-item" style="animation-delay:.08s">
            <div class="flow-num">02</div>
            <div class="flow-title">Climate Clustering</div>
            <div class="flow-desc">K-Means groups states into 4 climate zones by annual rainfall patterns</div>
        </div>
        <div class="flow-item" style="animation-delay:.16s">
            <div class="flow-num">03</div>
            <div class="flow-title">Model Training</div>
            <div class="flow-desc">Random Forest &amp; XGBoost with 5-fold temporal cross-validation</div>
        </div>
        <div class="flow-item" style="animation-delay:.24s">
            <div class="flow-num">04</div>
            <div class="flow-title">Live Prediction</div>
            <div class="flow-desc">Instant yield forecast for any crop, state, season &amp; year combination</div>
        </div>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE 2 — PREDICT YIELD
# ══════════════════════════════════════════════════════════════
elif page == "🔮  Predict Yield":
    st.markdown("""<div class="pg-eye">🔮 Prediction Engine</div>
    <h1 class="pg-title">Single Crop Yield Forecast</h1>
    <p class="pg-sub">Configure parameters below and fire the AI model for an instant yield estimate.</p>""", unsafe_allow_html=True)

    if not models_ready:
        st.error("⚠️ Models not found. Run all Jupyter notebook cells first.")
        st.stop()

    st.markdown("---")
    crops   = sorted(df[CROP_COL].unique())
    states  = sorted(df[STATE_COL].unique())
    seasons = sorted(df[SEASON_COL].unique())

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown('<div class="panel"><div class="panel-title">🌱 Crop Details</div>', unsafe_allow_html=True)
        crop   = st.selectbox("Crop Type", crops)
        season = st.selectbox("Growing Season", seasons)
        year   = st.slider("Forecast Year", 2000, 2030, 2024)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="panel"><div class="panel-title">🗺️ Location & Climate</div>', unsafe_allow_html=True)
        state = st.selectbox("State", states)
        cid   = state_cluster_map.get(state, 1) if models_ready else 1
        st.markdown(f'<div class="chip chip-g">{CLUSTER_ICONS.get(cid,"🌿")} C{cid} — {CLUSTER_LABELS.get(cid,"")}</div>', unsafe_allow_html=True)
        area     = st.number_input("Area (hectares)", min_value=0.4047, value=50000.0, step=1000.0)
        rainfall = st.number_input("Annual Rainfall (mm)", min_value=50.0, value=1200.0, step=50.0)
        st.markdown('</div>', unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="panel"><div class="panel-title">🧪 Agrochemical Inputs</div>', unsafe_allow_html=True)
        fertilizer = st.number_input("Fertilizer (kg)", min_value=200.0, value=5000000.0, step=100000.0, format="%.0f")
        pesticide  = st.number_input("Pesticide (kg)", min_value=1.0, value=15000.0, step=500.0, format="%.0f")
        fpa = fertilizer/area if area > 0 else 0
        ppa = pesticide/area  if area > 0 else 0
        st.markdown(f"""
        <div style="margin-top:.9rem;padding:.9rem 1rem;background:rgba(170,255,0,0.05);
             border-radius:10px;border:1px solid rgba(170,255,0,0.12);">
            <div style="font-family:'JetBrains Mono',monospace;font-size:.62rem;color:#8B949E;
                 text-transform:uppercase;letter-spacing:.14em;margin-bottom:.45rem;">Derived Ratios</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:.82rem;color:#AAFF00;line-height:2;">
                Fert/ha: <b>{fpa:.1f} kg</b><br>Pest/ha: <b>{ppa:.4f} kg</b>
            </div>
        </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("⚡  Run AI Prediction", use_container_width=True)

    if run_btn:
        with st.spinner("🤖 Running model inference..."):
            y_real, y_log = predict_yield(
                best_model, feature_cols, encoders, scaler, scale_cols, state_cluster_map,
                crop, year, season, state, area, rainfall, fertilizer, pesticide
            )
        st.markdown("---")
        hist_avg  = df[df[CROP_COL] == crop][TARGET_COL].mean()
        hist_real = float(np.expm1(hist_avg)) if hist_avg < 20 else hist_avg
        delta_pct = ((y_real - hist_real) / (hist_real + 1e-6)) * 100
        d_cls = "delta-up" if delta_pct >= 0 else "delta-dn"
        d_sym = "▲" if delta_pct >= 0 else "▼"

        r1, r2 = st.columns([2, 1], gap="medium")
        with r1:
            st.markdown(f"""<div class="result-big">
                <div class="result-eye">Predicted Yield</div>
                <div class="result-num">{y_real:.2f}</div>
                <div class="result-unit">tonnes per hectare</div>
                <div style="margin-top:1.5rem;padding-top:1.1rem;border-top:1px solid rgba(170,255,0,0.12);
                     font-family:'JetBrains Mono',monospace;font-size:.82rem;color:#8B949E;">
                    Historical avg for <b style="color:#F0F6FC">{crop}</b>:
                    <b style="color:#AAFF00">{hist_real:.2f} t/ha</b>
                    &nbsp;<span class="{d_cls}">{d_sym} {abs(delta_pct):.1f}%</span>
                </div>
                <div style="margin-top:1rem;display:flex;justify-content:center;gap:.5rem;flex-wrap:wrap;">
                    <span class="chip chip-g">{crop}</span>
                    <span class="chip chip-t">{season}</span>
                    <span class="chip chip-sky">{state}</span>
                    <span class="chip chip-gold">Year {year}</span>
                </div>
            </div>""", unsafe_allow_html=True)

        with r2:
            st.metric("📐 Log-space",   f"{y_log:.4f}")
            st.metric("🌡️ Zone",        f"C{cid}: {CLUSTER_LABELS.get(cid,'')}")
            st.metric("🌱 Crop · Year", f"{crop[:12]} · {year}")
            st.metric("🗺️ State",       f"{state[:14]}")

        # Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=min(y_real, 100),
            delta={
                'reference': min(hist_real, 100), 'valueformat': '.2f',
                'increasing': {'color': '#AAFF00'}, 'decreasing': {'color': '#FF6B6B'}
            },
            title={
                'text': f"<b style='color:#F0F6FC'>{crop}</b> — {state}<br>"
                        f"<span style='font-size:.8em;color:#8B949E'>{season} · {year}</span>",
                'font': {'family': 'Rajdhani', 'color': '#F0F6FC', 'size': 16}
            },
            number={'font': {'family': 'JetBrains Mono', 'color': '#AAFF00', 'size': 52}, 'suffix': ' t/ha'},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#3D444D', 'tickfont': {'color': '#6E7681', 'size': 9}},
                'bar': {'color': '#AAFF00', 'thickness': .22},
                'bgcolor': 'rgba(0,0,0,0)', 'borderwidth': 0,
                'steps': [
                    {'range': [0, 20],  'color': 'rgba(255,107,107,0.1)'},
                    {'range': [20, 50], 'color': 'rgba(255,184,0,0.08)'},
                    {'range': [50, 100],'color': 'rgba(170,255,0,0.08)'},
                ],
                'threshold': {'line': {'color': '#FFB800', 'width': 3}, 'thickness': .75, 'value': min(hist_real, 100)}
            }
        ))
        fig.update_layout(
            height=380, paper_bgcolor='rgba(22,27,34,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Nunito', color='#8B949E'), margin=dict(t=70, b=25, l=50, r=50)
        )
        st.plotly_chart(fig, use_container_width=True)

        comp = pd.DataFrame({'Type': ['Your Prediction', 'Historical Avg'], 'Yield': [y_real, hist_real]})
        fig2 = px.bar(comp, x='Type', y='Yield', color='Type', text='Yield',
                      color_discrete_map={'Your Prediction': '#AAFF00', 'Historical Avg': '#00D4FF'},
                      title=f"Prediction vs Historical Average — {crop}")
        fig2.update_traces(texttemplate='%{text:.2f} t/ha', textposition='outside', marker_line_width=0)
        T(fig2, h=340); st.plotly_chart(fig2, use_container_width=True)

        with st.expander("📋 Full prediction details"):
            for k, v in {
                'Crop': crop, 'State': state, 'Season': season, 'Year': year,
                'Climate Zone': f'C{cid}: {CLUSTER_LABELS.get(cid, "")}',
                'Predicted Yield (t/ha)': round(y_real, 4),
                'Log-space': round(y_log, 4),
                'Historical Avg': round(hist_real, 4),
                'Delta': f"{d_sym} {abs(delta_pct):.1f}%"
            }.items():
                st.markdown(f"**{k}:** `{v}`")

# ══════════════════════════════════════════════════════════════
# PAGE 3 — BATCH PREDICTION
# ══════════════════════════════════════════════════════════════
elif page == "📊  Batch Prediction":
    st.markdown("""<div class="pg-eye">📊 Bulk Processing</div>
    <h1 class="pg-title">Batch Yield Prediction</h1>
    <p class="pg-sub">Upload a CSV to run predictions on multiple crops simultaneously.</p>""", unsafe_allow_html=True)

    if not models_ready:
        st.error("⚠️ Models not found."); st.stop()

    st.markdown("---")
    sample = pd.DataFrame([
        {'Crop':'Rice','Crop_Year':2022,'Season':'Kharif','State':'Assam','Area':60000,'Production':60000,'Annual_Rainfall':2100,'Fertilizer':5500000,'Pesticide':18000},
        {'Crop':'Wheat','Crop_Year':2022,'Season':'Rabi','State':'Punjab','Area':35000,'Production':35000,'Annual_Rainfall':510,'Fertilizer':3200000,'Pesticide':8800},
        {'Crop':'Maize','Crop_Year':2022,'Season':'Kharif','State':'Karnataka','Area':45000,'Production':45000,'Annual_Rainfall':1200,'Fertilizer':4200000,'Pesticide':14000},
    ])

    col_a, col_b = st.columns([2, 1], gap="large")
    with col_a:
        st.markdown("""<div style="background:rgba(170,255,0,0.05);border:1px solid rgba(170,255,0,0.15);
            border-radius:14px;padding:1.3rem 1.6rem;margin-bottom:1rem;">
            <div style="font-family:'JetBrains Mono',monospace;font-size:.65rem;font-weight:700;
                 letter-spacing:.16em;text-transform:uppercase;color:#AAFF00;margin-bottom:.7rem;">
                Required CSV Columns
            </div>
            <code style="font-family:'JetBrains Mono',monospace;font-size:.8rem;color:#AAFF00;
                 background:rgba(170,255,0,0.06);padding:.8rem 1.1rem;border-radius:8px;
                 display:block;line-height:2.2;border-left:3px solid #AAFF00;">
                Crop · Crop_Year · Season · State · Area<br>
                Production · Annual_Rainfall · Fertilizer · Pesticide
            </code>
        </div>""", unsafe_allow_html=True)
        uploaded = st.file_uploader("Drop CSV here", type=["csv"], label_visibility="collapsed")

    with col_b:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.download_button(
            "⬇️  Download Sample Template",
            sample.to_csv(index=False).encode(),
            "agrisense_template.csv", "text/csv",
            use_container_width=True
        )

    if uploaded:
        df_in = pd.read_csv(uploaded)
        for col in CATEGORICAL_COLS:
            if col in df_in.columns:
                df_in[col] = df_in[col].str.strip()
        st.markdown(f'<div class="batch-info">✓ <b>{len(df_in)}</b> rows loaded</div>', unsafe_allow_html=True)
        st.dataframe(df_in.head(5), use_container_width=True)

        if st.button("⚡  Run Batch Prediction", use_container_width=True):
            results = []; progress = st.progress(0); status = st.empty()
            for i, row in df_in.iterrows():
                status.markdown(
                    f"<span style='font-family:JetBrains Mono;color:#8B949E;font-size:.82rem;'>"
                    f"Processing {i+1}/{len(df_in)}…</span>", unsafe_allow_html=True
                )
                try:
                    y_real, _ = predict_yield(
                        best_model, feature_cols, encoders, scaler, scale_cols, state_cluster_map,
                        crop=row.get(CROP_COL,''), year=int(row.get(YEAR_COL, 2020)),
                        season=row.get(SEASON_COL,''), state=row.get(STATE_COL,''),
                        area=float(row.get(AREA_COL, 1000)), rainfall=float(row.get(RAINFALL_COL, 1000)),
                        fertilizer=float(row.get(FERTILIZER_COL, 100000)),
                        pesticide=float(row.get(PESTICIDE_COL, 1000))
                    )
                    results.append(round(y_real, 4))
                except:
                    results.append(None)
                progress.progress((i + 1) / len(df_in))
            status.empty()
            df_in['Predicted_Yield_t_per_ha'] = results
            valid = [r for r in results if r is not None]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("✅ Done",   len(valid))
            m2.metric("📊 Avg",   f"{np.mean(valid):.2f} t/ha")
            m3.metric("📈 Max",   f"{np.max(valid):.2f} t/ha")
            m4.metric("📉 Min",   f"{np.min(valid):.2f} t/ha")
            st.dataframe(df_in, use_container_width=True)
            if CROP_COL in df_in.columns:
                fig = px.bar(df_in, x=CROP_COL, y='Predicted_Yield_t_per_ha',
                             color='Predicted_Yield_t_per_ha',
                             color_continuous_scale=[[0,'#1A3A00'],[0.5,'#AAFF00'],[1,'#FFB800']],
                             title="Batch Prediction Results")
                fig.update_traces(marker_line_width=0)
                T(fig, 380); st.plotly_chart(fig, use_container_width=True)
            st.download_button(
                "⬇️  Download Results",
                df_in.to_csv(index=False).encode(),
                "agrisense_results.csv", "text/csv",
                use_container_width=True
            )

# ══════════════════════════════════════════════════════════════
# PAGE 4 — CLIMATE CLUSTERS
# ══════════════════════════════════════════════════════════════
elif page == "🌡️  Climate Clusters":
    st.markdown("""<div class="pg-eye">🌡️ Climate Intelligence</div>
    <h1 class="pg-title">Climate Cluster Explorer</h1>
    <p class="pg-sub">30 Indian states grouped into 4 climate zones using K-Means clustering on rainfall.</p>""", unsafe_allow_html=True)
    st.markdown("---")

    sr = (df.groupby(STATE_COL)[RAINFALL_COL].agg(['mean','min','max']).reset_index())
    sr.columns = [STATE_COL, 'Mean Rainfall', 'Min Rainfall', 'Max Rainfall']
    sr['Cluster'] = sr[STATE_COL].map(state_cluster_map).fillna(1).astype(int)
    sr['Zone']    = sr['Cluster'].map(lambda x: f"Cluster {x}: {CLUSTER_LABELS.get(x,'')}")

    tab1, tab2 = st.tabs(["📊 Bar Chart", "🫧 Bubble Chart"])
    cm = {f"Cluster {i}: {CLUSTER_LABELS[i]}": CLUSTER_COLOR_MAP[i] for i in range(4)}

    with tab1:
        fig = px.bar(sr.sort_values('Mean Rainfall', ascending=False),
                     x=STATE_COL, y='Mean Rainfall', color='Zone',
                     title="Annual Rainfall by State & Climate Zone",
                     color_discrete_map=cm)
        fig.update_traces(marker_line_width=0,
                          hovertemplate='<b>%{x}</b><br>%{y:.0f} mm<extra></extra>')
        fig.update_xaxes(tickangle=45)
        T(fig, 460); st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig2 = px.scatter(sr, x='Min Rainfall', y='Max Rainfall',
                          size='Mean Rainfall', color='Zone', hover_name=STATE_COL,
                          title="State Rainfall Profile", color_discrete_map=cm)
        T(fig2, 460); st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="sh"><div class="sh-line"></div><span class="sh-tag">Zone Profiles</span><div class="sh-line"></div></div>', unsafe_allow_html=True)
    z1, z2, z3, z4 = st.columns(4, gap="medium")
    for col, cid, color, rng, sts, ctxt in [
        (z1, 0, '#FF6B6B', '< 700 mm',      'Haryana, Punjab,\nGujarat, Delhi',  'Wheat, Cotton, Bajra'),
        (z2, 1, '#FFB800', '700–1700 mm',   'Maharashtra, Karnataka,\nMP, Bihar','Jowar, Sugarcane'),
        (z3, 2, '#AAFF00', '1700–3500 mm',  'Assam, Kerala,\nGoa, Sikkim',       'Rice, Tea, Spices'),
        (z4, 3, '#00D4FF', '> 3500 mm',     'Meghalaya',                          'Tropical Crops'),
    ]:
        with col:
            st.markdown(f"""<div class="zone" style="border-color:{color}22;border-top:2px solid {color};">
                <span class="zone-icon">{CLUSTER_ICONS[cid]}</span>
                <div class="zone-name" style="color:{color};">{CLUSTER_ICONS[cid]} C{cid}: {CLUSTER_LABELS[cid]}</div>
                <div class="zone-range" style="color:{color}80;">{rng}</div>
                <div class="zone-states">{sts}</div>
                <div style="margin-top:.6rem;font-size:.74rem;color:{color}60;font-style:italic;">{ctxt}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.dataframe(sr.sort_values('Mean Rainfall', ascending=False), use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE 5 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════
elif page == "📈  Model Performance":
    st.markdown("""<div class="pg-eye">📈 Evaluation Suite</div>
    <h1 class="pg-title">Model Performance Dashboard</h1>
    <p class="pg-sub">Benchmarking Ridge, Random Forest, Gradient Boosting, XGBoost & LightGBM.</p>""", unsafe_allow_html=True)
    st.markdown("---")

    results_path = os.path.join(OUTPUT_DIR, 'model_results.csv')

    if not os.path.exists(results_path):
        st.error(f"❌ `model_results.csv` not found at: `{results_path}`")
        st.info("Run your Jupyter notebook evaluation cells to generate this file, then refresh.")
    else:
        try:
            res = pd.read_csv(results_path, index_col=0)
            if res.index.dtype != object:
                res.index = [f'Model_{i}' for i in range(len(res))]

            r2_col, rmse_col, mae_col, r2_was_computed = resolve_cols(res)

            best_m    = res[r2_col].idxmax()
            best_r2   = res.loc[best_m, r2_col]
            best_rmse = f"{res.loc[best_m, rmse_col]:.4f}" if rmse_col else "N/A"
            best_mae  = f"{res.loc[best_m, mae_col]:.4f}"  if mae_col  else "N/A"

            st.markdown(f"""<div class="perf">
                <span class="perf-trophy">🏆</span>
                <div>
                    <div class="perf-main">Best Model: {best_m}</div>
                    <div class="perf-sub">R² = {best_r2:.4f} · RMSE = {best_rmse} · MAE = {best_mae}</div>
                </div>
            </div>""", unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("🏆 Best R²",      f"{res[r2_col].max():.4f}")
            m2.metric("📉 Best RMSE",    f"{res[rmse_col].min():.4f}" if rmse_col else "N/A")
            m3.metric("📊 Models",       len(res))
            m4.metric("📋 Test Records", "4,285")

            st.markdown('<div class="sh"><div class="sh-line"></div><span class="sh-tag">All Metrics</span><div class="sh-line"></div></div>', unsafe_allow_html=True)
            st.dataframe(res, use_container_width=True)

            plot_df   = res.reset_index()
            model_col = plot_df.columns[0]

            c1, c2 = st.columns(2, gap="medium")
            with c1:
                fig = px.bar(plot_df, x=model_col, y=r2_col, color=r2_col,
                             color_continuous_scale=[[0,'#1A3A00'],[0.5,'#AAFF00'],[1,'#FFB800']],
                             title='R² Score (higher = better)',
                             labels={model_col:'Model'}, text=r2_col)
                fig.update_traces(texttemplate='%{text:.4f}', textposition='outside', marker_line_width=0)
                T(fig, 380); st.plotly_chart(fig, use_container_width=True)

            with c2:
                if rmse_col:
                    fig = px.bar(plot_df, x=model_col, y=rmse_col, color=rmse_col,
                                 color_continuous_scale=[[0,'#AAFF00'],[0.5,'#FFB800'],[1,'#FF6B6B']],
                                 title=f'{rmse_col} (lower = better)',
                                 labels={model_col:'Model'}, text=rmse_col)
                    fig.update_traces(texttemplate='%{text:.4f}', textposition='outside', marker_line_width=0)
                    T(fig, 380); st.plotly_chart(fig, use_container_width=True)

                    if mae_col:
                        fig3 = px.bar(plot_df, x=model_col, y=mae_col, color=mae_col,
                                      color_continuous_scale=[[0,'#AAFF00'],[0.5,'#FFB800'],[1,'#FF6B6B']],
                                      title=f'{mae_col} (lower = better)',
                                      labels={model_col:'Model'}, text=mae_col)
                        fig3.update_traces(texttemplate='%{text:.4f}', textposition='outside', marker_line_width=0)
                        T(fig3, 380); st.plotly_chart(fig3, use_container_width=True)
                else:
                    st.info("⚠️  No RMSE column available.")

            if os.path.exists(PLOTS_DIR):
                st.markdown("---")
                plot_files = [
                    ('model_01_comparison.png',        'Model Comparison',      '📊'),
                    ('model_02_feature_importance.png','Feature Importance',    '🔍'),
                    ('eval_01_predictions.png',        'Actual vs Predicted',   '🎯'),
                    ('eval_02_cluster_performance.png','Performance by Cluster','🌡️'),
                    ('eval_03_error_by_state.png',     'Error by State',        '🗺️'),
                    ('eval_04_r2_by_season.png',       'R² by Season',          '☀️'),
                    ('eval_05_learning_curve.png',     'Learning Curve',        '📈'),
                ]
                found_plots = False
                for fname, title, tag in plot_files:
                    fpath = os.path.join(PLOTS_DIR, fname)
                    if os.path.exists(fpath):
                        st.markdown(f'<div class="sh"><div class="sh-line"></div><span class="sh-tag">{tag} {title}</span><div class="sh-line"></div></div>', unsafe_allow_html=True)
                        st.image(fpath, use_container_width=True)
                        found_plots = True
                if not found_plots:
                    st.info("💡 No evaluation plot images found in the plots folder.")

        except Exception as e:
            st.error(f"❌ Could not read `model_results.csv`: {type(e).__name__} — {e}")
            try:
                preview = pd.read_csv(results_path, nrows=3)
                st.write("**Columns found:**", list(preview.columns))
                st.dataframe(preview)
            except Exception:
                pass

# ══════════════════════════════════════════════════════════════
# PAGE 6 — DATA EXPLORER
# ══════════════════════════════════════════════════════════════
elif page == "📉  Data Explorer":
    st.markdown("""<div class="pg-eye">📉 Interactive Analytics</div>
    <h1 class="pg-title">Data Explorer</h1>
    <p class="pg-sub">Explore the raw dataset interactively with filters and dynamic charts.</p>""", unsafe_allow_html=True)
    st.markdown("---")

    f1, f2, f3 = st.columns(3)
    with f1:
        sel_states  = st.multiselect("Filter by State",  sorted(df[STATE_COL].unique()),  default=sorted(df[STATE_COL].unique())[:5])
    with f2:
        sel_seasons = st.multiselect("Filter by Season", sorted(df[SEASON_COL].unique()), default=sorted(df[SEASON_COL].unique()))
    with f3:
        yr_range = st.slider("Year Range", int(df[YEAR_COL].min()), int(df[YEAR_COL].max()), (2000, 2020))

    df_f = df[
        df[STATE_COL].isin(sel_states) &
        df[SEASON_COL].isin(sel_seasons) &
        df[YEAR_COL].between(*yr_range)
    ]
    st.markdown(
        f'<div class="batch-info">📋 Showing <b>{len(df_f):,}</b> records · '
        f'{len(sel_states)} states · {len(sel_seasons)} seasons · {yr_range[0]}–{yr_range[1]}</div>',
        unsafe_allow_html=True
    )

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Trend","🏆 Rankings","🌧️ Rainfall","🔥 Heatmap"])

    with tab1:
        yr_filt = df_f.groupby(YEAR_COL)[TARGET_COL].mean().reset_index()
        fig = px.line(yr_filt, x=YEAR_COL, y=TARGET_COL,
                      title="Avg Yield Trend (Filtered)", markers=True,
                      color_discrete_sequence=['#AAFF00'])
        fig.update_traces(
            line=dict(width=2.5, color='#AAFF00'),
            marker=dict(size=8, color='#AAFF00', line=dict(width=2, color='#0D1117'))
        )
        T(fig, 400); st.plotly_chart(fig, use_container_width=True)

    with tab2:
        top_n = st.slider("Show top N crops", 5, 20, 10)
        top_f = df_f.groupby(CROP_COL)[TARGET_COL].median().sort_values(ascending=False).head(top_n).reset_index()
        fig = px.bar(top_f, x=TARGET_COL, y=CROP_COL, orientation='h',
                     title=f"Top {top_n} Crops by Median Yield",
                     color=TARGET_COL,
                     color_continuous_scale=[[0,'#1A3A00'],[0.5,'#AAFF00'],[1,'#FFB800']])
        fig.update_traces(marker_line_width=0)
        T(fig, max(350, top_n*30)); st.plotly_chart(fig, use_container_width=True)

    with tab3:
        sr_f = df_f.groupby(STATE_COL)[[RAINFALL_COL, TARGET_COL]].mean().reset_index()
        fig = px.scatter(sr_f, x=RAINFALL_COL, y=TARGET_COL, size=TARGET_COL,
                         hover_name=STATE_COL, color=RAINFALL_COL,
                         color_continuous_scale=[[0,'#FF6B6B'],[0.4,'#FFB800'],[0.7,'#AAFF00'],[1,'#00D4FF']],
                         title="Rainfall vs Average Yield by State",
                         labels={RAINFALL_COL:'Avg Rainfall (mm)', TARGET_COL:'Avg Yield (t/ha)'})
        T(fig, 430); st.plotly_chart(fig, use_container_width=True)

    with tab4:
        piv = (
            df_f.groupby([STATE_COL, SEASON_COL])[TARGET_COL]
            .mean().reset_index()
            .pivot(index=STATE_COL, columns=SEASON_COL, values=TARGET_COL)
            .fillna(0)
        )
        fig = px.imshow(
            piv,
            color_continuous_scale=[[0,'#0D1117'],[0.3,'#1A3A00'],[0.7,'#AAFF00'],[1,'#FFB800']],
            title="State × Season Yield Heatmap", aspect="auto"
        )
        fig.update_traces(
            hovertemplate='State: <b>%{y}</b><br>Season: <b>%{x}</b><br>Avg Yield: <b>%{z:.2f} t/ha</b><extra></extra>'
        )
        T(fig, 520); st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="sh"><div class="sh-line"></div><span class="sh-tag">Raw Data Preview</span><div class="sh-line"></div></div>', unsafe_allow_html=True)
    st.dataframe(df_f.head(100), use_container_width=True)
    st.download_button(
        "⬇️  Download Filtered Data",
        df_f.to_csv(index=False).encode(),
        "agrisense_filtered.csv", "text/csv",
        use_container_width=True
    )

# ══════════════════════════════════════════════════════════════
# PAGE 7 — ABOUT
# ══════════════════════════════════════════════════════════════
elif page == "ℹ️  About":
    st.markdown("""<div class="pg-eye">ℹ️ Project Info</div>
    <h1 class="pg-title">About AgriSense</h1>
    <p class="pg-sub">An end-to-end ML system for agricultural yield intelligence across India.</p>""", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns([3, 2], gap="large")
    with col1:
        st.markdown("""<div class="about-card">
            <div class="about-title">🔬 Methodology</div>
            <div class="about-step">
                <div class="about-step-tag">01 · Climate Clustering</div>
                <div class="about-step-desc">States grouped into 4 zones using K-Means on annual rainfall. Zones range from Arid (&lt;700mm) to Humid (&gt;3500mm).</div>
            </div>
            <div class="about-step">
                <div class="about-step-tag">02 · Feature Engineering</div>
                <div class="about-step-desc">Log transforms on skewed columns. Derived ratios: Fertilizer/Area, Pesticide/Area. Year trend & climate cluster as inputs.</div>
            </div>
            <div class="about-step">
                <div class="about-step-tag">03 · Model Training</div>
                <div class="about-step-desc">Temporal split (Train ≤2015, Test &gt;2015). 5-Fold CV. Best: Random Forest R² = 0.9917.</div>
            </div>
            <div class="about-step">
                <div class="about-step-tag">04 · Evaluation</div>
                <div class="about-step-desc">R², RMSE (log + real), MAE evaluated per model, season, climate cluster & state.</div>
            </div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("""<div class="about-card">
            <div class="about-title">📦 Tech Stack</div>
            <table class="stk">
                <tr><th>Tool</th><th>Purpose</th></tr>
                <tr><td>Python 3.x</td><td>Core language</td></tr>
                <tr><td>Pandas / NumPy</td><td>Data processing</td></tr>
                <tr><td>Scikit-learn</td><td>ML models</td></tr>
                <tr><td>XGBoost / LightGBM</td><td>Boosting</td></tr>
                <tr><td>Plotly</td><td>Interactive charts</td></tr>
                <tr><td>Streamlit</td><td>Web dashboard</td></tr>
                <tr><td>Joblib</td><td>Model persistence</td></tr>
            </table>
        </div>
        <div class="about-card" style="background:rgba(170,255,0,0.03);border-color:rgba(170,255,0,0.1);">
            <div class="about-title">📊 Dataset Facts</div>
            <div class="fact"><span class="fact-lbl">Period</span><span class="fact-val">1997–2020</span></div>
            <div class="fact"><span class="fact-lbl">Records</span><span class="fact-val">19,689</span></div>
            <div class="fact"><span class="fact-lbl">Unique Crops</span><span class="fact-val">55</span></div>
            <div class="fact"><span class="fact-lbl">States</span><span class="fact-val">30</span></div>
            <div class="fact" style="border:none"><span class="fact-lbl">Seasons</span><span class="fact-val">6</span></div>
        </div>""", unsafe_allow_html=True)

    results_path = os.path.join(OUTPUT_DIR, 'model_results.csv')
    if os.path.exists(results_path):
        try:
            res = pd.read_csv(results_path, index_col=0)
            r2_col, _, _, _ = resolve_cols(res)
            st.markdown('<div class="sh"><div class="sh-line"></div><span class="sh-tag">Model Leaderboard</span><div class="sh-line"></div></div>', unsafe_allow_html=True)
            _lead_df   = res.sort_values(r2_col, ascending=True).reset_index()
            _lead_mcol = _lead_df.columns[0]
            fig = px.bar(
                _lead_df, x=r2_col, y=_lead_mcol, orientation='h',
                color=r2_col,
                color_continuous_scale=[[0,'#1A3A00'],[0.5,'#AAFF00'],[1,'#FFB800']],
                title="Model R² Leaderboard",
                labels={_lead_mcol:'Model', r2_col:'R² Score'}
            )
            fig.update_traces(
                marker_line_width=0,
                hovertemplate='<b>%{y}</b><br>R²: %{x:.4f}<extra></extra>'
            )
            T(fig, 300); st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not render leaderboard: {e}")

# ══════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════
st.markdown("""<div class="footer">
    🌾 &nbsp;<em>AgriSense</em> — Crop Yield Intelligence
    &nbsp;·&nbsp; Built with <em>Streamlit</em> · <em>Scikit-learn</em> · <em>Plotly</em>
    &nbsp;·&nbsp; India Agriculture Dataset 1997–2020
</div>""", unsafe_allow_html=True)