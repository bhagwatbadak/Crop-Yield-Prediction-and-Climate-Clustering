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
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', filename)
    if not os.path.exists(path):
        return ''
    with open(path, 'rb') as fh:
        return _b64.b64encode(fh.read()).decode()

def _bg_css(b64, mime='image/avif'):
    if not b64:
        return ''
    return f"url('data:{mime};base64,{b64}')"

IMG_HOME   = _img_b64('homeweb.avif')
IMG_MAIZE  = _img_b64('maize.avif')
IMG_CARROT = _img_b64('carrot.avif')
IMG_CROP   = _img_b64('crop.avif')


# ══════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════
# PATH SETUP
# ══════════════════════════════════════════════════════════════
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))   # → .../app/
ROOT_DIR   = os.path.dirname(BASE_DIR)                    # → one level up
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
CLUSTER_COLOR_MAP = {0:'#EF476F', 1:'#FFB347', 2:'#40916C', 3:'#06A77D'}
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
# CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Space+Mono:wght@400;700&family=Outfit:wght@300;400;500;600;700&display=swap');

:root {
    --bg-light:     #F0F9F6;
    --bg-darker:    #E8F5F1;
    --white:        #FFFFFF;
    --forest:       #0F3D2C;
    --forest-light: #1B5E48;
    --forest-dark:  #0A2818;
    --emerald:      #2ECC71;
    --emerald-bright: #52B788;
    --teal:         #16A085;
    --lime:         #7FD8BE;
    --gold:         #F39C12;
    --coral:        #E74C3C;
    --sky:          #00BFA5;
    --violet:       #6C63FF;
    --text-dark:    #0F3D2C;
    --text-medium:  #2C5E3D;
    --text-light:   #5A7A6A;
    --border:       rgba(46, 204, 113, 0.2);
    --border-strong: rgba(46, 204, 113, 0.35);
    --shadow-sm:    0 2px 12px rgba(15, 61, 44, 0.1);
    --shadow-md:    0 8px 28px rgba(15, 61, 44, 0.15);
    --shadow-lg:    0 20px 50px rgba(15, 61, 44, 0.2);
    --glow-sm:      0 0 24px rgba(46, 204, 113, 0.18);
    --glow-md:      0 0 50px rgba(46, 204, 113, 0.28);
    --r:            20px;
    --r-sm:         12px;
    --ease:         cubic-bezier(0.34, 1.56, 0.64, 1);
}

*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}

html,body,[data-testid="stAppViewContainer"],[data-testid="stMain"]{
    background: linear-gradient(135deg, #F0F9F6 0%, #E8F5F1 50%, #F0F5FA 100%)!important;
    font-family:'Outfit',sans-serif!important;
    color:var(--text-dark)!important;
}

[data-testid="stAppViewContainer"]::before{
    content:'';position:fixed;inset:0;
    background:
        radial-gradient(circle at 15% 40%, rgba(82, 183, 136, 0.12) 0%, transparent 50%),
        radial-gradient(circle at 85% 60%, rgba(29, 120, 116, 0.1) 0%, transparent 50%),
        radial-gradient(circle at 50% 100%, rgba(6, 167, 125, 0.08) 0%, transparent 40%),
        linear-gradient(135deg, rgba(64, 145, 108, 0.02) 0%, transparent 50%);
    pointer-events:none;z-index:0;
}

[data-testid="stAppViewContainer"]::after{
    content:'';position:fixed;
    width:600px;height:600px;border-radius:50%;
    background:radial-gradient(circle, rgba(82, 183, 136, 0.05) 0%, transparent 70%);
    top:-200px;right:-200px;
    animation:orbFloat 20s ease-in-out infinite;
    pointer-events:none;z-index:0;
}

@keyframes orbFloat{
    0%,100%{transform:translate(0,0) scale(1);}
    33%{transform:translate(-80px,100px) scale(1.1);}
    66%{transform:translate(60px,-70px) scale(0.95);}
}

.main .block-container{padding:2rem 3rem 5rem!important;max-width:1600px!important;position:relative;z-index:1;}

[data-testid="stSidebar"]{
    background: linear-gradient(180deg, #FFFFFF 0%, #F0F9F6 50%, #E8F5F1 100%)!important;
    border-right: 2px solid var(--border)!important;
    box-shadow: var(--shadow-lg)!important;
}

[data-testid="stSidebar"] *{font-family:'Outfit',sans-serif!important;}
[data-testid="stSidebar"] .stRadio>div{gap:2px!important;}

[data-testid="stSidebar"] .stRadio label{
    display:flex!important;align-items:center!important;
    padding:13px 16px!important;border-radius:12px!important;
    margin:4px 0!important;color:var(--text-medium)!important;
    font-size:.9rem!important;font-weight:600!important;
    transition:all .3s ease!important;cursor:pointer!important;
    border:1px solid transparent!important;
    background:transparent!important;
    letter-spacing:.01em!important;position:relative!important;
}

[data-testid="stSidebar"] .stRadio label:hover{
    background: rgba(46, 204, 113, 0.12)!important;
    color:var(--forest)!important;
    border-color:var(--border-strong)!important;
    transform:translateX(8px)!important;
    box-shadow: inset 4px 0 0 var(--emerald)!important;
}

[data-testid="stSidebar"] [aria-selected="true"] label{
    background: linear-gradient(135deg, rgba(46, 204, 113, 0.15) 0%, rgba(82, 183, 136, 0.1) 100%)!important;
    color:var(--forest)!important;
    border-color:var(--emerald)!important;
    box-shadow: var(--glow-sm), inset 4px 0 0 var(--emerald)!important;
    font-weight:700!important;
}

h1,h2,h3,h4{font-family:'Syne',sans-serif!important;color:var(--text-dark)!important;letter-spacing:-.02em;}
hr{border:none!important;border-top:1px solid var(--border)!important;margin:2rem 0!important;}
code{font-family:'Space Mono',monospace!important;color:var(--forest)!important;background:rgba(64, 145, 108, 0.08)!important;padding:.2em .4em!important;border-radius:4px!important;}

[data-testid="stMetric"]{
    background:rgba(255, 255, 255, 0.7)!important;
    backdrop-filter: blur(10px)!important;
    border:1px solid rgba(64, 145, 108, 0.2)!important;
    border-radius:var(--r)!important;padding:1.4rem 1.6rem!important;
    box-shadow:var(--shadow-sm), inset 0 1px 0 rgba(255,255,255,0.5)!important;
    transition:all .4s var(--ease)!important;
    position:relative!important;overflow:hidden!important;
    animation:glassSlideUp .6s var(--ease) both;
}

[data-testid="stMetric"]::before{
    content:'';position:absolute;top:0;left:0;right:0;height:1px;
    background:linear-gradient(90deg,transparent 0%,rgba(64,145,108,0.4) 50%,transparent 100%);
    opacity:.8;
}

[data-testid="stMetric"]:hover{
    transform:translateY(-8px)!important;
    border-color:var(--border-strong)!important;
    box-shadow:var(--shadow-lg),inset 0 1px 0 rgba(255,255,255,0.8),var(--glow-sm)!important;
    background:rgba(255, 255, 255, 0.85)!important;
}

@keyframes glassSlideUp{from{opacity:0;transform:translateY(20px);}to{opacity:1;transform:translateY(0);}}

[data-testid="stMetricLabel"]{
    font-size:.7rem!important;font-weight:700!important;letter-spacing:.15em!important;
    text-transform:uppercase!important;color:var(--text-light)!important;
    font-family:'Space Mono',monospace!important;
}

[data-testid="stMetricValue"]{
    font-family:'Space Mono',monospace!important;font-size:1.8rem!important;
    font-weight:800!important;color:var(--forest)!important;line-height:1.2!important;
    text-shadow:0 2px 4px rgba(27,67,50,0.1)!important;
}

.stButton>button{
    background: linear-gradient(135deg, #0F3D2C 0%, #1B5E48 35%, #2ECC71 70%, #00BFA5 100%)!important;
    color:white!important;font-family:'Syne',sans-serif!important;font-size:1rem!important;
    font-weight:800!important;letter-spacing:.08em!important;text-transform:uppercase!important;
    border:none!important;border-radius:var(--r-sm)!important;padding:1rem 2.2rem!important;
    width:100%!important;
    box-shadow:0 10px 32px rgba(46, 204, 113, 0.32), inset 0 1px 0 rgba(255,255,255,0.25)!important;
    transition:all .4s var(--ease)!important;position:relative!important;overflow:hidden!important;
}

.stButton>button::before{
    content:'';position:absolute;top:0;left:-100%;width:100%;height:100%;
    background:linear-gradient(90deg,transparent,rgba(255,255,255,0.3),transparent);
    transition:left .6s ease!important;
}

.stButton>button:hover::before{left:100%!important;}
.stButton>button:hover{transform:translateY(-4px)!important;box-shadow:0 16px 48px rgba(46, 204, 113, 0.42), inset 0 1px 0 rgba(255,255,255,0.35)!important;}
.stButton>button:active{transform:translateY(-1px)!important;}

.stSelectbox>div>div,.stNumberInput>div>div>input{
    background:rgba(255, 255, 255, 0.7)!important;backdrop-filter: blur(10px)!important;
    border:1.5px solid var(--border)!important;border-radius:var(--r-sm)!important;
    color:var(--text-dark)!important;transition:all .3s ease!important;font-family:'Outfit',sans-serif!important;
}

.stSelectbox>div>div:focus-within,.stNumberInput>div>div>input:focus{
    border-color:var(--emerald)!important;
    box-shadow:0 0 0 3px rgba(64, 145, 108, 0.1), inset 0 0 0 1px var(--emerald)!important;
    background:rgba(255, 255, 255, 0.95)!important;
}

[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"]{background:var(--emerald)!important;box-shadow:0 0 16px rgba(64, 145, 108, 0.5)!important;width:20px!important;height:20px!important;}
[data-testid="stSlider"] [data-baseweb="slider"] div:nth-child(2){background:linear-gradient(90deg,var(--forest-light),var(--emerald))!important;height:4px!important;}

[data-testid="stDataFrame"]{border:1px solid var(--border)!important;border-radius:var(--r)!important;overflow:hidden!important;background:rgba(255, 255, 255, 0.7)!important;box-shadow:var(--shadow-sm)!important;}

.stAlert{border-radius:12px!important;border-left-width:4px!important;font-size:.9rem!important;backdrop-filter:blur(10px)!important;}
[data-testid="stInfo"]{background:rgba(6, 167, 125, 0.08)!important;border-color:var(--sky)!important;}
[data-testid="stSuccess"]{background:rgba(64, 145, 108, 0.08)!important;border-color:var(--emerald)!important;}
[data-testid="stWarning"]{background:rgba(255, 180, 71, 0.08)!important;border-color:#FFB347!important;}
[data-testid="stError"]{background:rgba(239, 71, 111, 0.08)!important;border-color:var(--coral)!important;}

[data-testid="stProgressBar"]>div>div{background:linear-gradient(90deg, var(--forest-light), var(--emerald), var(--sky))!important;border-radius:99px!important;box-shadow:0 0 16px rgba(64, 145, 108, 0.4)!important;}

[data-testid="stPlotlyChart"]{border-radius:var(--r)!important;overflow:hidden!important;border:1px solid var(--border)!important;box-shadow:var(--shadow-sm)!important;transition:all .4s var(--ease)!important;backdrop-filter:blur(10px)!important;background:rgba(255, 255, 255, 0.5)!important;}
[data-testid="stPlotlyChart"]:hover{border-color:var(--border-strong)!important;box-shadow:var(--shadow-md), var(--glow-sm)!important;transform:translateY(-6px)!important;}

.stTabs [data-baseweb="tab-list"]{background:rgba(255, 255, 255, 0.7)!important;border-radius:14px!important;padding:6px!important;gap:4px!important;border:1px solid var(--border)!important;backdrop-filter:blur(10px)!important;}
.stTabs [data-baseweb="tab"]{background:transparent!important;border-radius:10px!important;color:var(--text-medium)!important;font-weight:600!important;font-size:.9rem!important;transition:all .3s ease!important;font-family:'Outfit',sans-serif!important;}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg, var(--forest-light) 0%, var(--emerald) 100%)!important;color:white!important;box-shadow:0 4px 16px rgba(64, 145, 108, 0.3)!important;}

[data-testid="stFileUploader"]{background:rgba(255, 255, 255, 0.7)!important;border:2px dashed var(--border-strong)!important;border-radius:var(--r)!important;transition:all .3s ease!important;}
[data-testid="stFileUploader"]:hover{border-color:var(--emerald)!important;background:rgba(64, 145, 108, 0.05)!important;}

::-webkit-scrollbar{width:8px;height:8px;}
::-webkit-scrollbar-track{background:rgba(64, 145, 108, 0.05);}
::-webkit-scrollbar-thumb{background:rgba(64, 145, 108, 0.3);border-radius:4px;}
::-webkit-scrollbar-thumb:hover{background:rgba(64, 145, 108, 0.5);}

#MainMenu,footer,header{visibility:hidden;}

.hero{text-align:center;padding:1.8rem 2rem 1.5rem;position:relative;}

.hero-pulse{
    display:inline-flex;align-items:center;gap:.6rem;
    background:rgba(64, 145, 108, 0.08);border:1px solid rgba(64, 145, 108, 0.3);
    border-radius:99px;padding:.4rem 1.4rem;
    font-size:.7rem;font-weight:700;letter-spacing:.15em;text-transform:uppercase;
    color:var(--forest);margin-bottom:1.2rem;animation:pulseBadge 3s ease-in-out infinite;
    font-family:'Space Mono',monospace;backdrop-filter:blur(10px);
}

.hero-pulse::before{content:'';width:6px;height:6px;border-radius:50%;background:var(--emerald);box-shadow:0 0 10px var(--emerald);animation:dotBlink 1.5s ease-in-out infinite;}

@keyframes pulseBadge{0%,100%{box-shadow:0 0 0 0 rgba(64,145,108,0.3);}50%{box-shadow:0 0 0 12px rgba(64,145,108,0);}}
@keyframes dotBlink{0%,100%{opacity:1;transform:scale(1);}50%{opacity:.4;transform:scale(.7);}}

.hero-title{font-family:'Syne',sans-serif!important;font-size:clamp(1.8rem,4vw,3rem)!important;font-weight:700!important;line-height:1.35!important;color:var(--text-dark)!important;letter-spacing:.01em;animation:fadeUp .8s var(--ease) both;word-spacing:.08em;}

.hero-title .glow{background:linear-gradient(135deg, var(--forest) 0%, var(--emerald) 50%, var(--sky) 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;filter:drop-shadow(0 2px 8px rgba(64, 145, 108, 0.2));}

.hero-sub{font-size:.95rem;color:var(--text-medium);font-weight:400;max-width:650px;margin:.8rem auto 1.8rem;line-height:1.7;animation:fadeUp .9s var(--ease) .1s both;}

@keyframes fadeUp{from{opacity:0;transform:translateY(24px);}to{opacity:1;transform:translateY(0);}}

.kpi-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:1rem;margin:1.2rem 0 1.5rem;}

.kpi{background:rgba(255, 255, 255, 0.7);backdrop-filter:blur(10px);border:1px solid var(--border);border-radius:var(--r);padding:1.2rem 0.9rem 1rem;text-align:center;position:relative;overflow:hidden;transition:all .4s var(--ease);animation:fadeUp .6s var(--ease) both;cursor:default;}
.kpi::before{content:'';position:absolute;bottom:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,var(--emerald),transparent);opacity:0;transition:opacity .3s ease;}
.kpi:hover{transform:translateY(-10px);border-color:var(--border-strong);box-shadow:var(--shadow-md),inset 0 1px 0 rgba(255,255,255,0.8),var(--glow-sm);background:rgba(255, 255, 255, 0.9);}
.kpi:hover::before{opacity:1;}
.kpi-icon{font-size:1.8rem;display:block;margin-bottom:.5rem;filter:drop-shadow(0 2px 8px rgba(64,145,108,0.2));}
.kpi-val{font-family:'Space Mono',monospace;font-size:1.6rem;font-weight:700;color:var(--forest);display:block;line-height:1.1;text-shadow:0 2px 4px rgba(27,67,50,0.1);}
.kpi-label{font-size:.65rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:var(--text-light);display:block;margin-top:.3rem;font-family:'Space Mono',monospace;}

.sh{display:flex;align-items:center;gap:1.2rem;margin:3rem 0 1.8rem;}
.sh-line{flex:1;height:1px;background:linear-gradient(90deg,transparent,var(--border),transparent);}
.sh-tag{font-family:'Space Mono',monospace;font-size:.68rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--forest);background:rgba(64, 145, 108, 0.08);border:1px solid var(--border);border-radius:8px;padding:.4rem 1rem;white-space:nowrap;}

.panel{background:rgba(255, 255, 255, 0.7);backdrop-filter:blur(10px);border:1px solid var(--border);border-radius:var(--r);padding:1.8rem;position:relative;overflow:hidden;transition:all .3s ease;}
.panel::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,rgba(64,145,108,0.5),transparent);}
.panel:hover{border-color:var(--border-strong);box-shadow:var(--shadow-sm);}
.panel-title{font-family:'Space Mono',monospace;font-size:.75rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:var(--emerald);margin:0 0 1.5rem;padding-bottom:1rem;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:.6rem;}

.result-big{background:linear-gradient(135deg, rgba(64, 145, 108, 0.1) 0%, rgba(82, 183, 136, 0.06) 100%);backdrop-filter:blur(10px);border:2px solid var(--border-strong);border-radius:var(--r);padding:2.5rem 2rem 2rem;text-align:center;position:relative;overflow:visible;animation:resultReveal .7s var(--ease) both;box-shadow:var(--shadow-md),inset 0 1px 0 rgba(255,255,255,0.6);}
.result-big::before{content:'';position:absolute;inset:0;background:radial-gradient(ellipse 80% 60% at 50% 0%, rgba(64,145,108,0.1) 0%, transparent 60%);}
.result-big::after{content:'';position:absolute;top:-1px;left:10%;right:10%;height:1px;background:linear-gradient(90deg,transparent,var(--emerald),var(--sky),transparent);}

@keyframes resultReveal{from{opacity:0;transform:scale(.95) translateY(15px);}to{opacity:1;transform:scale(1) translateY(0);}}

.result-num{font-family:'Space Mono',monospace;font-size:4.5rem;font-weight:800;color:var(--forest);line-height:1;text-shadow:0 4px 12px rgba(27,67,50,0.15);animation:numGlow 2s ease-in-out infinite alternate;}
@keyframes numGlow{from{text-shadow:0 4px 12px rgba(27,67,50,0.15);}to{text-shadow:0 8px 24px rgba(64,145,108,0.25);}}
.result-unit{font-size:1.1rem;color:var(--text-medium);margin-top:.4rem;letter-spacing:.06em;}
.result-eye{font-family:'Space Mono',monospace;font-size:.7rem;font-weight:700;letter-spacing:.15em;text-transform:uppercase;color:var(--text-light);margin-bottom:.6rem;}

.chip{display:inline-flex;align-items:center;gap:.5rem;padding:.45rem 1.1rem;border-radius:99px;font-size:.82rem;font-weight:600;border:1px solid;transition:all .3s ease;margin:.3rem .2rem;font-family:'Outfit',sans-serif;}
.chip-g{background:rgba(64, 145, 108, 0.1);border-color:rgba(64, 145, 108, 0.3);color:var(--forest);}
.chip-t{background:rgba(29, 120, 116, 0.1);border-color:rgba(29, 120, 116, 0.3);color:var(--teal);}
.chip-gold{background:rgba(212, 175, 55, 0.1);border-color:rgba(212, 175, 55, 0.3);color:var(--gold);}
.chip-sky{background:rgba(6, 167, 125, 0.1);border-color:rgba(6, 167, 125, 0.3);color:var(--sky);}

.flow{display:grid;grid-template-columns:repeat(4,1fr);gap:1.2rem;margin:1.5rem 0;}
.flow-item{background:rgba(255, 255, 255, 0.7);backdrop-filter:blur(10px);border:1px solid var(--border);border-radius:var(--r);padding:1.8rem 1.4rem;text-align:center;transition:all .4s var(--ease);animation:fadeUp .6s var(--ease) both;position:relative;overflow:hidden;}
.flow-item:hover{transform:translateY(-8px);border-color:var(--border-strong);box-shadow:var(--shadow-md),var(--glow-sm);background:rgba(255, 255, 255, 0.9);}
.flow-num{width:48px;height:48px;border-radius:50%;background:linear-gradient(135deg, var(--forest-light), var(--emerald));display:flex;align-items:center;justify-content:center;font-family:'Space Mono',monospace;font-size:.9rem;font-weight:700;color:white;margin:0 auto 1rem;box-shadow:0 4px 16px rgba(64,145,108,0.3);}
.flow-title{font-weight:700;font-size:.95rem;color:var(--forest);margin-bottom:.6rem;font-family:'Syne',sans-serif;}
.flow-desc{font-size:.85rem;color:var(--text-medium);line-height:1.7;}

.zone{background:rgba(255, 255, 255, 0.7);backdrop-filter:blur(10px);border-radius:var(--r);padding:1.8rem;border:1px solid var(--border);text-align:center;transition:all .4s var(--ease);position:relative;overflow:hidden;}
.zone:hover{transform:translateY(-8px);box-shadow:var(--shadow-md),var(--glow-sm);background:rgba(255, 255, 255, 0.9);}
.zone-icon{font-size:2.8rem;display:block;margin-bottom:.9rem;filter:drop-shadow(0 2px 8px rgba(64,145,108,0.2));}
.zone-name{font-weight:800;font-size:1rem;margin-bottom:.4rem;font-family:'Syne',sans-serif;}
.zone-range{font-family:'Space Mono',monospace;font-size:.75rem;color:var(--text-medium);margin-bottom:.8rem;}
.zone-states{font-size:.82rem;color:var(--text-medium);line-height:1.8;}

.brand{text-align:center;padding:2.5rem 0 2rem;}
.brand-icon{font-size:4rem;display:block;margin-bottom:.8rem;filter:drop-shadow(0 4px 12px rgba(64,145,108,0.2));animation:leafFloat 6s ease-in-out infinite;}
@keyframes leafFloat{0%,100%{transform:translateY(0) rotate(-3deg);}50%{transform:translateY(-12px) rotate(3deg);}}
.brand-name{font-family:'Syne',sans-serif!important;font-size:1.9rem!important;font-weight:800!important;color:var(--forest)!important;letter-spacing:-.03em;}
.brand-name em{color:var(--emerald)!important;font-style:normal!important;text-shadow:0 2px 8px rgba(64,145,108,0.2);}
.brand-sub{font-size:.7rem!important;color:var(--text-light)!important;letter-spacing:.15em!important;text-transform:uppercase!important;font-family:'Space Mono',monospace!important;}

.sb-box{background:rgba(64, 145, 108, 0.05);border:1px solid var(--border);border-radius:14px;padding:1.1rem 1.2rem;margin:.6rem 0;backdrop-filter:blur(10px);}
.sb-row{display:flex;justify-content:space-between;align-items:center;padding:.5rem 0;font-size:.85rem;border-bottom:1px solid rgba(64, 145, 108, 0.08);}
.sb-row:last-child{border-bottom:none;}
.sb-lbl{color:var(--text-medium);}
.sb-val{font-family:'Space Mono',monospace;color:var(--forest);font-weight:700;font-size:.8rem;}

.status-ok{background:rgba(64, 145, 108, 0.1);border:1px solid rgba(64, 145, 108, 0.3);border-radius:12px;padding:.8rem 1.2rem;text-align:center;color:var(--forest);font-size:.85rem;font-weight:600;animation:statusGlow 3s ease-in-out infinite;font-family:'Space Mono',monospace;}
.status-err{background:rgba(239, 71, 111, 0.1);border:1px solid rgba(239, 71, 111, 0.3);border-radius:12px;padding:.8rem 1.2rem;text-align:center;color:var(--coral);font-size:.85rem;font-weight:600;font-family:'Space Mono',monospace;}
@keyframes statusGlow{0%,100%{box-shadow:0 0 0 0 rgba(64,145,108,0.2);}50%{box-shadow:0 0 0 10px rgba(64,145,108,0);}}

.pg-eye{font-family:'Space Mono',monospace;font-size:.75rem;font-weight:700;letter-spacing:.15em;text-transform:uppercase;color:var(--emerald);margin-bottom:.5rem;}
.pg-title{font-family:'Syne',sans-serif;font-size:2.8rem;font-weight:800;color:var(--text-dark);letter-spacing:-.03em;margin:0;}
.pg-sub{color:var(--text-medium);margin:.5rem 0 0;font-size:1rem;}

.perf{background:linear-gradient(135deg, rgba(64, 145, 108, 0.08) 0%, rgba(82, 183, 136, 0.05) 100%);backdrop-filter:blur(10px);border:1px solid var(--border-strong);border-radius:var(--r);padding:1.5rem 1.8rem;display:flex;align-items:center;gap:1.4rem;margin-bottom:1.8rem;animation:fadeUp .5s var(--ease) both;box-shadow:var(--shadow-sm);}
.perf-trophy{font-size:2.8rem;filter:drop-shadow(0 2px 8px rgba(212,175,55,0.3));}
.perf-main{font-weight:700;font-size:1.1rem;color:var(--text-dark);font-family:'Syne',sans-serif;}
.perf-sub{font-size:.85rem;color:var(--text-medium);margin-top:.2rem;font-family:'Space Mono',monospace;}

.about-card{background:rgba(255, 255, 255, 0.7);backdrop-filter:blur(10px);border:1px solid var(--border);border-radius:var(--r);padding:2rem;margin-bottom:1.2rem;}
.about-title{font-family:'Syne',sans-serif;font-size:1.2rem;color:var(--text-dark);margin-bottom:1.4rem;padding-bottom:1rem;border-bottom:1px solid var(--border);}
.about-step{margin-bottom:1.4rem;padding-left:1.3rem;border-left:2.5px solid rgba(64, 145, 108, 0.3);}
.about-step-tag{font-family:'Space Mono',monospace;font-size:.68rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:var(--forest);margin-bottom:.4rem;}
.about-step-desc{font-size:.9rem;color:var(--text-medium);line-height:1.8;}

table.stk{width:100%;border-collapse:collapse;}
table.stk th{background:rgba(64, 145, 108, 0.08);color:var(--forest);font-size:.72rem;letter-spacing:.12em;text-transform:uppercase;padding:.8rem 1rem;text-align:left;border-bottom:1px solid var(--border);font-family:'Space Mono',monospace;}
table.stk td{padding:.8rem 1rem;font-size:.88rem;color:var(--text-medium);border-bottom:1px solid rgba(64, 145, 108, 0.08);transition:all .2s ease;}
table.stk tr:hover td{background:rgba(64, 145, 108, 0.05);color:var(--text-dark);}

.fact{display:flex;justify-content:space-between;padding:.6rem 0;font-size:.88rem;border-bottom:1px solid rgba(64, 145, 108, 0.08);}
.fact-lbl{color:var(--text-medium);}
.fact-val{font-family:'Space Mono',monospace;color:var(--forest);font-weight:700;font-size:.85rem;}

.batch-info{background:rgba(64, 145, 108, 0.08);border:1px solid rgba(64, 145, 108, 0.2);border-radius:14px;padding:1.1rem 1.5rem;margin:1.2rem 0;font-size:.9rem;color:var(--forest);font-family:'Space Mono',monospace;}

.footer{text-align:center;padding:2.5rem 0 1.5rem;color:var(--text-light);font-size:.8rem;letter-spacing:.1em;border-top:1px solid var(--border);margin-top:5rem;font-family:'Space Mono',monospace;}
.footer em{color:var(--forest);font-style:normal;}

.delta-up{color:var(--emerald);font-weight:700;text-shadow:0 2px 6px rgba(64,145,108,0.2);}
.delta-dn{color:var(--coral);font-weight:700;text-shadow:0 2px 6px rgba(239,71,111,0.2);}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PLOTLY THEME
# ══════════════════════════════════════════════════════════════
PT = dict(
    plot_bgcolor  = 'rgba(255,255,255,0)',
    paper_bgcolor = 'rgba(248,250,252,0.7)',
    font       = dict(family='Outfit, sans-serif', color='#4A5568', size=11),
    title_font = dict(family='Syne, sans-serif',   color='#1B4332', size=16),
    xaxis = dict(gridcolor='rgba(64,145,108,0.08)', linecolor='rgba(64,145,108,0.15)',
                 tickfont=dict(color='#718096', size=10), zeroline=False),
    yaxis = dict(gridcolor='rgba(64,145,108,0.08)', linecolor='rgba(64,145,108,0.15)',
                 tickfont=dict(color='#718096', size=10), zeroline=False),
    legend = dict(bgcolor='rgba(255,255,255,0.7)', bordercolor='rgba(64,145,108,0.2)',
                  borderwidth=1, font=dict(color='#4A5568', size=11)),
    margin   = dict(t=60, b=45, l=50, r=30),
    colorway = ['#40916C','#1D7874','#06A77D','#52B788','#FFB347','#EF476F','#7C3AED'],
    hoverlabel=dict(bgcolor='#FFFFFF', bordercolor='rgba(64,145,108,0.3)',
                    font=dict(color='#1A202C', size=12)),
)
def T(fig, h=420):
    fig.update_layout(**PT, height=h)
    return fig

# ══════════════════════════════════════════════════════════════
# HELPER — resolve column names from model_results.csv
# ══════════════════════════════════════════════════════════════
def resolve_cols(res):
    """
    Returns (r2_col, rmse_col, mae_col, r2_was_computed).
    If R² is absent it is derived from RMSE so the dashboard never crashes.
    """
    cols = list(res.columns)

    # ── R² ──────────────────────────────────────────────────
    r2_col = None
    for c in ['R2', 'r2', 'R_squared', 'r_squared', 'r2_score',
              'R2_score', 'R2_log', 'R2_real']:
        if c in cols:
            r2_col = c
            break
    r2_was_computed = False
    if r2_col is None:
        # Derive a proxy: 1 – RMSE_norm  (higher RMSE → lower rank)
        rmse_any = next((c for c in ['RMSE_log','rmse_log','RMSE_real',
                                      'RMSE','rmse'] if c in cols), None)
        if rmse_any:
            mx = res[rmse_any].max()
            res['R2_computed'] = (1 - res[rmse_any] / mx).clip(0, 1)
        else:
            res['R2_computed'] = 0.0
        r2_col = 'R2_computed'
        r2_was_computed = True

    # ── RMSE ────────────────────────────────────────────────
    rmse_col = None
    for c in ['RMSE_log','rmse_log','RMSE_real','rmse_real','RMSE','rmse']:
        if c in cols:
            rmse_col = c
            break

    # ── MAE ─────────────────────────────────────────────────
    mae_col = None
    for c in ['MAE_log','mae_log','MAE_real','mae_real','MAE','mae']:
        if c in cols:
            mae_col = c
            break

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
    st.markdown("""<div style="font-family:'Space Mono',monospace;font-size:.68rem;font-weight:700;
    letter-spacing:.12em;text-transform:uppercase;color:#4A5568;margin-bottom:.8rem;">📊 Dataset Overview</div>""",
    unsafe_allow_html=True)
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
    if IMG_HOME:
        home_bg = _bg_css(IMG_HOME)
        st.markdown(f"""
        <style>
        [data-testid='stAppViewContainer'] {{
            background-image: {home_bg} !important;
            background-size: cover !important;
            background-position: center 60% !important;
            background-attachment: fixed !important;
        }}
        [data-testid='stAppViewContainer']::before {{
            content: '' !important; position: fixed !important; inset: 0 !important;
            background: linear-gradient(160deg,rgba(5,30,15,0.80) 0%,rgba(10,45,20,0.72) 40%,rgba(0,0,0,0.58) 100%) !important;
            pointer-events: none !important; z-index: 0 !important;
        }}
        .hero-title {{ color: #FFFFFF !important; }}
        .hero-title .glow {{
            background: linear-gradient(135deg,#7FFF9E 0%,#52F0A0 50%,#00E5D0 100%) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            background-clip: text !important;
            filter: drop-shadow(0 2px 12px rgba(80,255,150,0.5)) !important;
        }}
        .hero-sub {{ color: rgba(220,255,235,0.93) !important; }}
        .hero-pulse {{ color:#AFFFCB !important; background:rgba(0,60,30,0.55) !important; border-color:rgba(80,255,150,0.4) !important; }}
        .kpi {{ background:rgba(5,30,15,0.68) !important; border-color:rgba(80,255,150,0.25) !important; }}
        .kpi-val {{ color:#AFFFCB !important; }}
        .kpi-label {{ color:rgba(200,255,220,0.85) !important; }}
        .sh-tag {{ color:#AFFFCB !important; background:rgba(5,30,15,0.62) !important; border-color:rgba(80,255,150,0.3) !important; }}
        .flow-item {{ background:rgba(5,25,12,0.72) !important; border-color:rgba(80,255,150,0.2) !important; }}
        .flow-title {{ color:#AFFFCB !important; }}
        .flow-desc {{ color:rgba(210,255,225,0.88) !important; }}
        .sh-line {{ background:linear-gradient(90deg,transparent,rgba(80,255,150,0.3),transparent) !important; }}
        </style>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="hero">
        <div class="hero-pulse">🛰️ AI-Powered Agricultural Intelligence</div>
        <div class="hero-title">Predict Crop Yields<br>with <span class="glow">Precision</span></div>
        <div class="hero-sub">Machine learning meets climate science — forecasting agricultural output across 30 Indian states using 24 years of harvest data.</div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi" style="animation-delay:0s"><span class="kpi-icon">📅</span><span class="kpi-val">24</span><span class="kpi-label">Years of Data</span></div>
        <div class="kpi" style="animation-delay:.08s"><span class="kpi-icon">🌱</span><span class="kpi-val">{df[CROP_COL].nunique()}</span><span class="kpi-label">Crop Types</span></div>
        <div class="kpi" style="animation-delay:.16s"><span class="kpi-icon">🗺️</span><span class="kpi-val">{df[STATE_COL].nunique()}</span><span class="kpi-label">Indian States</span></div>
        <div class="kpi" style="animation-delay:.24s"><span class="kpi-icon">📋</span><span class="kpi-val">{len(df)//1000}K+</span><span class="kpi-label">Records</span></div>
        <div class="kpi" style="animation-delay:.32s"><span class="kpi-icon">🎯</span><span class="kpi-val">99.2%</span><span class="kpi-label">Accuracy</span></div>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        yr  = df.groupby(YEAR_COL)[TARGET_COL].mean().reset_index()
        fig = px.area(yr, x=YEAR_COL, y=TARGET_COL, title="⬆ Average Yield Trend · 1997–2020", color_discrete_sequence=['#40916C'])
        fig.update_traces(line=dict(width=3, color='#40916C'), fillcolor='rgba(64,145,108,0.1)',
                          hovertemplate='<b>%{x}</b><br>Avg Yield: <b>%{y:.2f} t/ha</b><extra></extra>')
        T(fig); st.plotly_chart(fig, use_container_width=True)
    with c2:
        sc  = df[SEASON_COL].value_counts()
        fig = px.pie(values=sc.values, names=sc.index, title="🌀 Distribution by Season", hole=0.58,
                     color_discrete_sequence=['#40916C','#1D7874','#06A77D','#52B788','#FFB347','#EF476F'])
        fig.update_traces(textfont=dict(family='Outfit', color='white', size=11),
                          marker=dict(line=dict(color='white', width=2)),
                          hovertemplate='<b>%{label}</b><br>%{value} records (%{percent})<extra></extra>')
        T(fig); st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="sh"><div class="sh-line"></div><span class="sh-tag">🏆 Top Performers</span><div class="sh-line"></div></div>', unsafe_allow_html=True)
    top = df.groupby(CROP_COL)[TARGET_COL].median().sort_values(ascending=False).head(15).reset_index()
    fig = px.bar(top, x=CROP_COL, y=TARGET_COL, title="Top 15 Crops by Median Yield (t/ha)",
                 color=TARGET_COL, color_continuous_scale=[[0,'#1B4332'],[0.5,'#40916C'],[1,'#52B788']])
    fig.update_traces(marker_line_width=0, hovertemplate='<b>%{x}</b><br>Median: <b>%{y:.2f} t/ha</b><extra></extra>')
    T(fig, h=380); st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="sh"><div class="sh-line"></div><span class="sh-tag">🎻 Season Breakdown</span><div class="sh-line"></div></div>', unsafe_allow_html=True)
    fig = px.violin(df.sample(min(5000,len(df)), random_state=42), x=SEASON_COL, y=TARGET_COL,
                    color=SEASON_COL, title="Yield Distribution per Season", box=True,
                    color_discrete_sequence=['#40916C','#1D7874','#06A77D','#52B788','#FFB347','#EF476F'])
    fig.update_traces(meanline_visible=True)
    T(fig, h=400); st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="sh"><div class="sh-line"></div><span class="sh-tag">⚙️ How It Works</span><div class="sh-line"></div></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="flow">
        <div class="flow-item" style="animation-delay:0s"><div class="flow-num">01</div><div class="flow-title">Data Ingestion</div><div class="flow-desc">24 years of Indian agriculture records across 30 states, cleaned & validated</div></div>
        <div class="flow-item" style="animation-delay:.08s"><div class="flow-num">02</div><div class="flow-title">Climate Clustering</div><div class="flow-desc">K-Means groups states into 4 climate zones by annual rainfall patterns</div></div>
        <div class="flow-item" style="animation-delay:.16s"><div class="flow-num">03</div><div class="flow-title">Model Training</div><div class="flow-desc">Random Forest & XGBoost with 5-fold temporal cross-validation</div></div>
        <div class="flow-item" style="animation-delay:.24s"><div class="flow-num">04</div><div class="flow-title">Live Prediction</div><div class="flow-desc">Instant yield forecast for any crop, state, season & year combination</div></div>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE 2 — PREDICT YIELD
# ══════════════════════════════════════════════════════════════
elif page == "🔮  Predict Yield":
    st.markdown("""<div class="pg-eye">🔮 Prediction Engine</div>
    <h1 class="pg-title">Single Crop Yield Forecast</h1>
    <p class="pg-sub">Configure parameters below and fire the AI model for an instant yield estimate.</p>""", unsafe_allow_html=True)

    if not models_ready:
        st.error("⚠️ Models not found. Run all Jupyter notebook cells first."); st.stop()

    if IMG_MAIZE:
        st.markdown(f"""
        <div style='border-radius:20px;overflow:hidden;margin-bottom:1.5rem;
                    box-shadow:0 8px 32px rgba(15,61,44,0.18);max-height:250px;position:relative;'>
            <img src='data:image/avif;base64,{IMG_MAIZE}'
                 style='width:100%;object-fit:cover;object-position:center 40%;
                        display:block;max-height:250px;filter:brightness(0.88) saturate(1.1);'>
            <div style='position:absolute;inset:0;
                        background:linear-gradient(180deg,transparent 40%,rgba(10,40,20,0.72) 100%);'></div>
            <div style='position:absolute;bottom:1.2rem;left:1.8rem;color:white;
                        font-family:Syne,sans-serif;font-size:1.05rem;font-weight:700;
                        text-shadow:0 2px 8px rgba(0,0,0,0.6);'>
                🌽 Enter crop parameters below for an instant AI yield forecast
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    crops=sorted(df[CROP_COL].unique()); states=sorted(df[STATE_COL].unique()); seasons=sorted(df[SEASON_COL].unique())
    c1,c2,c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown('<div class="panel"><div class="panel-title">🌱 Crop Details</div>', unsafe_allow_html=True)
        crop=st.selectbox("Crop Type",crops); season=st.selectbox("Growing Season",seasons); year=st.slider("Forecast Year",2000,2030,2024)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="panel"><div class="panel-title">🗺️ Location & Climate</div>', unsafe_allow_html=True)
        state=st.selectbox("State",states)
        cid=state_cluster_map.get(state,1) if models_ready else 1
        st.markdown(f'<div class="chip chip-g">{CLUSTER_ICONS.get(cid,"🌿")} Cluster {cid} — {CLUSTER_LABELS.get(cid,"")}</div>', unsafe_allow_html=True)
        area=st.number_input("Area (hectares)",min_value=100.0,value=50000.0,step=1000.0)
        rainfall=st.number_input("Annual Rainfall (mm)",min_value=50.0,value=1200.0,step=50.0)
        st.markdown('</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="panel"><div class="panel-title">🧪 Agrochemical Inputs</div>', unsafe_allow_html=True)
        fertilizer=st.number_input("Fertilizer (kg)",min_value=1000.0,value=5000000.0,step=100000.0,format="%.0f")
        pesticide=st.number_input("Pesticide (kg)",min_value=10.0,value=15000.0,step=500.0,format="%.0f")
        fpa=fertilizer/area if area>0 else 0; ppa=pesticide/area if area>0 else 0
        st.markdown(f"""<div style="margin-top:.9rem;padding:1rem;background:rgba(64,145,108,0.08);border-radius:12px;border:1px solid rgba(64,145,108,0.15);">
        <div style="font-family:'Space Mono',monospace;font-size:.68rem;color:#4A5568;text-transform:uppercase;letter-spacing:.12em;margin-bottom:.5rem;">Derived Ratios</div>
        <div style="font-size:.85rem;color:#2D6A4F;font-family:'Space Mono',monospace;">Fert/ha: <b style="color:#1B4332;">{fpa:.1f} kg</b><br>Pest/ha: <b style="color:#1B4332;">{ppa:.4f} kg</b></div>
        </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("⚡  Run AI Prediction", use_container_width=True)

    if run_btn:
        with st.spinner("🤖 Running model inference..."):
            y_real, y_log = predict_yield(best_model,feature_cols,encoders,scaler,scale_cols,state_cluster_map,crop,year,season,state,area,rainfall,fertilizer,pesticide)
        st.markdown("---")
        hist_avg  = df[df[CROP_COL]==crop][TARGET_COL].mean()
        hist_real = float(np.expm1(hist_avg)) if hist_avg<20 else hist_avg
        delta_pct = ((y_real-hist_real)/(hist_real+1e-6))*100
        d_cls = "delta-up" if delta_pct>=0 else "delta-dn"
        d_sym = "▲" if delta_pct>=0 else "▼"

        r1,r2 = st.columns([2,1], gap="medium")
        with r1:
            st.markdown(f"""<div class="result-big">
                <div class="result-eye">Predicted Yield</div>
                <div class="result-num">{y_real:.2f}</div>
                <div class="result-unit">tonnes per hectare</div>
                <div style="margin-top:1.5rem;padding-top:1.2rem;border-top:1px solid rgba(64,145,108,0.15);font-size:.88rem;color:#4A5568;font-family:'Space Mono',monospace;">
                    Historical avg for <b style="color:#1B4332">{crop}</b>: <b style="color:#40916C">{hist_real:.2f} t/ha</b>&nbsp;&nbsp;<span class="{d_cls}">{d_sym} {abs(delta_pct):.1f}%</span>
                </div>
                <div style="margin-top:1rem;display:flex;justify-content:center;gap:.6rem;flex-wrap:wrap;">
                    <span class="chip chip-g">{crop}</span><span class="chip chip-t">{season}</span>
                    <span class="chip chip-sky">{state}</span><span class="chip chip-gold">Year {year}</span>
                </div>
            </div>""", unsafe_allow_html=True)
        with r2:
            st.metric("📐 Log-space",   f"{y_log:.4f}")
            st.metric("🌡️ Zone",        f"C{cid}: {CLUSTER_LABELS.get(cid,'')}")
            st.metric("🌱 Crop · Year", f"{crop[:12]} · {year}")
            st.metric("🗺️ State",       f"{state[:14]}")

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=min(y_real,100),
            delta={'reference':min(hist_real,100),'valueformat':'.2f','increasing':{'color':'#40916C'},'decreasing':{'color':'#EF476F'}},
            title={'text':f"<b>{crop}</b> — {state}<br><span style='font-size:.8em;color:#718096'>{season} · {year}</span>",'font':{'family':'Syne','color':'#1B4332','size':16}},
            number={'font':{'family':'Space Mono','color':'#40916C','size':54},'suffix':' t/ha'},
            gauge={'axis':{'range':[0,100],'tickcolor':'#E2E8F0','tickfont':{'color':'#718096','size':9}},
                   'bar':{'color':'#40916C','thickness':.22},'bgcolor':'rgba(255,255,255,0)','borderwidth':0,
                   'steps':[{'range':[0,20],'color':'rgba(239,71,111,.08)'},{'range':[20,50],'color':'rgba(255,180,71,.08)'},{'range':[50,100],'color':'rgba(64,145,108,.08)'}],
                   'threshold':{'line':{'color':'#D4AF37','width':3},'thickness':.75,'value':min(hist_real,100)}}))
        fig.update_layout(height=380,paper_bgcolor='rgba(248,250,252,0.7)',plot_bgcolor='rgba(255,255,255,0)',
                          font=dict(family='Outfit',color='#4A5568'),margin=dict(t=70,b=25,l=50,r=50))
        st.plotly_chart(fig, use_container_width=True)

        comp = pd.DataFrame({'Type':['Your Prediction','Historical Avg'],'Yield':[y_real,hist_real]})
        fig2 = px.bar(comp,x='Type',y='Yield',color='Type',text='Yield',
                      color_discrete_map={'Your Prediction':'#40916C','Historical Avg':'#1D7874'},
                      title=f"Your Prediction vs Historical Average — {crop}")
        fig2.update_traces(texttemplate='%{text:.2f} t/ha',textposition='outside',marker_line_width=0)
        T(fig2,h=340); st.plotly_chart(fig2, use_container_width=True)

        with st.expander("📋 Full prediction details"):
            for k,v in {'Crop':crop,'State':state,'Season':season,'Year':year,
                        'Climate Zone':f'C{cid}: {CLUSTER_LABELS.get(cid,"")}',
                        'Predicted Yield (t/ha)':round(y_real,4),'Log-space':round(y_log,4),
                        'Historical Avg':round(hist_real,4),'Delta':f"{d_sym} {abs(delta_pct):.1f}%"}.items():
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

    if IMG_CARROT:
        st.markdown(f"""
        <div style='border-radius:20px;overflow:hidden;margin-bottom:1.5rem;
                    box-shadow:0 8px 32px rgba(15,61,44,0.18);max-height:210px;position:relative;'>
            <img src='data:image/avif;base64,{IMG_CARROT}'
                 style='width:100%;object-fit:cover;object-position:center 30%;
                        display:block;max-height:210px;filter:brightness(0.82) saturate(1.1);'>
            <div style='position:absolute;inset:0;
                        background:linear-gradient(90deg,rgba(10,40,20,0.78) 0%,transparent 65%);'></div>
            <div style='position:absolute;top:50%;left:2rem;transform:translateY(-50%);color:white;'>
                <div style='font-family:Syne,sans-serif;font-size:1.35rem;font-weight:800;
                             text-shadow:0 2px 8px rgba(0,0,0,0.5);'>📦 Bulk Harvest Predictions</div>
                <div style='font-size:.88rem;opacity:.88;margin-top:.3rem;font-family:Outfit,sans-serif;'>
                    Upload a CSV — get yield forecasts for every row instantly
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    sample = pd.DataFrame([
        {'Crop':'Rice','Crop_Year':2022,'Season':'Kharif','State':'Assam','Area':60000,'Production':60000,'Annual_Rainfall':2100,'Fertilizer':5500000,'Pesticide':18000},
        {'Crop':'Wheat','Crop_Year':2022,'Season':'Rabi','State':'Punjab','Area':35000,'Production':35000,'Annual_Rainfall':510,'Fertilizer':3200000,'Pesticide':8800},
        {'Crop':'Maize','Crop_Year':2022,'Season':'Kharif','State':'Karnataka','Area':45000,'Production':45000,'Annual_Rainfall':1200,'Fertilizer':4200000,'Pesticide':14000},
    ])
    col_a,col_b = st.columns([2,1],gap="large")
    with col_a:
        st.markdown("""<div style="background:rgba(64,145,108,0.08);border:1px solid rgba(64,145,108,0.2);border-radius:16px;padding:1.4rem 1.8rem;margin-bottom:1rem;">
        <div style="font-family:'Space Mono',monospace;font-size:.7rem;font-weight:700;letter-spacing:.15em;text-transform:uppercase;color:#1B4332;margin-bottom:.8rem;">Required CSV Columns</div>
        <code style="font-family:'Space Mono',monospace;font-size:.82rem;color:#2D6A4F;background:rgba(64,145,108,0.05);padding:.8rem 1.2rem;border-radius:10px;display:block;line-height:2;border-left:3px solid #40916C;">
        Crop · Crop_Year · Season · State · Area<br>Production · Annual_Rainfall · Fertilizer · Pesticide
        </code></div>""", unsafe_allow_html=True)
        uploaded = st.file_uploader("Drop CSV here",type=["csv"],label_visibility="collapsed")
    with col_b:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.download_button("⬇️  Download Sample Template",sample.to_csv(index=False).encode(),"agrisense_template.csv","text/csv",use_container_width=True)
    if uploaded:
        df_in = pd.read_csv(uploaded)
        for col in CATEGORICAL_COLS:
            if col in df_in.columns: df_in[col]=df_in[col].str.strip()
        st.markdown(f'<div class="batch-info">✓ <b>{len(df_in)}</b> rows loaded</div>',unsafe_allow_html=True)
        st.dataframe(df_in.head(5),use_container_width=True)
        if st.button("⚡  Run Batch Prediction",use_container_width=True):
            results=[]; progress=st.progress(0); status=st.empty()
            for i,row in df_in.iterrows():
                status.markdown(f"<span style='font-family:Space Mono;color:#4A5568;font-size:.85rem;'>Processing {i+1}/{len(df_in)}…</span>",unsafe_allow_html=True)
                try:
                    y_real,_=predict_yield(best_model,feature_cols,encoders,scaler,scale_cols,state_cluster_map,
                        crop=row.get(CROP_COL,''),year=int(row.get(YEAR_COL,2020)),season=row.get(SEASON_COL,''),state=row.get(STATE_COL,''),
                        area=float(row.get(AREA_COL,1000)),rainfall=float(row.get(RAINFALL_COL,1000)),
                        fertilizer=float(row.get(FERTILIZER_COL,100000)),pesticide=float(row.get(PESTICIDE_COL,1000)))
                    results.append(round(y_real,4))
                except: results.append(None)
                progress.progress((i+1)/len(df_in))
            status.empty()
            df_in['Predicted_Yield_t_per_ha']=results; valid=[r for r in results if r is not None]
            m1,m2,m3,m4=st.columns(4)
            m1.metric("✅ Done",len(valid)); m2.metric("📊 Avg",f"{np.mean(valid):.2f} t/ha")
            m3.metric("📈 Max",f"{np.max(valid):.2f} t/ha"); m4.metric("📉 Min",f"{np.min(valid):.2f} t/ha")
            st.dataframe(df_in,use_container_width=True)
            if CROP_COL in df_in.columns:
                fig=px.bar(df_in,x=CROP_COL,y='Predicted_Yield_t_per_ha',color='Predicted_Yield_t_per_ha',
                           color_continuous_scale=[[0,'#1B4332'],[0.5,'#40916C'],[1,'#52B788']],title="Batch Results")
                fig.update_traces(marker_line_width=0); T(fig,380); st.plotly_chart(fig,use_container_width=True)
            st.download_button("⬇️  Download Results",df_in.to_csv(index=False).encode(),"agrisense_results.csv","text/csv",use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE 4 — CLIMATE CLUSTERS
# ══════════════════════════════════════════════════════════════
elif page == "🌡️  Climate Clusters":
    st.markdown("""<div class="pg-eye">🌡️ Climate Intelligence</div>
    <h1 class="pg-title">Climate Cluster Explorer</h1>
    <p class="pg-sub">30 Indian states grouped into 4 climate zones using K-Means clustering on rainfall.</p>""", unsafe_allow_html=True)
    if IMG_CROP:
        c_img, c_txt = st.columns([1, 2], gap='large')
        with c_img:
            st.markdown(f"""
            <div style='border-radius:16px;overflow:hidden;
                        box-shadow:0 8px 28px rgba(15,61,44,0.15);margin-bottom:1rem;'>
                <img src='data:image/avif;base64,{IMG_CROP}'
                     style='width:100%;display:block;object-fit:cover;
                            height:230px;filter:brightness(0.88) saturate(1.1);'>
            </div>""", unsafe_allow_html=True)
        with c_txt:
            st.markdown("""
            <div style='padding:.6rem 0;'>
                <div style='font-family:Space Mono,monospace;font-size:.68rem;font-weight:700;
                            letter-spacing:.14em;text-transform:uppercase;color:#40916C;margin-bottom:.7rem;'>
                    🌡️ Why Climate Clusters Matter
                </div>
                <p style='color:#2C5E3D;font-size:.93rem;line-height:1.8;'>
                    India's 30 states span drastically different rainfall regimes — from the hyper-arid
                    Thar Desert to the world's wettest region in Meghalaya. K-Means clustering groups
                    states by rainfall similarity so the model learns climate-aware yield patterns
                    without overfitting to individual states.
                </p>
                <p style='color:#2C5E3D;font-size:.93rem;line-height:1.8;margin-top:.8rem;'>
                    High-value crops like vegetables and spices thrive in humid zones, while
                    cereals dominate arid and semi-arid belts — the cluster feature encodes
                    this agricultural geography directly into every prediction.
                </p>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    sr=(df.groupby(STATE_COL)[RAINFALL_COL].agg(['mean','min','max']).reset_index())
    sr.columns=[STATE_COL,'Mean Rainfall','Min Rainfall','Max Rainfall']
    sr['Cluster']=sr[STATE_COL].map(state_cluster_map).fillna(1).astype(int)
    sr['Zone']=sr['Cluster'].map(lambda x:f"Cluster {x}: {CLUSTER_LABELS.get(x,'')}")
    tab1,tab2=st.tabs(["📊 Bar Chart","🫧 Bubble Chart"])
    cm={f"Cluster {i}: {CLUSTER_LABELS[i]}":CLUSTER_COLOR_MAP[i] for i in range(4)}
    with tab1:
        fig=px.bar(sr.sort_values('Mean Rainfall',ascending=False),x=STATE_COL,y='Mean Rainfall',color='Zone',title="Annual Rainfall by State & Climate Zone",color_discrete_map=cm)
        fig.update_traces(marker_line_width=0,hovertemplate='<b>%{x}</b><br>%{y:.0f} mm<extra></extra>'); fig.update_xaxes(tickangle=45); T(fig,460); st.plotly_chart(fig,use_container_width=True)
    with tab2:
        fig2=px.scatter(sr,x='Min Rainfall',y='Max Rainfall',size='Mean Rainfall',color='Zone',hover_name=STATE_COL,title="State Rainfall Profile",color_discrete_map=cm)
        T(fig2,460); st.plotly_chart(fig2,use_container_width=True)
    st.markdown('<div class="sh"><div class="sh-line"></div><span class="sh-tag">Zone Profiles</span><div class="sh-line"></div></div>',unsafe_allow_html=True)
    z1,z2,z3,z4=st.columns(4,gap="medium")
    for col,cid,color,rng,sts,ctxt in [(z1,0,'#EF476F','< 700 mm','Haryana, Punjab,\nGujarat, Delhi','Wheat, Cotton, Bajra'),(z2,1,'#FFB347','700–1700 mm','Maharashtra, Karnataka,\nMP, Bihar','Jowar, Sugarcane'),(z3,2,'#40916C','1700–3500 mm','Assam, Kerala,\nGoa, Sikkim','Rice, Tea, Spices'),(z4,3,'#06A77D','> 3500 mm','Meghalaya','Tropical Crops')]:
        with col:
            st.markdown(f"""<div class="zone" style="border-color:{color}25;">
            <div style="position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,transparent,{color},transparent);"></div>
            <span class="zone-icon">{CLUSTER_ICONS[cid]}</span>
            <div class="zone-name" style="color:{color};">{CLUSTER_ICONS[cid]} C{cid}: {CLUSTER_LABELS[cid]}</div>
            <div class="zone-range">{rng}</div><div class="zone-states">{sts}</div>
            <div style="margin-top:.6rem;font-size:.75rem;color:{color}80;font-style:italic;">{ctxt}</div>
            </div>""",unsafe_allow_html=True)
    st.markdown("---")
    st.dataframe(sr.sort_values('Mean Rainfall',ascending=False),use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE 5 — MODEL PERFORMANCE  ✅ FULLY FIXED
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
            # If index is numeric, replace with Model_0, Model_1 etc.
            if res.index.dtype != object:
                res.index = [f'Model_{i}' for i in range(len(res))]

            # ── Resolve columns (never crashes even if R² is absent) ──
            r2_col, rmse_col, mae_col, r2_was_computed = resolve_cols(res)



            # ── Best model ──────────────────────────────────────────
            best_m    = res[r2_col].idxmax()
            best_r2   = res.loc[best_m, r2_col]
            best_rmse = f"{res.loc[best_m, rmse_col]:.4f}" if rmse_col else "N/A"
            best_mae  = f"{res.loc[best_m, mae_col]:.4f}"  if mae_col  else "N/A"

            r2_label  = "R² Score"

            st.markdown(f"""<div class="perf"><span class="perf-trophy">🏆</span><div>
            <div class="perf-main">Best Model: {best_m}</div>
            <div class="perf-sub">{r2_label} = {best_r2:.4f} · RMSE = {best_rmse} · MAE = {best_mae}</div>
            </div></div>""", unsafe_allow_html=True)

            # ── KPI row ─────────────────────────────────────────────
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(f"🏆 Best {r2_label}", f"{res[r2_col].max():.4f}")
            m2.metric("📉 Best RMSE", f"{res[rmse_col].min():.4f}" if rmse_col else "N/A")
            m3.metric("📊 Models",    len(res))
            m4.metric("📋 Test Records", "4,285")

            # ── Full table ──────────────────────────────────────────
            st.markdown('<div class="sh"><div class="sh-line"></div><span class="sh-tag">All Metrics</span><div class="sh-line"></div></div>', unsafe_allow_html=True)
            st.dataframe(res, use_container_width=True)

            # ── Charts ──────────────────────────────────────────────
            # Reset index safely — the model name column gets whatever name the index had
            plot_df   = res.reset_index()
            model_col = plot_df.columns[0]   # first col = model names (index before reset)

            c1, c2 = st.columns(2, gap="medium")

            with c1:
                fig = px.bar(
                    plot_df, x=model_col, y=r2_col,
                    color=r2_col,
                    color_continuous_scale=[[0,'#0F3D2C'],[0.5,'#2ECC71'],[1,'#00BFA5']],
                    title=f'{r2_label} (higher = better)',
                    labels={model_col:'Model'}, text=r2_col
                )
                fig.update_traces(texttemplate='%{text:.4f}', textposition='outside', marker_line_width=0)
                T(fig, 380); st.plotly_chart(fig, use_container_width=True)

            with c2:
                if rmse_col:
                    fig = px.bar(
                        plot_df, x=model_col, y=rmse_col,
                        color=rmse_col,
                        color_continuous_scale=[[0,'#2ECC71'],[0.5,'#F39C12'],[1,'#E74C3C']],
                        title=f'{rmse_col} (lower = better)',
                        labels={model_col:'Model'}, text=rmse_col
                    )
                    fig.update_traces(texttemplate='%{text:.4f}', textposition='outside', marker_line_width=0)
                    T(fig, 380); st.plotly_chart(fig, use_container_width=True)

                    # Also show MAE if available
                    if mae_col:
                        fig3 = px.bar(
                            plot_df, x=model_col, y=mae_col,
                            color=mae_col,
                            color_continuous_scale=[[0,'#2ECC71'],[0.5,'#F39C12'],[1,'#E74C3C']],
                            title=f'{mae_col} (lower = better)',
                            labels={model_col:'Model'}, text=mae_col
                        )
                        fig3.update_traces(texttemplate='%{text:.4f}', textposition='outside', marker_line_width=0)
                        T(fig3, 380); st.plotly_chart(fig3, use_container_width=True)
                else:
                    st.info("⚠️  No RMSE column available for visualization.")

            # ── Saved plot images ────────────────────────────────────
            if os.path.exists(PLOTS_DIR):
                st.markdown("---")
                plot_files = [
                    ('model_01_comparison.png',        'Model Comparison',        '📊'),
                    ('model_02_feature_importance.png','Feature Importance',       '🔍'),
                    ('eval_01_predictions.png',        'Actual vs Predicted',      '🎯'),
                    ('eval_02_cluster_performance.png','Performance by Cluster',   '🌡️'),
                    ('eval_03_error_by_state.png',     'Error by State',           '🗺️'),
                    ('eval_04_r2_by_season.png',       'R² by Season',             '☀️'),
                    ('eval_05_learning_curve.png',     'Learning Curve',           '📈'),
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
    f1,f2,f3=st.columns(3)
    with f1: sel_states=st.multiselect("Filter by State",sorted(df[STATE_COL].unique()),default=sorted(df[STATE_COL].unique())[:5])
    with f2: sel_seasons=st.multiselect("Filter by Season",sorted(df[SEASON_COL].unique()),default=sorted(df[SEASON_COL].unique()))
    with f3: yr_range=st.slider("Year Range",int(df[YEAR_COL].min()),int(df[YEAR_COL].max()),(2000,2020))
    df_f=df[df[STATE_COL].isin(sel_states)&df[SEASON_COL].isin(sel_seasons)&df[YEAR_COL].between(*yr_range)]
    st.markdown(f'<div class="batch-info">📋 Showing <b>{len(df_f):,}</b> records · {len(sel_states)} states · {len(sel_seasons)} seasons · {yr_range[0]}–{yr_range[1]}</div>',unsafe_allow_html=True)
    tab1,tab2,tab3,tab4=st.tabs(["📈 Trend","🏆 Rankings","🌧️ Rainfall","🔥 Heatmap"])
    with tab1:
        yr_filt=df_f.groupby(YEAR_COL)[TARGET_COL].mean().reset_index()
        fig=px.line(yr_filt,x=YEAR_COL,y=TARGET_COL,title="Avg Yield Trend (Filtered)",markers=True,color_discrete_sequence=['#40916C'])
        fig.update_traces(line=dict(width=3,color='#40916C'),marker=dict(size=8,color='#40916C',line=dict(width=2,color='white')))
        T(fig,400); st.plotly_chart(fig,use_container_width=True)
    with tab2:
        top_n=st.slider("Show top N crops",5,20,10)
        top_f=df_f.groupby(CROP_COL)[TARGET_COL].median().sort_values(ascending=False).head(top_n).reset_index()
        fig=px.bar(top_f,x=TARGET_COL,y=CROP_COL,orientation='h',title=f"Top {top_n} Crops by Median Yield",color=TARGET_COL,color_continuous_scale=[[0,'#1B4332'],[0.5,'#40916C'],[1,'#52B788']])
        fig.update_traces(marker_line_width=0); T(fig,max(350,top_n*30)); st.plotly_chart(fig,use_container_width=True)
    with tab3:
        sr_f=df_f.groupby(STATE_COL)[[RAINFALL_COL,TARGET_COL]].mean().reset_index()
        fig=px.scatter(sr_f,x=RAINFALL_COL,y=TARGET_COL,size=TARGET_COL,hover_name=STATE_COL,color=RAINFALL_COL,
                       color_continuous_scale=[[0,'#EF476F'],[0.4,'#FFB347'],[0.7,'#40916C'],[1,'#06A77D']],
                       title="Rainfall vs Average Yield by State",labels={RAINFALL_COL:'Avg Rainfall (mm)',TARGET_COL:'Avg Yield (t/ha)'})
        T(fig,430); st.plotly_chart(fig,use_container_width=True)
    with tab4:
        piv=(df_f.groupby([STATE_COL,SEASON_COL])[TARGET_COL].mean().reset_index().pivot(index=STATE_COL,columns=SEASON_COL,values=TARGET_COL).fillna(0))
        fig=px.imshow(piv,color_continuous_scale=[[0,'#F8FAFC'],[0.3,'#E2E8F0'],[0.7,'#40916C'],[1,'#1B4332']],title="State × Season Yield Heatmap",aspect="auto")
        fig.update_traces(hovertemplate='State: <b>%{y}</b><br>Season: <b>%{x}</b><br>Avg Yield: <b>%{z:.2f} t/ha</b><extra></extra>')
        T(fig,520); st.plotly_chart(fig,use_container_width=True)
    st.markdown('<div class="sh"><div class="sh-line"></div><span class="sh-tag">Raw Data Preview</span><div class="sh-line"></div></div>',unsafe_allow_html=True)
    st.dataframe(df_f.head(100),use_container_width=True)
    st.download_button("⬇️  Download Filtered Data",df_f.to_csv(index=False).encode(),"agrisense_filtered.csv","text/csv",use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE 7 — ABOUT  ✅ FIXED (no hardcoded 'R2' column reference)
# ══════════════════════════════════════════════════════════════
elif page == "ℹ️  About":
    st.markdown("""<div class="pg-eye">ℹ️ Project Info</div>
    <h1 class="pg-title">About AgriSense</h1>
    <p class="pg-sub">An end-to-end ML system for agricultural yield intelligence across India.</p>""", unsafe_allow_html=True)
    st.markdown("---")
    col1,col2=st.columns([3,2],gap="large")
    with col1:
        st.markdown("""<div class="about-card"><div class="about-title">🔬 Methodology</div>
        <div class="about-step"><div class="about-step-tag">01 · Climate Clustering</div><div class="about-step-desc">States grouped into 4 zones using K-Means on annual rainfall. Zones range from Arid (&lt;700mm) to Humid (&gt;3500mm).</div></div>
        <div class="about-step"><div class="about-step-tag">02 · Feature Engineering</div><div class="about-step-desc">Log transforms on skewed columns. Derived ratios: Fertilizer/Area, Pesticide/Area. Year trend & climate cluster as inputs.</div></div>
        <div class="about-step"><div class="about-step-tag">03 · Model Training</div><div class="about-step-desc">Temporal split (Train ≤2015, Test &gt;2015). 5-Fold CV. Best: Random Forest R² = 0.9917.</div></div>
        <div class="about-step"><div class="about-step-tag">04 · Evaluation</div><div class="about-step-desc">R², RMSE (log + real), MAE evaluated per model, season, climate cluster & state.</div></div>
        </div>""",unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="about-card"><div class="about-title">📦 Tech Stack</div>
        <table class="stk"><tr><th>Tool</th><th>Purpose</th></tr>
        <tr><td>Python 3.x</td><td>Core language</td></tr><tr><td>Pandas / NumPy</td><td>Data processing</td></tr>
        <tr><td>Scikit-learn</td><td>ML models</td></tr><tr><td>XGBoost / LightGBM</td><td>Boosting</td></tr>
        <tr><td>Plotly</td><td>Interactive charts</td></tr><tr><td>Streamlit</td><td>Web dashboard</td></tr>
        <tr><td>Joblib</td><td>Model persistence</td></tr></table></div>
        <div class="about-card" style="background:rgba(64,145,108,0.05);"><div class="about-title">📊 Dataset Facts</div>
        <div class="fact"><span class="fact-lbl">Period</span><span class="fact-val">1997–2020</span></div>
        <div class="fact"><span class="fact-lbl">Records</span><span class="fact-val">19,689</span></div>
        <div class="fact"><span class="fact-lbl">Unique Crops</span><span class="fact-val">55</span></div>
        <div class="fact"><span class="fact-lbl">States</span><span class="fact-val">30</span></div>
        <div class="fact" style="border:none"><span class="fact-lbl">Seasons</span><span class="fact-val">6</span></div>
        </div>""",unsafe_allow_html=True)

    # ── Crop image gallery ─────────────────────────────────────
    st.markdown('<div class="sh"><div class="sh-line"></div><span class="sh-tag">🌿 In The Field</span><div class="sh-line"></div></div>', unsafe_allow_html=True)
    g1, g2, g3 = st.columns(3, gap='medium')
    gallery = [
        (g1, IMG_MAIZE,  'Maize / Corn',   'One of India\'s top Kharif cereals'),
        (g2, IMG_CARROT, 'Root Crops',      'Harvested across moderate & humid zones'),
        (g3, IMG_CROP,   'Vegetables',      'High-value horticulture driving rural income'),
    ]
    for col, img, title, desc in gallery:
        with col:
            if img:
                st.markdown(f"""
                <div style='border-radius:16px;overflow:hidden;
                            box-shadow:0 6px 24px rgba(15,61,44,0.14);margin-bottom:.6rem;'>
                    <img src='data:image/avif;base64,{img}'
                         style='width:100%;height:180px;object-fit:cover;display:block;
                                filter:brightness(0.88) saturate(1.1);'>
                </div>
                <div style='font-family:Syne,sans-serif;font-weight:700;
                            color:#0F3D2C;font-size:.95rem;margin-bottom:.25rem;'>{title}</div>
                <div style='font-size:.82rem;color:#5A7A6A;line-height:1.6;'>{desc}</div>
                """, unsafe_allow_html=True)

    # ── Model Leaderboard — safe version ────────────────────────
    results_path = os.path.join(OUTPUT_DIR, 'model_results.csv')
    if os.path.exists(results_path):
        try:
            res = pd.read_csv(results_path, index_col=0)
            r2_col, _, _, r2_was_computed = resolve_cols(res)
            r2_label = "R² Score"

            st.markdown('<div class="sh"><div class="sh-line"></div><span class="sh-tag">Model Leaderboard</span><div class="sh-line"></div></div>',unsafe_allow_html=True)

            _lead_df   = res.sort_values(r2_col, ascending=True).reset_index()
            _lead_mcol = _lead_df.columns[0]   # model name column
            fig = px.bar(
                _lead_df,
                x=r2_col, y=_lead_mcol, orientation='h',
                color=r2_col,
                color_continuous_scale=[[0,'#1B4332'],[0.5,'#40916C'],[1,'#52B788']],
                title=f"Model {r2_label} Leaderboard",
                labels={_lead_mcol:'Model', r2_col: r2_label}
            )
            fig.update_traces(marker_line_width=0,
                              hovertemplate=f'<b>%{{y}}</b><br>{r2_label}: %{{x:.4f}}<extra></extra>')
            T(fig, 300)
            st.plotly_chart(fig, use_container_width=True)
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
