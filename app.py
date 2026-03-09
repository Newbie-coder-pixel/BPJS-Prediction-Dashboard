import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import warnings, io, hashlib, re
from datetime import datetime
warnings.filterwarnings('ignore')

import streamlit as st

def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        html,body,[class*="css"]{font-family:'Inter',sans-serif;background:#0a110d;}
        .login-wrap{display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:80vh;gap:20px;}
        .login-logo{font-size:2.8rem;font-weight:700;color:#4ade80;font-family:'Inter',sans-serif;letter-spacing:-0.5px;}
        .login-sub{color:#5a8a68;font-size:.82rem;letter-spacing:1.5px;text-transform:uppercase;font-family:'Inter',sans-serif;}
        </style>
        <div class="login-wrap">
          <div class="login-logo">🟢 BPJS ML</div>
          <div class="login-sub">Prediction Intelligence Dashboard</div>
        </div>
        """, unsafe_allow_html=True)
        password = st.text_input("Masukkan password:", type="password", label_visibility="collapsed",
                                 placeholder="🔑  Masukkan password…")
        if st.button("Masuk →", type="primary", use_container_width=True):
            if password == "bpjs2026":
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("❌  Password salah. Coba lagi.")
        st.stop()

check_password()

# XGBoost (optional)
try:
    from xgboost import XGBRegressor
    XGBOOST_OK = True
except ImportError:
    XGBOOST_OK = False

# LightGBM (optional)
try:
    from lightgbm import LGBMRegressor
    LGBM_OK = True
except ImportError:
    LGBM_OK = False

# Prophet (optional)
try:
    from prophet import Prophet
    PROPHET_OK = True
except ImportError:
    try:
        from fbprophet import Prophet
        PROPHET_OK = True
    except ImportError:
        PROPHET_OK = False

st.set_page_config(page_title="BPJS ML Dashboard", layout="wide", page_icon="⬡", initial_sidebar_state="expanded")

st.markdown("""
<script>
(function keepAlive() {
  setInterval(function() {
    fetch(window.location.href, {method: 'GET', cache: 'no-cache'})
      .catch(function() {});
  }, 600000);
})();
</script>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL DESIGN SYSTEM — Refined Dark Precision Theme
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Font: Inter (Google Fonts) — clean, readable, professional ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ══════════════════════════════════════════════
   WARNA DASAR — BPJS Ketenagakerjaan
   Hijau utama  : #00a550  (logo BPJS)
   Hijau gelap  : #007a3d
   Hijau muda   : #e8f5ee
   Background   : #0a110d  (gelap kehijauan)
══════════════════════════════════════════════ */

*{box-sizing:border-box;margin:0;padding:0;}
html,body,[class*="css"]{
  font-family:'Inter',sans-serif;
  background:#0a110d !important;
  color:#d1e8d8;
  font-size:14px;
  line-height:1.6;
}
.stApp{background:#0a110d;}

/* ── Scrollbar ── */
::-webkit-scrollbar{width:5px;height:5px;}
::-webkit-scrollbar-track{background:#0d160f;}
::-webkit-scrollbar-thumb{background:#1a3d22;border-radius:4px;}
::-webkit-scrollbar-thumb:hover{background:#00a550;}

/* ══════════════════════════════════════════════
   SIDEBAR
══════════════════════════════════════════════ */
section[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#0a110d 0%,#0d1610 100%) !important;
  border-right:1px solid #1a3d22 !important;
}
section[data-testid="stSidebar"] > div{padding-top:0 !important;}
section[data-testid="stSidebar"] .stMarkdown{color:#5a8a68;}
section[data-testid="stSidebar"] label{color:#5a8a68 !important;font-size:.82rem !important;font-family:'Inter',sans-serif;}
section[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMin"],
section[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMax"]{color:#2d5c3a;}
section[data-testid="stSidebar"] .stSlider [data-testid="stSliderThumbValue"]{
  background:#1a3d22;color:#4ade80;font-size:.72rem;border-radius:4px;padding:2px 6px;
}
section[data-testid="stSidebar"] .stButton>button{
  background:transparent;border:1px solid #1a3d22;color:#5a8a68;
  border-radius:8px;font-size:.82rem;padding:6px 10px;
  transition:all .2s;font-family:'Inter',sans-serif;
}
section[data-testid="stSidebar"] .stButton>button:hover{
  border-color:#00a550;color:#4ade80;background:#0f1f14;
}

/* ══════════════════════════════════════════════
   KPI CARDS
══════════════════════════════════════════════ */
.kpi{
  background:#0f1f14;
  border:1px solid #1a3d22;border-radius:14px;
  padding:20px 14px 16px;text-align:center;
  position:relative;overflow:hidden;
  transition:transform .2s ease, box-shadow .2s ease;
  cursor:default;min-height:120px;
  display:flex;flex-direction:column;align-items:center;justify-content:center;
}
.kpi:hover{
  transform:translateY(-3px);
  box-shadow:0 8px 32px rgba(0,165,80,.15), 0 0 0 1px #00a550;
}
.kpi::before{
  content:'';position:absolute;top:0;left:0;right:0;height:3px;
  background:#00a550;
}
.kpi .val{
  font-size:clamp(1rem, 2.8vw, 1.8rem);
  font-weight:700;
  font-family:'Inter',sans-serif;
  color:#4ade80;
  line-height:1.2;
  word-break:break-word;overflow-wrap:break-word;
  white-space:normal;
}
.kpi .val-tooltip{cursor:help;}
.kpi .lbl{
  font-size:.72rem;color:#5a8a68;text-transform:uppercase;
  letter-spacing:1px;margin-top:8px;font-weight:600;
  font-family:'Inter',sans-serif;
}
.kpi .delta{font-size:.75rem;margin-top:6px;font-weight:500;font-family:'Inter',sans-serif;}
.delta-pos{color:#4ade80;} .delta-neg{color:#f87171;} .delta-neu{color:#5a8a68;}

/* ══════════════════════════════════════════════
   INSIGHT CARDS
══════════════════════════════════════════════ */
.insight-card{
  background:#0f1f14;
  border:1px solid #1a3d22;border-radius:12px;padding:20px 18px;
  transition:border-color .2s ease;
}
.insight-card:hover{border-color:#00a550;}
.insight-card .ic-title{
  font-size:.72rem;color:#5a8a68;text-transform:uppercase;
  letter-spacing:1px;font-weight:600;margin-bottom:8px;
  font-family:'Inter',sans-serif;
}
.insight-card .ic-val{
  font-size:1.4rem;font-weight:700;color:#d1e8d8;
  font-family:'Inter',sans-serif;
}
.insight-card .ic-sub{font-size:.8rem;color:#5a8a68;margin-top:4px;font-family:'Inter',sans-serif;}

/* ══════════════════════════════════════════════
   SECTION HEADERS
══════════════════════════════════════════════ */
.sec{
  font-size:.72rem;font-weight:600;color:#5a8a68;
  text-transform:uppercase;letter-spacing:1.5px;
  margin:28px 0 12px;padding-bottom:8px;
  border-bottom:1px solid #1a3d22;
  font-family:'Inter',sans-serif;
  display:flex;align-items:center;gap:8px;
}
.sec::before{content:'';display:inline-block;width:14px;height:2px;background:#00a550;border-radius:2px;}

/* ══════════════════════════════════════════════
   BADGE / INFO BOXES
══════════════════════════════════════════════ */
.badge{
  background:#0f1f14;
  border:1px solid #1a3d22;border-radius:10px;
  padding:14px 18px;margin:12px 0;font-size:.85rem;line-height:1.8;
  border-left:3px solid #00a550;color:#d1e8d8;
  font-family:'Inter',sans-serif;
}
.badge b{color:#4ade80;}

.warn{
  background:#1a1200;
  border:1px solid #3d3000;border-radius:10px;
  padding:14px 18px;color:#fcd34d;font-size:.84rem;margin:10px 0;
  border-left:3px solid #f59e0b;line-height:1.7;
  font-family:'Inter',sans-serif;
}
.info-box{
  background:#0f1f14;
  border:1px solid #1a3d22;border-radius:10px;
  padding:14px 18px;font-size:.84rem;color:#86efac;
  margin:10px 0;line-height:1.8;border-left:3px solid #00a550;
  font-family:'Inter',sans-serif;
}
.success-box{
  background:#0a1f10;
  border:1px solid #1a3d22;border-radius:10px;
  padding:14px 18px;font-size:.84rem;color:#86efac;
  margin:10px 0;border-left:3px solid #22c55e;line-height:1.8;
  font-family:'Inter',sans-serif;
}

/* ══════════════════════════════════════════════
   TAGS
══════════════════════════════════════════════ */
.tag-add{display:inline-block;background:#0f2e16;color:#4ade80;
  border:1px solid #1a5c28;border-radius:5px;padding:2px 9px;
  font-size:.76rem;margin:2px;font-weight:600;font-family:'Inter',sans-serif;}
.tag-rem{display:inline-block;background:#2e0f0f;color:#fca5a5;
  border:1px solid #5c1a1a;border-radius:5px;padding:2px 9px;
  font-size:.76rem;margin:2px;font-weight:600;font-family:'Inter',sans-serif;}
.tag-stable{display:inline-block;background:#0f1f14;color:#86efac;
  border:1px solid #1a3d22;border-radius:5px;padding:2px 9px;
  font-size:.76rem;margin:2px;font-weight:600;font-family:'Inter',sans-serif;}

/* ══════════════════════════════════════════════
   MODEL PILLS
══════════════════════════════════════════════ */
.mpill{display:inline-block;padding:2px 10px;border-radius:20px;font-size:.75rem;font-weight:600;margin:2px;font-family:'Inter',sans-serif;}
.mpill-green{background:#0f2e16;color:#4ade80;border:1px solid #1a5c28;}
.mpill-blue{background:#0f1f2e;color:#60a5fa;border:1px solid #1a3a5f;}
.mpill-yellow{background:#2e1f00;color:#fbbf24;border:1px solid #5c3d00;}
.mpill-red{background:#2e0f0f;color:#f87171;border:1px solid #5c1a1a;}

/* ══════════════════════════════════════════════
   TABS
══════════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"]{
  gap:2px;background:#0d160f;border-radius:10px;padding:4px;
  border:1px solid #1a3d22;
}
.stTabs [data-baseweb="tab"]{
  border-radius:8px;padding:9px 20px;font-size:.84rem;font-weight:500;
  color:#5a8a68;font-family:'Inter',sans-serif;
  transition:all .2s ease;
}
.stTabs [aria-selected="true"]{
  background:#1a3d22 !important;
  color:#4ade80 !important;
  box-shadow:0 0 0 1px #00a550 !important;
}
.stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]){
  color:#86efac;background:#0f1f14;
}

/* ══════════════════════════════════════════════
   DATAFRAME
══════════════════════════════════════════════ */
.stDataFrame{border-radius:10px;overflow:hidden;border:1px solid #1a3d22;}
.stDataFrame [data-testid="stDataFrameResizable"]{background:#0d160f;}

/* ══════════════════════════════════════════════
   BUTTONS
══════════════════════════════════════════════ */
.stButton>button{
  border-radius:8px;font-family:'Inter',sans-serif;
  font-weight:600;font-size:.85rem;
  transition:all .2s ease;
}
.stButton>button[kind="primary"]{
  background:#00a550 !important;
  border:none !important;color:#fff !important;
  box-shadow:0 2px 12px rgba(0,165,80,.4);
}
.stButton>button[kind="primary"]:hover{
  background:#007a3d !important;
  box-shadow:0 4px 20px rgba(0,165,80,.55) !important;
}

/* ══════════════════════════════════════════════
   SELECT / INPUT
══════════════════════════════════════════════ */
.stSelectbox [data-baseweb="select"]{border-radius:8px !important;}
.stSelectbox [data-baseweb="select"] > div{
  background:#0f1f14 !important;border-color:#1a3d22 !important;
  color:#d1e8d8 !important;font-family:'Inter',sans-serif;font-size:.88rem;
}
.stTextInput input{
  background:#0f1f14 !important;border:1px solid #1a3d22 !important;
  border-radius:8px !important;color:#d1e8d8 !important;
  font-family:'Inter',sans-serif !important;font-size:.88rem;
}
.stTextInput input:focus{border-color:#00a550 !important;box-shadow:0 0 0 2px rgba(0,165,80,.2) !important;}

/* ══════════════════════════════════════════════
   FILE UPLOADER
══════════════════════════════════════════════ */
[data-testid="stFileUploader"]{
  background:#0d160f;border:1px dashed #1a3d22;border-radius:10px;padding:8px;
}
[data-testid="stFileUploader"]:hover{border-color:#00a550;}
[data-testid="stFileUploadDropzone"]{background:transparent !important;}

/* ══════════════════════════════════════════════
   EXPANDER
══════════════════════════════════════════════ */
[data-testid="stExpander"]{
  border:1px solid #1a3d22;border-radius:10px;overflow:hidden;
  background:#0d160f;
}
[data-testid="stExpander"] summary{
  background:#0d160f;padding:12px 16px;
  color:#5a8a68;font-size:.85rem;font-family:'Inter',sans-serif;
}
[data-testid="stExpander"] summary:hover{color:#86efac;}

/* ══════════════════════════════════════════════
   METRIC
══════════════════════════════════════════════ */
[data-testid="stMetric"]{
  background:#0f1f14;border:1px solid #1a3d22;border-radius:10px;padding:16px;
}
[data-testid="stMetricValue"]{
  color:#4ade80;font-family:'Inter',sans-serif;font-weight:700;
}
[data-testid="stMetricLabel"]{color:#5a8a68;font-size:.76rem;letter-spacing:.5px;}

/* ══════════════════════════════════════════════
   ALERT
══════════════════════════════════════════════ */
[data-testid="stAlert"]{border-radius:10px;border:none;}

/* ══════════════════════════════════════════════
   DOWNLOAD BUTTON
══════════════════════════════════════════════ */
[data-testid="stDownloadButton"] button{
  background:#007a3d !important;
  border:1px solid #00a550 !important;color:#fff !important;
  border-radius:8px;font-family:'Inter',sans-serif;font-weight:600;
}
[data-testid="stDownloadButton"] button:hover{
  background:#005c2d !important;
  box-shadow:0 4px 16px rgba(0,165,80,.35) !important;
}

/* ══════════════════════════════════════════════
   SPINNER
══════════════════════════════════════════════ */
[data-testid="stSpinner"]>div{
  border-color:#1a3d22 #1a3d22 #00a550 !important;
}

/* ══════════════════════════════════════════════
   HERO HEADER
══════════════════════════════════════════════ */
.hero-wrap{
  padding:24px 28px;
  background:#0f1f14;
  border:1px solid #1a3d22;border-radius:14px;
  border-left:4px solid #00a550;
  position:relative;overflow:hidden;margin-bottom:8px;
}
.hero-logo{
  font-size:1.5rem;font-weight:700;font-family:'Inter',sans-serif;
  color:#4ade80;line-height:1.2;margin-bottom:4px;
}
.hero-sub{font-size:.8rem;color:#5a8a68;font-family:'Inter',sans-serif;letter-spacing:.3px;}

/* ══════════════════════════════════════════════
   SIDEBAR BRAND
══════════════════════════════════════════════ */
.sb-brand{
  padding:24px 18px 18px;
  border-bottom:1px solid #1a3d22;margin-bottom:14px;
}
.sb-logo{
  font-size:1.25rem;font-weight:700;font-family:'Inter',sans-serif;
  color:#4ade80;
}
.sb-sub{font-size:.72rem;color:#5a8a68;margin-top:3px;font-family:'Inter',sans-serif;}

/* ══════════════════════════════════════════════
   CONCLUSION CARDS
══════════════════════════════════════════════ */
.concl-card{
  background:#0f1f14;
  border:1px solid #1a3d22;border-radius:10px;
  padding:18px 20px;margin:8px 0;border-left:3px solid #00a550;
}
.concl-card .ct{font-size:.72rem;color:#5a8a68;text-transform:uppercase;letter-spacing:1px;font-weight:600;margin-bottom:6px;font-family:'Inter',sans-serif;}
.concl-card .cv{color:#d1e8d8;font-size:.88rem;line-height:1.8;}

/* ══════════════════════════════════════════════
   PROGRAM MODEL CARDS
══════════════════════════════════════════════ */
.prog-card{
  background:#0f1f14;border:1px solid #1a3d22;border-radius:10px;
  padding:10px 14px;min-width:120px;display:inline-block;
}
.prog-card .pc-name{font-size:.72rem;color:#5a8a68;text-transform:uppercase;letter-spacing:.8px;font-weight:600;margin-bottom:4px;font-family:'Inter',sans-serif;}
.prog-card .pc-model{font-size:.88rem;font-weight:700;color:#d1e8d8;margin-bottom:4px;font-family:'Inter',sans-serif;}

/* ══════════════════════════════════════════════
   HOLIDAY EFFECT CARDS
══════════════════════════════════════════════ */
.hol-card{border-radius:10px;padding:16px 18px;transition:transform .2s ease;}
.hol-card:hover{transform:translateY(-2px);}

/* ══════════════════════════════════════════════
   EMPTY STATE
══════════════════════════════════════════════ */
.empty-state{text-align:center;padding:70px 0 50px;}
.empty-icon{font-size:3.5rem;margin-bottom:16px;}
.empty-title{
  font-size:1.8rem;font-weight:700;font-family:'Inter',sans-serif;
  color:#4ade80;margin-bottom:12px;
}
.empty-sub{color:#5a8a68;max-width:500px;margin:auto;font-size:.9rem;line-height:1.8;font-family:'Inter',sans-serif;}

/* ══════════════════════════════════════════════
   FEATURE CARDS (landing)
══════════════════════════════════════════════ */
.feat-card{
  background:#0f1f14;
  border:1px solid #1a3d22;border-radius:12px;padding:22px 18px;
  text-align:center;height:180px;position:relative;overflow:hidden;
  transition:border-color .2s ease, transform .2s ease;
  border-top:3px solid #00a550;
}
.feat-card:hover{transform:translateY(-4px);border-color:#00a550;}
.feat-icon{font-size:1.8rem;margin-bottom:8px;}
.feat-title{font-weight:700;color:#d1e8d8;margin-bottom:6px;font-size:.92rem;font-family:'Inter',sans-serif;}
.feat-desc{color:#5a8a68;font-size:.8rem;line-height:1.6;font-family:'Inter',sans-serif;}

/* ══════════════════════════════════════════════
   EXPORT STATUS BOX
══════════════════════════════════════════════ */
.export-box{
  background:#0d160f;border:1px solid #1a3d22;border-radius:10px;
  padding:16px 18px;line-height:2;font-size:.84rem;font-family:'Inter',sans-serif;
}
.export-box .ok{color:#4ade80;} .export-box .nok{color:#5a8a68;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PERSISTENT HISTORY
# ══════════════════════════════════════════════════════════════════════════════
import json, os, pickle

HISTORY_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.bpjs_history')
HISTORY_META = os.path.join(HISTORY_DIR, 'history_meta.json')
os.makedirs(HISTORY_DIR, exist_ok=True)

def _hpath(eid):
    return os.path.join(HISTORY_DIR, f'{eid}.pkl')

def load_history_meta():
    if not os.path.exists(HISTORY_META):
        return []
    try:
        with open(HISTORY_META, 'r') as f:
            return json.load(f)
    except:
        return []

def save_history_meta(meta_list):
    with open(HISTORY_META, 'w') as f:
        json.dump(meta_list, f, ensure_ascii=False, indent=2)

def save_history_entry(eid, df, results, extra=None):
    try:
        with open(_hpath(eid), 'wb') as f:
            pickle.dump({'df': df, 'results': results, 'extra': extra or {}}, f)
        return True
    except:
        return False

def load_history_entry(eid):
    p = _hpath(eid)
    if not os.path.exists(p):
        return None, {}, {}
    try:
        with open(p, 'rb') as f:
            d = pickle.load(f)
        return d.get('df'), d.get('results', {}), d.get('extra', {})
    except:
        return None, {}, {}

def delete_history_entry(eid):
    p = _hpath(eid)
    if os.path.exists(p):
        os.remove(p)

def add_to_history(label, eid, df, results, extra=None):
    meta = load_history_meta()
    meta = [m for m in meta if m['id'] != eid]
    meta.append({'id': eid, 'label': label, 'timestamp': datetime.now().isoformat()})
    if len(meta) > 20:
        for m in meta[:-20]:
            delete_history_entry(m['id'])
        meta = meta[-20:]
    save_history_meta(meta)
    save_history_entry(eid, df, results, extra)

for k, v in [('active_data', None), ('active_results', {}), ('active_entry_id', None), ('history_loaded', False)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
# CHART DESIGN SYSTEM
# ══════════════════════════════════════════════════════════════════════════════
DARK = dict(
    template='plotly_dark',
    paper_bgcolor='rgba(10,17,13,0)',
    plot_bgcolor='rgba(13,22,15,0)',
    font_color='#5a8a68',
    font_family='Inter',
)

# Palet warna — hijau BPJS sebagai warna utama
COLORS = [
    '#00a550',  # hijau BPJS utama
    '#4ade80',  # hijau muda
    '#fbbf24',  # kuning (aksen)
    '#34d399',  # emerald
    '#86efac',  # hijau pastel
    '#a3e635',  # lime
    '#fb923c',  # oranye (kontras)
    '#f472b6',  # pink (kontras)
    '#38bdf8',  # biru (kontras)
    '#f87171',  # merah (kontras)
]

def hex_to_rgba(hex_c, alpha=1.0):
    h = hex_c.lstrip('#')
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f'rgba({r},{g},{b},{alpha})'

def styled_chart(fig, height=420, legend_bottom=True, margin_b=90, title=None):
    """Apply unified green BPJS chart styling."""
    grid_color  = 'rgba(26,61,34,0.7)'
    grid_color2 = 'rgba(15,31,20,0.8)'

    updates = dict(
        **DARK,
        height=height,
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='rgba(10,17,13,.96)',
            font_size=12,
            font_family='Inter',
            bordercolor='#1a3d22',
        ),
        legend=dict(
            orientation='h', y=-0.22,
            font=dict(size=11, family='Inter'),
            groupclick='toggleitem',
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(0,0,0,0)',
        ) if legend_bottom else dict(
            font=dict(size=11, family='Inter'),
            bgcolor='rgba(0,0,0,0)',
        ),
        margin=dict(b=margin_b if legend_bottom else 40, t=30 if not title else 50, l=60, r=24),
        xaxis=dict(
            showgrid=True, gridcolor=grid_color, gridwidth=1,
            zeroline=False, linecolor='#1a3d22', linewidth=1,
            tickfont=dict(size=11, color='#5a8a68', family='Inter'),
        ),
        yaxis=dict(
            showgrid=True, gridcolor=grid_color2, gridwidth=1,
            zeroline=False, linecolor='#1a3d22', linewidth=1,
            tickfont=dict(size=11, color='#5a8a68', family='Inter'),
        ),
    )
    if title:
        updates['title'] = dict(
            text=title,
            font=dict(size=13, color='#5a8a68', family='Inter'),
            x=0, pad=dict(b=10)
        )

    fig.update_layout(**updates)
    fig.update_xaxes(gridcolor=grid_color, zerolinecolor='#1a3d22')
    fig.update_yaxes(gridcolor=grid_color2, zerolinecolor='#1a3d22')
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# DATA PARSING (unchanged logic, same as original)
# ══════════════════════════════════════════════════════════════════════════════

def clean_num(val):
    s = re.sub(r'[^\d\.\-]', '', str(val).replace(',', ''))
    try: return float(s)
    except: return np.nan

@st.cache_data(show_spinner=False)
def load_raw(fb, fname):
    try:
        df = pd.read_csv(io.BytesIO(fb)) if fname.lower().endswith('.csv') \
             else pd.read_excel(io.BytesIO(fb))
        df.columns = df.columns.str.strip().str.upper()
        return df
    except:
        return None

def parse_dataset(df, year_hint):
    cols = df.columns.tolist()
    prog_col = None
    for c in cols:
        if c in ('PROGRAM', 'KATEGORI', 'CATEGORY', 'JENIS', 'JAMINAN'):
            prog_col = c; break
    if prog_col is None:
        for c in cols:
            if df[c].dtype == object and df[c].nunique() < 50:
                if not any(k in c for k in ('DATE','TANGGAL','KODE','ID','PERIODE')):
                    prog_col = c; break
    if prog_col is None:
        return None, f"Kolom PROGRAM tidak ditemukan. Kolom: {cols}"

    kasus_col = None
    for candidate in ('AKTUAL_KASUS', 'KASUS', 'CASE', 'KLAIM', 'COUNT'):
        if candidate in cols:
            kasus_col = candidate; break
    if kasus_col is None:
        return None, f"Kolom KASUS tidak ditemukan. Kolom: {cols}"

    nominal_col = None
    for candidate in ('AKTUAL_NOMINAL', 'NOMINAL', 'AMOUNT', 'NILAI', 'MANFAAT', 'BEBAN'):
        if candidate in cols:
            nominal_col = candidate; break

    date_col = None
    for candidate in ('DATE', 'PERIODE', 'TANGGAL', 'BULAN', 'MONTH', 'PERIOD'):
        if candidate in cols:
            date_col = candidate; break

    df = df.copy()
    df['Tahun'] = year_hint
    df['Bulan'] = 12

    if date_col:
        raw_date = df[date_col].astype(str).str.strip()
        ym = raw_date.str.extract(r'(\d{4})[-/]?(\d{2})', expand=True)
        has_ym = ym[0].notna() & ym[1].notna()
        if has_ym.mean() > 0.5:
            df.loc[has_ym, 'Tahun'] = ym[0][has_ym].astype(int)
            df.loc[has_ym, 'Bulan'] = ym[1][has_ym].astype(int)
        else:
            try:
                dt = pd.to_datetime(raw_date, errors='coerce')
                ok = dt.notna()
                if ok.mean() > 0.5:
                    df.loc[ok, 'Tahun'] = dt[ok].dt.year.astype(int)
                    df.loc[ok, 'Bulan'] = dt[ok].dt.month.astype(int)
            except:
                pass

    tmp = pd.DataFrame()
    tmp['Kategori'] = df[prog_col].astype(str).str.strip().str.upper()
    tmp['Kasus']    = df[kasus_col].apply(clean_num)
    if nominal_col:
        tmp['Nominal'] = df[nominal_col].apply(clean_num)
    tmp['Tahun'] = df['Tahun'].astype(int)
    tmp['Bulan'] = df['Bulan'].astype(int)

    tmp = tmp[~tmp['Kategori'].str.lower().isin(['nan','none',''])]
    tmp = tmp.dropna(subset=['Kasus'])
    tmp = tmp[tmp['Kasus'] >= 0]

    rows_per_group = tmp.groupby(['Tahun', 'Kategori']).size()
    is_monthly = rows_per_group.max() > 1

    if is_monthly:
        agg_dict = {'Kasus': 'sum'}
        if 'Nominal' in tmp.columns:
            agg_dict['Nominal'] = 'sum'
        yearly = tmp.groupby(['Tahun', 'Kategori'], as_index=False).agg(agg_dict)
    else:
        yearly = tmp.drop(columns=['Bulan'])

    return yearly, None

def _detect_cols_quick(df):
    cols = df.columns.tolist()
    prog_col, kasus_col, nominal_col, date_col = None, None, None, None
    for c in cols:
        if c in ('PROGRAM','KATEGORI','CATEGORY','JENIS','JAMINAN') and prog_col is None:
            prog_col = c
        if c in ('AKTUAL_KASUS','KASUS','CASE','KLAIM') and kasus_col is None:
            kasus_col = c
        if c in ('AKTUAL_NOMINAL','NOMINAL','AMOUNT','NILAI','MANFAAT','BEBAN') and nominal_col is None:
            nominal_col = c
        if c in ('DATE','PERIODE','TANGGAL','BULAN','MONTH') and date_col is None:
            date_col = c
    if prog_col and kasus_col:
        return {'prog': prog_col, 'kasus': kasus_col, 'nominal': nominal_col, 'date': date_col}
    return None

def _build_raw_monthly(df, year_hint, m):
    try:
        tmp = pd.DataFrame()
        tmp['Kategori'] = df[m['prog']].astype(str).str.strip().str.upper()
        tmp['Kasus']    = df[m['kasus']].apply(clean_num)
        if m['nominal']:
            tmp['Nominal'] = df[m['nominal']].apply(clean_num)
        tmp['Tahun'] = year_hint
        tmp['Bulan'] = 12
        if m['date']:
            raw_d = df[m['date']].astype(str).str.strip()
            ym = raw_d.str.extract(r'(\d{4})[-/]?(\d{2})', expand=True)
            ok = ym[0].notna() & ym[1].notna()
            if ok.mean() > 0.5:
                tmp.loc[ok, 'Tahun'] = ym[0][ok].astype(int)
                tmp.loc[ok, 'Bulan'] = ym[1][ok].astype(int)
        tmp = tmp[~tmp['Kategori'].str.lower().isin(['nan','none',''])]
        tmp = tmp.dropna(subset=['Kasus'])
        tmp = tmp[tmp['Kasus'] >= 0]
        tmp['Tahun'] = tmp['Tahun'].astype(int)
        tmp['Bulan'] = tmp['Bulan'].astype(int)
        return tmp
    except Exception as e:
        return None

def merge_all(files_info):
    all_dfs, errors = [], []
    for year_hint, raw, fname in files_info:
        parsed, err = parse_dataset(raw, year_hint)
        if err:
            errors.append(f"⚠️ {fname}: {err}")
        elif parsed is not None and len(parsed) > 0:
            all_dfs.append(parsed)
    if not all_dfs:
        return None, errors
    combined = pd.concat(all_dfs, ignore_index=True)
    agg = {'Kasus': 'last'}
    if 'Nominal' in combined.columns:
        agg['Nominal'] = 'last'
    combined = combined.groupby(['Tahun', 'Kategori'], as_index=False).agg(agg)
    return combined, errors

def analyze_program_changes(df):
    years = sorted(df['Tahun'].unique())
    changes = {}
    for i in range(len(years) - 1):
        y0, y1 = years[i], years[i+1]
        p0 = set(df[df['Tahun'] == y0]['Kategori'].unique())
        p1 = set(df[df['Tahun'] == y1]['Kategori'].unique())
        changes[(y0, y1)] = {'added': sorted(p1-p0), 'removed': sorted(p0-p1), 'stable': sorted(p0&p1)}
    return changes

def get_active_programs(df):
    latest = df['Tahun'].max()
    return sorted(df[df['Tahun'] == latest]['Kategori'].unique())


# ══════════════════════════════════════════════════════════════════════════════
# ML CORE (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def score_model(yt, yp):
    yt, yp = np.array(yt, dtype=float), np.array(yp, dtype=float)
    mae  = float(mean_absolute_error(yt, yp))
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    r2   = float(r2_score(yt, yp)) if len(yt) >= 3 else None
    mape = float(np.mean(np.abs((yt - yp) / (np.abs(yt) + 1e-9))) * 100)
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE (%)': mape}

def forecast_holt(history, n_steps, alpha=None, beta=None):
    y = np.array(history, dtype=float)
    n = len(y)
    if n < 2:
        return np.array([y[-1]] * n_steps)
    best_mape = np.inf
    best_a, best_b = 0.3, 0.1
    for a in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        for b in [0.05, 0.1, 0.15, 0.2, 0.3]:
            try:
                lvl = np.zeros(n); trnd = np.zeros(n)
                lvl[0] = y[0]; trnd[0] = y[1]-y[0] if n>1 else 0
                for i in range(1, n):
                    lvl[i]  = a*y[i] + (1-a)*(lvl[i-1]+trnd[i-1])
                    trnd[i] = b*(lvl[i]-lvl[i-1]) + (1-b)*trnd[i-1]
                fitted = lvl[:-1] + trnd[:-1]
                mape = np.mean(np.abs((y[1:]-fitted)/(np.abs(y[1:])+1e-9)))*100
                if mape < best_mape:
                    best_mape=mape; best_a=a; best_b=b
            except:
                pass
    lvl = np.zeros(n); trnd = np.zeros(n)
    lvl[0] = y[0]; trnd[0] = y[1]-y[0] if n>1 else 0
    for i in range(1, n):
        lvl[i]  = best_a*y[i] + (1-best_a)*(lvl[i-1]+trnd[i-1])
        trnd[i] = best_b*(lvl[i]-lvl[i-1]) + (1-best_b)*trnd[i-1]
    preds = np.array([lvl[-1]+(s+1)*trnd[-1] for s in range(n_steps)])
    return preds, best_a, best_b, lvl, trnd

def forecast_ses(history, n_steps):
    y = np.array(history, dtype=float)
    best_mape=np.inf; best_a=0.3
    for a in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        lvl=y[0]; fitted=[]
        for i in range(1,len(y)):
            fitted.append(lvl); lvl=a*y[i]+(1-a)*lvl
        if fitted:
            mape=np.mean(np.abs((y[1:len(fitted)+1]-np.array(fitted))/(np.abs(y[1:len(fitted)+1])+1e-9)))*100
            if mape < best_mape:
                best_mape=mape; best_a=a
    lvl=y[0]
    for v in y[1:]: lvl=best_a*v+(1-best_a)*lvl
    return np.array([lvl]*n_steps), best_a

def forecast_moving_avg(history, n_steps, window=3):
    y = np.array(history, dtype=float)
    w = min(window, len(y))
    weights = np.arange(1, w+1, dtype=float); weights /= weights.sum()
    base = float(np.dot(y[-w:], weights))
    trend = float(y[-1]-y[-2]) if len(y)>=2 else 0.0
    trend = np.clip(trend, -abs(base)*0.3, abs(base)*0.3)
    return np.array([base+trend*(s+1)*0.5 for s in range(n_steps)])

def loo_cv_stat(history, method_fn, n_steps=1):
    y = np.array(history, dtype=float)
    n = len(y)
    yt_all, yp_all = [], []
    for leave in range(1, n):
        train = y[:leave]; actual = y[leave]
        try:
            preds = method_fn(train, n_steps)
            if isinstance(preds, tuple): preds = preds[0]
            yp_all.append(float(preds[0])); yt_all.append(actual)
        except: pass
    if len(yt_all) < 1:
        return {'MAE': np.inf, 'RMSE': np.inf, 'R2': -999, 'MAPE (%)': np.inf}
    return score_model(np.array(yt_all), np.array(yp_all))

def build_features(series, n_lags=1, cat_id=0.0):
    pad = list(series)
    while len(pad) <= n_lags: pad.insert(0, pad[0])
    X_all, y_all = [], []
    for i in range(n_lags, len(pad)):
        lags = [pad[i-l] for l in range(1, n_lags+1)]
        win  = pad[max(0,i-3):i]
        feat = lags + [np.mean(win), np.std(win) if len(win)>1 else 0.0,
                       pad[i-1]-pad[i-2] if i>=2 else 0.0, cat_id]
        X_all.append(feat); y_all.append(pad[i])
    return np.array(X_all), np.array(y_all)

SCALED_MODELS = {'SVR', 'KNN', 'Ridge', 'Lasso', 'ElasticNet', 'Linear Regression', 'Huber'}

def get_ml_models(n_train):
    k = min(3, max(1, n_train-1))
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge':             Ridge(alpha=1.0),
        'Lasso':             Lasso(alpha=0.1, max_iter=10000),
        'ElasticNet':        ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000),
        'Random Forest':     RandomForestRegressor(n_estimators=100, max_depth=4, n_jobs=-1, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42),
        'SVR':               SVR(kernel='rbf', C=100, gamma='scale'),
        'KNN':               KNeighborsRegressor(n_neighbors=k, weights='distance'),
    }
    if XGBOOST_OK:
        models['XGBoost'] = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, n_jobs=-1, random_state=42, verbosity=0)
    if LGBM_OK:
        models['LightGBM'] = LGBMRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, n_jobs=-1, random_state=42, verbose=-1)
    return models

def loo_cv_ml(Xc, yc, model_name, model_obj):
    import copy
    n = len(Xc)
    yt_all, yp_all = [], []
    for leave in range(n):
        idx_tr = [i for i in range(n) if i != leave]
        if not idx_tr: continue
        Xtr, ytr = Xc[idx_tr], yc[idx_tr]
        Xte, yte = Xc[[leave]], yc[[leave]]
        sc = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
        try:
            mdl = copy.deepcopy(model_obj)
            mdl.fit(Xtr_s if model_name in SCALED_MODELS else Xtr, ytr)
            yp = float(mdl.predict(Xte_s if model_name in SCALED_MODELS else Xte)[0])
            yt_all.append(float(yte[0])); yp_all.append(yp)
        except: pass
    if len(yt_all) < 2:
        return {'MAE': np.inf, 'RMSE': np.inf, 'R2': -999, 'MAPE (%)': np.inf}
    return score_model(np.array(yt_all), np.array(yp_all))

def train_best_per_program(df, target, n_lags, test_ratio):
    from collections import Counter
    import copy

    active  = get_active_programs(df)
    cat_enc = {c: float(i) for i, c in enumerate(active)}
    years   = sorted(df['Tahun'].unique())
    single  = len(years) == 1
    per_prog = {}; detail_rows = []

    for cat in active:
        sub = (df[df['Kategori']==cat].sort_values('Tahun')[target].dropna().values.astype(float))
        if len(sub)==0: continue
        if len(sub)<2:
            per_prog[cat] = {'best_name':'Holt Smoothing','method_type':'stat','history':list(sub),'single':True,
                             'metrics':{'R2':None,'MAPE (%)':None,'MAE':None,'RMSE':None},'cat_id':cat_enc.get(cat,0.0)}
            continue
        use_ml = (len(sub) >= 8)
        stat_candidates = {
            'Holt Smoothing': lambda h,s: forecast_holt(h,s),
            'Exp Smoothing':  lambda h,s: (forecast_ses(h,s)[0],),
            'Weighted MA':    lambda h,s: (forecast_moving_avg(h,s),),
        }
        stat_scores = {}
        for mname, fn in stat_candidates.items():
            sc = loo_cv_stat(list(sub), fn)
            stat_scores[mname] = sc
            detail_rows.append({'Program':cat,'Model':mname,**sc})
        ml_scores = {}; ml_models_fitted = {}
        if use_ml:
            Xc, yc = build_features(sub, min(n_lags,2), cat_enc.get(cat,0.0))
            sc_full = StandardScaler().fit(Xc)
            ml_defs = get_ml_models(len(Xc)-1)
            for mname, mdl_obj in ml_defs.items():
                sc = loo_cv_ml(Xc, yc, mname, mdl_obj)
                ml_scores[mname] = sc
                detail_rows.append({'Program':cat,'Model':mname,**sc})
                try:
                    mdl_full = copy.deepcopy(mdl_obj)
                    Xc_s = sc_full.transform(Xc)
                    mdl_full.fit(Xc_s if mname in SCALED_MODELS else Xc, yc)
                    ml_models_fitted[mname] = {'model':mdl_full,'scaler':sc_full,'Xc':Xc,'yc':yc}
                except: pass
        all_scores = {**stat_scores,**ml_scores}
        valid = {m:s for m,s in all_scores.items() if s['MAPE (%)']<200}
        pool  = valid if valid else all_scores
        if not pool: pool = stat_scores
        best_name = min(pool, key=lambda m: pool[m]['MAPE (%)'])
        best_sc   = all_scores[best_name]
        is_stat   = best_name in stat_candidates
        entry = {'best_name':best_name,'method_type':'stat' if is_stat else 'ml','history':list(sub),'single':False,'metrics':best_sc,'all_scores':all_scores,'cat_id':cat_enc.get(cat,0.0)}
        if is_stat: entry['stat_fn_name'] = best_name
        else:
            if best_name in ml_models_fitted:
                info_ml = ml_models_fitted[best_name]
                entry['best_model']=info_ml['model']; entry['scaler']=info_ml['scaler']; entry['n_lags_used']=min(n_lags,2)
        per_prog[cat] = entry

    bpp_rows = []
    for cat, info in per_prog.items():
        m = info.get('metrics',{})
        bpp_rows.append({'Program':cat,'Model':info['best_name'],'Tipe':'📊 Statistik' if info.get('method_type')=='stat' else '🤖 ML',
                         'R2':m.get('R2'),'MAPE (%)':m.get('MAPE (%)'),'MAE':m.get('MAE'),'RMSE':m.get('RMSE')})
    best_per_prog = pd.DataFrame(bpp_rows)
    detail_df     = pd.DataFrame(detail_rows) if detail_rows else pd.DataFrame()

    valid_bpp = [r for r in bpp_rows if r['MAPE (%)'] is not None and r['MAPE (%)']<200]
    avg_r2    = float(np.mean([r['R2'] for r in valid_bpp if r['R2'] is not None])) if valid_bpp else None
    avg_mape  = float(np.mean([r['MAPE (%)'] for r in valid_bpp])) if valid_bpp else 0.0
    avg_mae   = float(np.mean([r['MAE']      for r in valid_bpp])) if valid_bpp else 0.0

    from collections import Counter
    overall_best = Counter(r['Model'] for r in bpp_rows).most_common(1)[0][0] if bpp_rows else 'N/A'

    return {'per_prog':per_prog,'best_per_program':best_per_prog,'detail':detail_df,'results_df':pd.DataFrame(),
            'best_name':overall_best,'best_r2':avg_r2,'best_mape':avg_mape,'best_mae':avg_mae,
            'cat_enc':cat_enc,'single':single,'n_lags':n_lags,'target':target,'active_programs':active}, None

def run_ml(df, target, n_lags, test_ratio):
    return train_best_per_program(df, target, n_lags, test_ratio)

def run_ml_per_program(df, target, n_lags, test_ratio):
    res, err = train_best_per_program(df, target, n_lags, test_ratio)
    return res


# ══════════════════════════════════════════════════════════════════════════════
# CONCLUSION METRICS
# ══════════════════════════════════════════════════════════════════════════════

def build_conclusion(ml_result, per_prog_result, df, target, n_future):
    lines = []
    if ml_result is None: return lines
    bpp = ml_result.get('best_per_program', pd.DataFrame())

    if not bpp.empty and 'R2' in bpp.columns:
        r2_val   = float(bpp['R2'].mean())
        mape_val = float(bpp['MAPE (%)'].mean())
    else:
        r2_val, mape_val = ml_result.get('best_r2',0.0), ml_result.get('best_mape',0.0)

    r2_grade   = ("Sangat Baik (>0.9)" if r2_val>0.9 else "Baik (0.8–0.9)" if r2_val>0.8 else "Cukup (0.6–0.8)" if r2_val>0.6 else "Lemah (<0.6)")
    mape_grade = ("Sangat Akurat (<10%)" if mape_val<10 else "Akurat (10–20%)" if mape_val<20 else "Cukup (20–50%)" if mape_val<50 else "Tidak Akurat (>50%)")

    lines.append(('🎯','Pendekatan Prediksi',
        f"Setiap program menggunakan **model terbaiknya sendiri** (per-program best model). "
        f"Rata-rata R² = **{r2_val:.4f}** ({r2_grade}), rata-rata MAPE = **{mape_val:.2f}%** ({mape_grade})."))

    if not bpp.empty:
        prog_str = ', '.join(f"{r['Program']} → **{r['Model']}**" for _,r in bpp.iterrows())
        lines.append(('📊','Model Terbaik per Program', prog_str))
        if len(bpp)>1:
            worst  = bpp.sort_values('R2').iloc[0]
            best_p = bpp.sort_values('R2',ascending=False).iloc[0]
            lines.append(('🔍','Akurasi per Program',
                f"Program **{best_p['Program']}** paling mudah diprediksi (R²={best_p['R2']:.3f}, MAPE={best_p['MAPE (%)']:.1f}%). "
                f"Program **{worst['Program']}** paling sulit (R²={worst['R2']:.3f}, MAPE={worst['MAPE (%)']:.1f}%) — "
                "pertimbangkan menambah data historis atau fitur eksternal."))

    base_yr = int(df['Tahun'].max())
    lines.append(('📅','Horizon Prediksi',
        f"Model dilatih pada data s/d **{base_yr}** dan mampu memproyeksikan hingga **{base_yr+n_future}** ({n_future} tahun ke depan). "
        "Akurasi menurun semakin jauh horizon waktu — gunakan prediksi jangka pendek untuk keputusan kritis."))

    yrs = sorted(df['Tahun'].unique())
    lines.append(('📁','Kualitas Data',
        f"Dataset mencakup **{len(yrs)} tahun** ({yrs[0]}–{yrs[-1]}) dengan **{len(get_active_programs(df))} program aktif**. "
        + ("✅ Jumlah tahun cukup untuk model lag." if len(yrs)>=4 else "⚠️ Tambah data historis untuk meningkatkan akurasi model.")))

    if r2_val>=0.8 and mape_val<=20:
        rec = "✅ Model layak digunakan untuk perencanaan anggaran dan proyeksi klaim BPJS."
    elif r2_val>=0.6:
        rec = "⚠️ Model cukup untuk proyeksi kasar. Validasi manual disarankan sebelum keputusan strategis."
    else:
        rec = "❌ Akurasi belum optimal. Tambah data historis minimal 5 tahun, atau gunakan Prophet untuk data bulanan."
    lines.append(('💡','Rekomendasi', rec))
    return lines


# ══════════════════════════════════════════════════════════════════════════════
# INDONESIAN HOLIDAYS — Google Calendar API
# ══════════════════════════════════════════════════════════════════════════════

GCAL_CALENDAR_ID = "en.indonesian%23holiday%40group.v.calendar.google.com"

_WINDOW_RULES = {
    'idul fitri': (-7,7), 'lebaran': (-7,7), 'ramadan': (0,29), 'ramadhan': (0,29),
    'puasa': (0,29), 'idul adha': (-3,3), 'natal': (-2,2), 'christmas': (-2,2),
    'tahun baru': (-1,2), 'new year': (-1,2), 'cuti bersama': (-1,1), 'default': (-1,1),
}

def _get_window(name: str):
    nl = name.lower()
    for keyword, (lo, hi) in _WINDOW_RULES.items():
        if keyword != 'default' and keyword in nl: return lo, hi
    return _WINDOW_RULES['default']

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_google_holidays(gcal_key: str, year_start: int = 2019, year_end: int = 2028) -> list:
    import urllib.request, urllib.parse, json as json_lib
    if not gcal_key: return []
    base_url = f"https://www.googleapis.com/calendar/v3/calendars/{GCAL_CALENDAR_ID}/events"
    all_rows = []; seen_keys = set()
    for year in range(year_start, year_end+1):
        page_token = None
        while True:
            params = {'key':gcal_key,'timeMin':f'{year}-01-01T00:00:00Z','timeMax':f'{year}-12-31T23:59:59Z',
                      'maxResults':'2500','singleEvents':'true','orderBy':'startTime'}
            if page_token: params['pageToken'] = page_token
            url = base_url + '?' + urllib.parse.urlencode(params)
            try:
                req = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=15) as r:
                    data = json_lib.loads(r.read().decode('utf-8'))
            except: break
            if 'error' in data:
                err = data['error']
                return [{'_error':True,'message':err.get('message','Unknown error'),'code':err.get('code',0)}]
            for item in data.get('items',[]):
                start_raw = (item.get('start',{}).get('date') or item.get('start',{}).get('dateTime','')[:10])
                name = item.get('summary','').strip()
                if not start_raw or not name: continue
                try: ds = pd.Timestamp(start_raw)
                except: continue
                key = (str(ds.date()), name)
                if key in seen_keys: continue
                seen_keys.add(key)
                lo, hi = _get_window(name)
                all_rows.append({'ds':ds,'holiday':name,'lower_window':lo,'upper_window':hi})
            page_token = data.get('nextPageToken')
            if not page_token: break
    return all_rows

def get_gcal_key() -> str:
    try:
        key = st.secrets.get("GCAL_KEY","")
        return key.strip() if key else ""
    except: return ""

def build_holiday_df() -> tuple:
    gcal_key = get_gcal_key()
    if not gcal_key:
        return (pd.DataFrame(columns=['ds','holiday','lower_window','upper_window']),
                "❌ GCAL_KEY tidak ditemukan di Streamlit Secrets.", False)
    rows = fetch_google_holidays(gcal_key)
    if rows and isinstance(rows[0],dict) and rows[0].get('_error'):
        err = rows[0]
        return (pd.DataFrame(columns=['ds','holiday','lower_window','upper_window']),
                f"❌ Google Calendar API error (code {err.get('code')}): {err.get('message')}", False)
    if not rows:
        return (pd.DataFrame(columns=['ds','holiday','lower_window','upper_window']),
                "⚠️ Google Calendar API tidak mengembalikan data.", False)
    df_h = pd.DataFrame(rows)
    df_h['ds'] = pd.to_datetime(df_h['ds'])
    df_h = df_h.drop_duplicates(subset=['ds','holiday']).sort_values('ds').reset_index(drop=True)
    n_total=len(df_h); n_types=df_h['holiday'].nunique(); yr_min=df_h['ds'].dt.year.min(); yr_max=df_h['ds'].dt.year.max()
    status = f"✅ **{n_total} hari libur** dari **{n_types} jenis** berhasil dimuat dari Google Calendar API ({yr_min}–{yr_max})."
    return df_h, status, True


# ══════════════════════════════════════════════════════════════════════════════
# PROPHET
# ══════════════════════════════════════════════════════════════════════════════

def run_prophet(df_monthly_raw, target, cat, n_months, holidays_df):
    if not PROPHET_OK:
        return None, "Prophet tidak terinstall."
    cat_df = df_monthly_raw[df_monthly_raw['Kategori']==cat].copy()
    if len(cat_df)<6: return None, f"Data {cat} kurang dari 6 bulan."
    cat_df = cat_df.sort_values(['Tahun','Bulan'])
    cat_df['ds'] = pd.to_datetime(cat_df['Tahun'].astype(str)+'-'+cat_df['Bulan'].astype(str).str.zfill(2)+'-01')
    cat_df = cat_df.groupby('ds')[target].sum().reset_index()
    cat_df.columns = ['ds','y']
    cat_df = cat_df[cat_df['y']>0].sort_values('ds').reset_index(drop=True)
    if len(cat_df)<6: return None, f"Data {cat} setelah filtering kurang dari 6 bulan."
    use_holidays = holidays_df is not None and len(holidays_df)>0
    y_floor=0.0; y_cap=float(cat_df['y'].max())*3.0
    try:
        n_data = len(cat_df)
        s_mode = 'multiplicative' if n_data>=24 else 'additive'
        cp_scale = 0.05 if n_data>=24 else 0.03
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
                    holidays=holidays_df if use_holidays else None, seasonality_mode=s_mode,
                    interval_width=0.80, changepoint_prior_scale=cp_scale,
                    seasonality_prior_scale=5.0, holidays_prior_scale=5.0,
                    growth='flat' if n_data<12 else 'linear')
        m.fit(cat_df)
        future = m.make_future_dataframe(periods=n_months, freq='MS')
        fc = m.predict(future)
        fc['yhat']       = fc['yhat'].clip(lower=y_floor, upper=y_cap)
        fc['yhat_lower'] = fc['yhat_lower'].clip(lower=y_floor)
        fc['yhat_upper'] = fc['yhat_upper'].clip(lower=y_floor, upper=y_cap*1.2)
        hist_pred = fc[fc['ds'].isin(cat_df['ds'])]
        if len(hist_pred)>0:
            yt = cat_df.set_index('ds').loc[hist_pred['ds'],'y'].values
            yp = hist_pred['yhat'].values
            r2_is   = float(r2_score(yt,yp)) if len(yt)>1 else 0.0
            mape_is = float(np.mean(np.abs((yt-yp)/(np.abs(yt)+1e-9)))*100)
        else: r2_is=mape_is=0.0
        return {'model':m,'forecast':fc,'history':cat_df,'r2_insample':r2_is,'mape_insample':mape_is,
                'n_holidays':len(holidays_df) if use_holidays else 0,'gcal_used':use_holidays}, None
    except Exception as e:
        return None, str(e)


def forecast(df, ml, n_years):
    target=ml['target']; nlags=ml['n_lags']; active=ml['active_programs']
    per_prog=ml.get('per_prog',{}); base_yr=int(df['Tahun'].max()); rows=[]
    STAT_METHODS = {'Holt Smoothing','Exp Smoothing','Weighted MA'}
    for cat in active:
        info    = per_prog.get(cat, None)
        history = list(df[df['Kategori']==cat].sort_values('Tahun')[target].dropna().values.astype(float))
        if not history: continue
        best_nm = info.get('best_name','Holt Smoothing') if info else 'Holt Smoothing'
        if info is None or info.get('single',True) or best_nm in STAT_METHODS:
            for fy in range(1, n_years+1):
                try:
                    if best_nm=='Holt Smoothing' or info is None:
                        pred=float(forecast_holt(history,1)[0][0])
                    elif best_nm=='Exp Smoothing':
                        pred=float(forecast_ses(history,1)[0][0])
                    elif best_nm=='Weighted MA':
                        pred=float(forecast_moving_avg(history,1)[0])
                    else:
                        pred=float(forecast_holt(history,1)[0][0])
                except: pred=history[-1]*1.05
                pred=max(0.0,pred)
                rows.append({'Kategori':cat,'Tahun':base_yr+fy,target:pred,'Type':f'Prediksi ({best_nm})'})
                history.append(pred)
            continue
        mdl=info.get('best_model'); sc=info.get('scaler'); cat_id=info.get('cat_id',0.0)
        nlags_use=info.get('n_lags_used',min(nlags,2)); last_actual=history[-1]
        for fy in range(1, n_years+1):
            Xc,_=build_features(history,nlags_use,cat_id); pred=None
            if len(Xc)>0 and mdl is not None:
                feat=Xc[-1].reshape(1,-1)
                try:
                    feat_use=sc.transform(feat) if best_nm in SCALED_MODELS else feat
                    pred=float(mdl.predict(feat_use)[0])
                    if pred<last_actual*0.5 or pred>last_actual*2.0:
                        holt_pred=float(forecast_holt(history[:fy+len(history)-1],1)[0][0])
                        pred=(pred+holt_pred)/2
                except: pred=None
            if pred is None:
                try: pred=float(forecast_holt(history,1)[0][0])
                except: pred=history[-1]*1.05
            pred=max(0.0,pred)
            rows.append({'Kategori':cat,'Tahun':base_yr+fy,target:pred,'Type':'Prediksi'})
            history.append(pred)
    return pd.DataFrame(rows)

def compute_monthly_breakdown(df_raw_monthly, yearly_pred_df, target):
    rows = []
    for cat in yearly_pred_df['Kategori'].unique():
        cat_hist = df_raw_monthly[df_raw_monthly['Kategori']==cat]
        if len(cat_hist)>=12:
            mo=cat_hist.groupby(['Tahun','Bulan'])[target].sum().reset_index()
            yr=mo.groupby('Tahun')[target].sum().reset_index(); yr.columns=['Tahun','YrTotal']
            mo=mo.merge(yr,on='Tahun'); mo['W']=mo[target]/(mo['YrTotal']+1e-9)
            weights=mo.groupby('Bulan')['W'].mean()
        else:
            weights=pd.Series({m:1/12 for m in range(1,13)})
        for m in range(1,13):
            if m not in weights.index: weights[m]=0.0
        weights=weights.sort_index(); wsum=weights.sum()
        weights=(weights/wsum) if wsum>0 else pd.Series({m:1/12 for m in range(1,13)})
        cat_pred=yearly_pred_df[yearly_pred_df['Kategori']==cat]
        for _,row in cat_pred.iterrows():
            yr_total=float(row[target]); yr_int=int(row['Tahun'])
            for bulan,w in weights.items():
                rows.append({'Kategori':cat,'Tahun':yr_int,'Bulan':int(bulan),
                             'Periode':f"{yr_int}-{int(bulan):02d}",
                             target:max(0.0,yr_total*w),'Type':'Prediksi Bulanan'})
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# EXPORT EXCEL (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def xl_col_to_name(col_idx):
    name=''
    col_idx+=1
    while col_idx:
        col_idx,remainder=divmod(col_idx-1,26)
        name=chr(65+remainder)+name
    return name

def export_excel(df, ml_result, fut_df, fut_kasus=None, fut_nominal=None, fut_monthly_kasus=None, fut_monthly_nominal=None):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
        wb  = writer.book
        hdr = wb.add_format({'bold':True,'bg_color':'#1e3a5f','font_color':'white','border':1})
        num_fmt = wb.add_format({'num_format':'#,##0','border':1})
        sec_fmt = wb.add_format({'bold':True,'bg_color':'#0f2744','font_color':'#93c5fd','font_size':12,'border':0})

        df_sorted = df.sort_values(['Tahun','Kategori']).reset_index(drop=True)
        df_sorted.to_excel(writer, sheet_name='Data Gabungan', index=False)
        ws1 = writer.sheets['Data Gabungan']
        for i,c in enumerate(df_sorted.columns):
            ws1.write(0,i,c,hdr); ws1.set_column(i,i,22)

        if 'Kasus' in df.columns:
            piv=df.pivot_table(index='Kategori',columns='Tahun',values='Kasus',aggfunc='sum',fill_value=0)
            piv.to_excel(writer, sheet_name='Pivot Kasus')
            ws2=writer.sheets['Pivot Kasus']
            for i in range(len(piv.columns)+1): ws2.set_column(i,i,18)
            nr,nc2=len(piv),len(piv.columns)
            chart_pk=wb.add_chart({'type':'line'})
            for i in range(nr):
                chart_pk.add_series({'name':['Pivot Kasus',i+1,0],'categories':['Pivot Kasus',0,1,0,nc2],'values':['Pivot Kasus',i+1,1,i+1,nc2]})
            chart_pk.set_title({'name':'Tren Kasus per Program'}); chart_pk.set_size({'width':600,'height':350})
            ws2.insert_chart(f'A{nr+4}', chart_pk)

        has_fut_k = fut_kasus is not None and len(fut_kasus)>0
        has_fut_n = fut_nominal is not None and len(fut_nominal)>0
        if not has_fut_k and not has_fut_n and fut_df is not None and len(fut_df)>0:
            tc_fb=[c for c in fut_df.columns if c not in ['Kategori','Tahun','Type']]
            if tc_fb:
                if tc_fb[0]=='Kasus': fut_kasus=fut_df; has_fut_k=True
                else: fut_nominal=fut_df; has_fut_n=True

        if has_fut_k or has_fut_n:
            ws3_name='Prediksi Tahunan'
            writer.book.add_worksheet(ws3_name)
            ws3=writer.sheets[ws3_name]
            hdr_sec=wb.add_format({'bold':True,'bg_color':'#0f2744','font_color':'#93c5fd','font_size':12})
            hdr_yr=wb.add_format({'bold':True,'bg_color':'#1e3a5f','font_color':'white','border':1,'align':'center'})
            num_k=wb.add_format({'num_format':'#,##0','border':1})
            num_hist=wb.add_format({'num_format':'#,##0','border':1,'bg_color':'#1a1a2e'})
            cursor=0

            def _write_annual_block(ws,wb,df_hist,df_pred,value_col,title_txt,cursor,ws_name):
                ph=(df_hist.groupby(['Tahun','Kategori'])[value_col].sum().reset_index()
                    .pivot(index='Tahun',columns='Kategori',values=value_col).reset_index())
                ph['_type']='Aktual'
                pp=(df_pred.groupby(['Tahun','Kategori'])[value_col].sum().reset_index()
                    .pivot(index='Tahun',columns='Kategori',values=value_col).reset_index())
                pp['_type']='Prediksi'
                combined=pd.concat([ph,pp],ignore_index=True).sort_values('Tahun').reset_index(drop=True)
                cats=[c for c in combined.columns if c not in ['Tahun','_type']]
                n_rows=len(combined); n_cats=len(cats)
                ws.merge_range(cursor,0,cursor,n_cats+1,title_txt,hdr_sec)
                hdr_row=cursor+1
                ws.write(hdr_row,0,'Tahun',hdr_yr); ws.set_column(0,0,10)
                ws.write(hdr_row,1,'Tipe',hdr_yr); ws.set_column(1,1,12)
                for ci,cat in enumerate(cats):
                    ws.write(hdr_row,ci+2,cat,hdr_yr); ws.set_column(ci+2,ci+2,20)
                data_start=hdr_row+1
                for ri,row in combined.iterrows():
                    r_abs=data_start+ri; fmt_use=num_hist if row['_type']=='Aktual' else num_k
                    ws.write(r_abs,0,int(row['Tahun'])); ws.write(r_abs,1,row['_type'])
                    for ci,cat in enumerate(cats):
                        val=row.get(cat,0); ws.write(r_abs,ci+2,float(val) if pd.notna(val) else 0,fmt_use)
                last_data_row=data_start+n_rows-1
                ch=wb.add_chart({'type':'line'})
                aktual_rows=[data_start+i for i,r in combined.iterrows() if r['_type']=='Aktual']
                pred_rows=[data_start+i for i,r in combined.iterrows() if r['_type']=='Prediksi']
                CHART_COLORS=['#4472C4','#ED7D31','#A9D18E','#FF0000','#7030A0','#00B0F0','#92D050','#FFC000']
                for ci,cat in enumerate(cats):
                    col_excel=ci+2; color=CHART_COLORS[ci%len(CHART_COLORS)]
                    if aktual_rows:
                        ch.add_series({'name':cat+' Aktual','categories':[ws_name,aktual_rows[0],0,aktual_rows[-1],0],
                                       'values':[ws_name,aktual_rows[0],col_excel,aktual_rows[-1],col_excel],
                                       'line':{'color':color,'width':2.25},'marker':{'type':'circle','size':6,'fill':{'color':color},'border':{'color':color}}})
                    if pred_rows:
                        ch.add_series({'name':cat+' Prediksi','categories':[ws_name,pred_rows[0],0,pred_rows[-1],0],
                                       'values':[ws_name,pred_rows[0],col_excel,pred_rows[-1],col_excel],
                                       'line':{'color':color,'width':2.25,'dash_type':'dash'},'marker':{'type':'diamond','size':7,'fill':{'color':color},'border':{'color':color}}})
                ch.set_title({'name':title_txt}); ch.set_x_axis({'name':'Tahun'}); ch.set_y_axis({'name':value_col,'num_format':'#,##0'})
                ch.set_legend({'position':'bottom','font':{'size':9}}); ch.set_size({'width':800,'height':450}); ch.set_style(10)
                chart_col=xl_col_to_name(n_cats+3); ws.insert_chart(f'{chart_col}{hdr_row+1}',ch)
                return last_data_row+4

            df_hist_k=df[['Tahun','Kategori','Kasus']].copy() if 'Kasus' in df.columns else None
            df_hist_n=df[['Tahun','Kategori','Nominal']].copy() if 'Nominal' in df.columns else None
            if has_fut_k and df_hist_k is not None:
                cursor=_write_annual_block(ws3,wb,df_hist_k,fut_kasus,'Kasus','📊 PREDIKSI KASUS (TAHUNAN)',cursor,ws3_name)
            if has_fut_n and df_hist_n is not None:
                cursor=_write_annual_block(ws3,wb,df_hist_n,fut_nominal,'Nominal','💰 PREDIKSI NOMINAL (TAHUNAN)',cursor,ws3_name)

        has_kasus=fut_monthly_kasus is not None and len(fut_monthly_kasus)>0
        has_nominal=fut_monthly_nominal is not None and len(fut_monthly_nominal)>0
        if has_kasus or has_nominal:
            ws4_name='Prediksi Bulanan'
            writer.book.add_worksheet(ws4_name)
            ws4=writer.sheets[ws4_name]; cursor=0
            if has_kasus:
                ws4.merge_range(cursor,0,cursor,7,'📊 PREDIKSI KASUS (BULANAN)',sec_fmt); cursor+=1
                piv_k=(fut_monthly_kasus.sort_values(['Tahun','Bulan','Kategori'])
                       .pivot_table(index='Periode',columns='Kategori',values='Kasus',aggfunc='sum')
                       .reset_index().sort_values('Periode').reset_index(drop=True))
                nrow_k=len(piv_k); ncat_k=len(piv_k.columns)-1
                header_row_k=cursor
                for ci,cn in enumerate(piv_k.columns):
                    ws4.write(cursor,ci,str(cn),hdr); ws4.set_column(ci,ci,16)
                cursor+=1
                for ri in range(nrow_k):
                    ws4.write(cursor+ri,0,piv_k.iloc[ri,0])
                    for ci in range(1,ncat_k+1): ws4.write(cursor+ri,ci,piv_k.iloc[ri,ci],num_fmt)
                last_data_row_k=cursor+nrow_k-1
                ch_k=wb.add_chart({'type':'line'})
                for ci in range(1,ncat_k+1):
                    ch_k.add_series({'name':[ws4_name,header_row_k,ci],'categories':[ws4_name,header_row_k+1,0,last_data_row_k,0],
                                     'values':[ws4_name,header_row_k+1,ci,last_data_row_k,ci],'marker':{'type':'circle','size':4}})
                ch_k.set_title({'name':'Prediksi Kasus per Program (Bulanan)'}); ch_k.set_size({'width':760,'height':420})
                chart_col_k=xl_col_to_name(ncat_k+2); ws4.insert_chart(f'{chart_col_k}{header_row_k+1}',ch_k)
                cursor=last_data_row_k+3
            if has_nominal:
                ws4.merge_range(cursor,0,cursor,7,'💰 PREDIKSI NOMINAL (BULANAN)',sec_fmt); cursor+=1
                piv_n=(fut_monthly_nominal.sort_values(['Tahun','Bulan','Kategori'])
                       .pivot_table(index='Periode',columns='Kategori',values='Nominal',aggfunc='sum')
                       .reset_index().sort_values('Periode').reset_index(drop=True))
                nrow_n=len(piv_n); ncat_n=len(piv_n.columns)-1
                header_row_n=cursor
                for ci,cn in enumerate(piv_n.columns):
                    ws4.write(cursor,ci,str(cn),hdr); ws4.set_column(ci,ci,16)
                cursor+=1
                for ri in range(nrow_n):
                    ws4.write(cursor+ri,0,piv_n.iloc[ri,0])
                    for ci in range(1,ncat_n+1): ws4.write(cursor+ri,ci,piv_n.iloc[ri,ci],num_fmt)
                last_data_row_n=cursor+nrow_n-1
                ch_n=wb.add_chart({'type':'line'})
                for ci in range(1,ncat_n+1):
                    ch_n.add_series({'name':[ws4_name,header_row_n,ci],'categories':[ws4_name,header_row_n+1,0,last_data_row_n,0],
                                     'values':[ws4_name,header_row_n+1,ci,last_data_row_n,ci],'marker':{'type':'circle','size':4}})
                ch_n.set_title({'name':'Prediksi Nominal per Program (Bulanan)'}); ch_n.set_size({'width':760,'height':420})
                chart_col_n=xl_col_to_name(ncat_n+2); ws4.insert_chart(f'{chart_col_n}{header_row_n+1}',ch_n)

        detail_frames=[]
        if has_kasus: detail_frames.append(fut_monthly_kasus[['Periode','Tahun','Bulan','Kategori','Kasus']].copy())
        if has_nominal: detail_frames.append(fut_monthly_nominal[['Periode','Tahun','Bulan','Kategori','Nominal']].copy())
        if detail_frames:
            detail_all=(detail_frames[0].merge(detail_frames[1],on=['Periode','Tahun','Bulan','Kategori'],how='outer') if len(detail_frames)==2 else detail_frames[0])
            detail_all=detail_all.sort_values(['Tahun','Bulan','Kategori']).reset_index(drop=True)
            detail_all.to_excel(writer, sheet_name='Bulanan Detail', index=False)
            ws5=writer.sheets['Bulanan Detail']
            for i,c in enumerate(detail_all.columns):
                ws5.write(0,i,c,hdr); ws5.set_column(i,i,18)
            for ci,col_name in enumerate(detail_all.columns):
                if col_name in ('Kasus','Nominal'):
                    for ri in range(len(detail_all)):
                        ws5.write(ri+1,ci,detail_all.iloc[ri][col_name],num_fmt)

        if ml_result:
            rdf=ml_result['results_df']
            rdf.to_excel(writer, sheet_name='ML Results', index=False)
            ws6=writer.sheets['ML Results']
            for i,c in enumerate(rdf.columns):
                ws6.write(0,i,c,hdr); ws6.set_column(i,i,18)

    buf.seek(0)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
      <div class="sb-logo">🟢 BPJS ML</div>
      <div class="sb-sub">Prediction Dashboard</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div style="font-size:.6rem;color:#1e3a5f;text-transform:uppercase;letter-spacing:2px;margin-bottom:8px;font-family:\'Instrument Sans\',sans-serif;">📂 Upload Dataset</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload dataset",
        type=['csv', 'xlsx', 'xls'],
        accept_multiple_files=True,
        help="Kolom: PROGRAM · KASUS · NOMINAL · DATE",
        label_visibility="collapsed",
    )
    st.markdown("""
    <div style="font-size:.75rem;color:#1e3a5f;line-height:1.8;margin-top:6px;font-family:'DM Mono',monospace;">
    <code style="color:#334155">PROGRAM</code> → Kategori<br>
    <code style="color:#334155">KASUS</code> → Jumlah kasus<br>
    <code style="color:#334155">NOMINAL</code> → Nilai (Rp)<br>
    <code style="color:#334155">DATE</code> → Periode
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div style="font-size:.6rem;color:#1e3a5f;text-transform:uppercase;letter-spacing:2px;margin-bottom:8px;font-family:\'Instrument Sans\',sans-serif;">⚙️ Parameter Model</div>', unsafe_allow_html=True)
    n_lags   = st.slider("Lag features", 1, 4, 2)
    test_pct = st.slider("Test split (%)", 10, 40, 25, 5)
    n_future = st.slider("Prediksi tahun ke depan", 1, 5, 3)

    st.markdown("---")
    st.markdown('<div style="font-size:.6rem;color:#1e3a5f;text-transform:uppercase;letter-spacing:2px;margin-bottom:8px;font-family:\'Instrument Sans\',sans-serif;">🕑 Riwayat Analisis</div>', unsafe_allow_html=True)
    history_meta = load_history_meta()
    if history_meta:
        for h in reversed(history_meta):
            col_h, col_del = st.columns([5, 1])
            with col_h:
                if st.button(h['label'], key=f"hbtn_{h['id']}", use_container_width=True):
                    df_h, res_h, extra_h = load_history_entry(h['id'])
                    if df_h is not None:
                        st.session_state.active_data     = df_h
                        st.session_state.active_results  = res_h
                        st.session_state.active_entry_id = h['id']
                        for k, v in extra_h.items():
                            st.session_state[k] = v
                        st.rerun()
                    else:
                        st.warning("Data riwayat tidak ditemukan.")
            with col_del:
                if st.button("✕", key=f"hdel_{h['id']}", help="Hapus"):
                    delete_history_entry(h['id'])
                    meta = load_history_meta()
                    meta = [m for m in meta if m['id'] != h['id']]
                    save_history_meta(meta)
                    st.rerun()
        if st.button("✕ Hapus Semua Riwayat", use_container_width=True):
            for h in history_meta:
                delete_history_entry(h['id'])
            save_history_meta([])
            st.rerun()
    else:
        st.markdown('<div style="font-size:.75rem;color:#1e3a5f;font-style:italic;padding:8px 0;">Belum ada riwayat.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

if uploaded:
    files_info = []
    for f in uploaded:
        ym = re.search(r'(\d{4})', f.name)
        year_hint = int(ym.group(1)) if ym else 2000
        raw = load_raw(f.read(), f.name)
        if raw is not None:
            files_info.append((year_hint, raw, f.name))
            st.sidebar.success(f"✅ {f.name}")
        else:
            st.sidebar.error(f"❌ {f.name}: gagal dibaca")

    if files_info:
        merged, errs = merge_all(files_info)
        for e in errs: st.sidebar.warning(e)
        if merged is not None and len(merged) > 0:
            dh  = hashlib.md5(merged.to_csv().encode()).hexdigest()[:8]
            cur = (hashlib.md5(st.session_state.active_data.to_csv().encode()).hexdigest()[:8]
                   if st.session_state.active_data is not None else None)
            if cur != dh:
                st.session_state.active_data    = merged
                st.session_state.active_results = {}

            raw_monthly_frames = []
            for yh, raw_df, fname in files_info:
                m_cols = _detect_cols_quick(raw_df)
                if m_cols:
                    rm = _build_raw_monthly(raw_df, yh, m_cols)
                    if rm is not None and len(rm) > 0:
                        raw_monthly_frames.append(rm)
            if raw_monthly_frames:
                rm_combined = pd.concat(raw_monthly_frames, ignore_index=True)
                st.session_state['raw_monthly'] = rm_combined
                st.sidebar.success(f"📆 {len(rm_combined)} baris monthly")
            else:
                st.session_state['raw_monthly'] = None
        else:
            st.error("Gagal memproses data. Pastikan file punya kolom PROGRAM dan KASUS.")

df            = st.session_state.active_data
results_cache = st.session_state.active_results
df_raw_monthly = st.session_state.get('raw_monthly', None)

# ══════════════════════════════════════════════════════════════════════════════
# EMPTY STATE
# ══════════════════════════════════════════════════════════════════════════════

if df is None:
    st.markdown("""
    <div class="empty-state">
      <div class="empty-icon">⬡</div>
      <div class="empty-title">BPJS ML Dashboard</div>
      <div class="empty-sub">
        Upload dataset untuk memulai analisis prediktif klaim.
        Sistem akan otomatis memilih model terbaik per program.
      </div>
    </div>
    """, unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)
    for col, icon, title, desc in [
        (f1,"🤖","Model Adaptif","Holt Smoothing, SES, WMA untuk data kecil. ML (XGBoost, RF) otomatis aktif untuk data ≥ 8 tahun."),
        (f2,"📅","Kalender Indonesia","Prophet + Google Calendar Indonesia. Semua hari libur nasional otomatis diambil dari API resmi Google."),
        (f3,"📥","Export Excel","Export prediksi tahunan & bulanan ke Excel dengan chart otomatis."),
    ]:
        with col:
            st.markdown(f'''<div class="feat-card">
            <div class="feat-icon">{icon}</div>
            <div class="feat-title">{title}</div>
            <div class="feat-desc">{desc}</div>
            </div>''', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="info-box">📋 <b>Format kolom:</b> <code>PROGRAM</code> · <code>KASUS</code> · <code>NOMINAL</code> · <code>DATE</code><br>Upload 1+ file CSV/Excel. Nama file tidak harus mengandung tahun.</div>', unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# DERIVED META
# ══════════════════════════════════════════════════════════════════════════════

years         = sorted(df['Tahun'].unique())
latest_year   = years[-1]
active_progs  = get_active_programs(df)
all_progs     = sorted(df['Kategori'].unique())
has_nom       = 'Nominal' in df.columns
targets       = ['Kasus'] + (['Nominal'] if has_nom else [])
single_yr     = len(years) == 1
prog_changes  = analyze_program_changes(df)

with st.expander("🔍 Info Parsing & Program Aktif", expanded=False):
    change_html = ""
    for (y0, y1), ch in prog_changes.items():
        change_html += f"<b style='color:#64748b'>{y0} → {y1}:</b> &nbsp;"
        for p in ch['added']:    change_html += f'<span class="tag-add">+ {p}</span> '
        for p in ch['removed']:  change_html += f'<span class="tag-rem">– {p}</span> '
        for p in ch['stable']:   change_html += f'<span class="tag-stable">{p}</span> '
        change_html += "<br>"
    st.markdown(f"""
    <div class="info-box">
    📋 <b>Kolom:</b> {', '.join(df.columns.tolist())}<br>
    📅 <b>Tahun:</b> {', '.join(map(str, years))}<br>
    🏷️ <b>Semua program ({len(all_progs)}):</b> {', '.join(all_progs)}<br>
    ✅ <b>Program aktif tahun {latest_year} ({len(active_progs)}):</b> {', '.join(active_progs)}<br>
    📊 <b>Total baris:</b> {len(df)}<br><br>
    <b>Perubahan Program:</b><br>{change_html if change_html else 'Hanya 1 tahun data.'}
    </div>""", unsafe_allow_html=True)
    verify_cols = ['Tahun', 'Kategori', 'Kasus']
    if 'Nominal' in df.columns: verify_cols.append('Nominal')
    vdf = df[verify_cols].copy().sort_values(['Tahun','Kategori'])
    if 'Nominal' in vdf.columns:
        vdf['Nominal (T)'] = (vdf['Nominal']/1e12).round(4)
        vdf['Nominal (B)'] = (vdf['Nominal']/1e9).round(2)
    st.dataframe(vdf, use_container_width=True, height=320)


# ══════════════════════════════════════════════════════════════════════════════
# HERO HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero-wrap">
  <div class="hero-logo">🟢 Dashboard Prediksi Klaim BPJS Ketenagakerjaan</div>
  <div class="hero-sub">Analisis tren historis &amp; proyeksi — metode statistik &amp; machine learning adaptif</div>
</div>""", unsafe_allow_html=True)

if single_yr:
    st.markdown('<div class="warn">⚠️ <b>Mode 1 Tahun:</b> Prediksi menggunakan ekstrapolasi asumsi pertumbuhan 5%/tahun. Upload data multi-tahun untuk prediksi ML penuh.</div>', unsafe_allow_html=True)

if prog_changes:
    last_change = list(prog_changes.items())[-1]
    (y0, y1), ch = last_change
    if ch['added'] or ch['removed']:
        added_str   = ', '.join(ch['added']) if ch['added'] else '–'
        removed_str = ', '.join(ch['removed']) if ch['removed'] else '–'
        st.markdown(f'<div class="warn">📌 <b>Perubahan {y0}→{y1}:</b> &nbsp; Ditambah: <b style="color:#86efac">{added_str}</b> &nbsp;|&nbsp; Dihapus: <b style="color:#fca5a5">{removed_str}</b></div>', unsafe_allow_html=True)

df_active_only = df[df['Kategori'].isin(active_progs)]


# ══════════════════════════════════════════════════════════════════════════════
# KPI STRIP
# ══════════════════════════════════════════════════════════════════════════════

kpi_delta_k = kpi_delta_n = kpi_avg_growth = ""
if len(years) >= 2:
    yr_kasus = df_active_only.groupby('Tahun')['Kasus'].sum()
    if yr_kasus.iloc[-1] > 0 and yr_kasus.iloc[-2] > 0:
        delta_k_pct = (yr_kasus.iloc[-1]/yr_kasus.iloc[-2]-1)*100
        sign_k = "▲" if delta_k_pct >= 0 else "▼"
        cls_k  = "delta-pos" if delta_k_pct >= 0 else "delta-neg"
        kpi_delta_k = f'<div class="delta {cls_k}">{sign_k} {abs(delta_k_pct):.1f}% vs {years[-2]}</div>'
    if has_nom:
        yr_nom = df_active_only.groupby('Tahun')['Nominal'].sum()
        if yr_nom.iloc[-1] > 0 and yr_nom.iloc[-2] > 0:
            delta_n_pct = (yr_nom.iloc[-1]/yr_nom.iloc[-2]-1)*100
            sign_n = "▲" if delta_n_pct >= 0 else "▼"
            cls_n  = "delta-pos" if delta_n_pct >= 0 else "delta-neg"
            kpi_delta_n = f'<div class="delta {cls_n}">{sign_n} {abs(delta_n_pct):.1f}% vs {years[-2]}</div>'
    growths = []
    for i in range(1, len(years)):
        k_prev = df_active_only[df_active_only['Tahun']==years[i-1]]['Kasus'].sum()
        k_curr = df_active_only[df_active_only['Tahun']==years[i]]['Kasus'].sum()
        if k_prev > 0: growths.append((k_curr/k_prev-1)*100)
    avg_g = np.mean(growths) if growths else 0
    sign_g = "▲" if avg_g >= 0 else "▼"
    cls_g  = "delta-pos" if avg_g >= 0 else "delta-neg"
    kpi_avg_growth = f'<div class="delta {cls_g}">{sign_g} {abs(avg_g):.1f}%/thn</div>'

tk        = int(df_active_only['Kasus'].sum())
tk_latest = int(df_active_only[df_active_only['Tahun']==latest_year]['Kasus'].sum())

def fmt_compact(n, prefix='', suffix=''):
    """Format large numbers compactly: 7,001,353 → 7.0 Jt"""
    n = float(n)
    if abs(n) >= 1e9:    return f"{prefix}{n/1e9:.2f}M{suffix}"   # miliar
    elif abs(n) >= 1e6:  return f"{prefix}{n/1e6:.2f}Jt{suffix}"  # juta
    elif abs(n) >= 1e3:  return f"{prefix}{n/1e3:.1f}Rb{suffix}"  # ribu
    else:                return f"{prefix}{n:,.0f}{suffix}"

c1,c2,c3,c4,c5 = st.columns(5)
with c1:
    st.markdown(f'<div class="kpi"><div class="val">{len(years)}</div><div class="lbl">📅 Tahun Data</div><div class="delta delta-neu">{years[0]} – {years[-1]}</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="kpi"><div class="val">{len(active_progs)}</div><div class="lbl">🏷️ Program Aktif</div><div class="delta delta-neu">{", ".join(active_progs[:2])}{"…" if len(active_progs)>2 else ""}</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="kpi" title="Total: {tk_latest:,} kasus"><div class="val val-tooltip">{fmt_compact(tk_latest)}</div><div class="lbl">📋 Kasus {latest_year}</div>{kpi_delta_k}</div>', unsafe_allow_html=True)
with c4:
    if has_nom:
        tn_raw = df_active_only[df_active_only['Tahun']==latest_year]['Nominal'].sum()
        tn_str = fmt_compact(tn_raw/1e9, prefix='Rp', suffix='B') if tn_raw < 1e12 else fmt_compact(tn_raw/1e12, prefix='Rp', suffix='T')
        st.markdown(f'<div class="kpi" title="Total: Rp {tn_raw:,.0f}"><div class="val val-tooltip">{tn_str}</div><div class="lbl">💰 Nominal {latest_year}</div>{kpi_delta_n}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="kpi"><div class="val">{latest_year}</div><div class="lbl">📅 Tahun Terbaru</div></div>', unsafe_allow_html=True)
with c5:
    st.markdown(f'<div class="kpi" title="Total: {tk:,} kasus (semua tahun)"><div class="val val-tooltip">{fmt_compact(tk)}</div><div class="lbl">📊 Total Kasus</div>{kpi_avg_growth}</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

df_plot = df[df['Kategori'].isin(active_progs)].copy()

tab1, tab2, tab3, tab4 = st.tabs(["📈  Overview", "🤖  ML Analysis", "🔮  Prediksi", "📥  Export"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    df_lat = df_plot[df_plot['Tahun'] == latest_year]

    if not single_yr:
        top_prog = df_lat.groupby('Kategori')['Kasus'].sum().idxmax()
        top_val  = int(df_lat.groupby('Kategori')['Kasus'].sum().max())
        growth_by_prog = {}
        for cp in active_progs:
            cd = df_plot[df_plot['Kategori']==cp].sort_values('Tahun')
            if len(cd)>=2 and cd['Kasus'].iloc[-2]>0:
                growth_by_prog[cp] = (cd['Kasus'].iloc[-1]/cd['Kasus'].iloc[-2]-1)*100
        fastest = max(growth_by_prog, key=growth_by_prog.get) if growth_by_prog else "-"
        fastest_g = growth_by_prog.get(fastest, 0)
        slowest = min(growth_by_prog, key=growth_by_prog.get) if growth_by_prog else "-"
        slowest_g = growth_by_prog.get(slowest, 0)

        ia,ib,ic,id_ = st.columns(4)
        for col, title, val, sub in [
            (ia,"🏆 Program Terbesar", top_prog, f"{top_val:,} kasus di {latest_year}"),
            (ib,"📈 Pertumbuhan Tertinggi", fastest, f"+{fastest_g:.1f}% vs tahun lalu"),
            (ic,"📉 Pertumbuhan Terendah", slowest, f"{slowest_g:+.1f}% vs tahun lalu"),
            (id_,"📋 Total Kasus", fmt_compact(int(df_lat['Kasus'].sum())), f"{len(active_progs)} program aktif"),
        ]:
            with col:
                st.markdown(f'<div class="insight-card"><div class="ic-title">{title}</div><div class="ic-val">{val}</div><div class="ic-sub">{sub}</div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    r1, r2 = st.columns(2)
    with r1:
        st.markdown('<div class="sec">Distribusi Kasus — Semua Tahun</div>', unsafe_allow_html=True)
        pie_d = df_plot.groupby('Kategori')['Kasus'].sum().reset_index()
        pie_d['pct'] = pie_d['Kasus']/pie_d['Kasus'].sum()*100
        fig = go.Figure(go.Pie(
            labels=pie_d['Kategori'], values=pie_d['Kasus'],
            hole=0.6, textinfo='label+percent', textposition='outside',
            marker=dict(colors=COLORS[:len(pie_d)], line=dict(color='#020c18', width=2)),
            hovertemplate='<b>%{label}</b><br>Kasus: %{value:,}<br>%{percent}<extra></extra>',
        ))
        total_kasus = int(pie_d['Kasus'].sum())
        fig.add_annotation(text=f"<b style='font-size:18px'>{fmt_compact(total_kasus)}</b><br>Total",
            showarrow=False, font=dict(size=13, color='#e2e8f0', family='Syne'), align='center')
        fig.update_layout(**DARK, showlegend=True, height=420,
            legend=dict(orientation='h', y=-0.1, font=dict(size=10.5, family='Instrument Sans'), bgcolor='rgba(0,0,0,0)'),
            margin=dict(t=20, b=60, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    with r2:
        st.markdown(f'<div class="sec">Market Share — {latest_year}</div>', unsafe_allow_html=True)
        bar_d = df_lat.groupby('Kategori')['Kasus'].sum().sort_values(ascending=True).reset_index()
        bar_d['Share'] = (bar_d['Kasus']/bar_d['Kasus'].sum()*100).round(1)
        fig2 = go.Figure()
        for i, row in bar_d.iterrows():
            col_c = COLORS[i % len(COLORS)]
            fig2.add_trace(go.Bar(
                x=[row['Kasus']], y=[row['Kategori']], orientation='h',
                name=row['Kategori'], showlegend=False,
                marker=dict(color=col_c, line=dict(width=0),
                    pattern=dict(shape='', bgcolor=hex_to_rgba(col_c,0.15))),
                text=f"{row['Kasus']:,} ({row['Share']}%)",
                textposition='outside', textfont=dict(size=10.5, color='#64748b', family='DM Mono'),
                hovertemplate=f"<b>{row['Kategori']}</b><br>{row['Kasus']:,} kasus ({row['Share']}%)<extra></extra>"
            ))
        styled_chart(fig2, height=420, legend_bottom=False)
        fig2.update_layout(showlegend=False, xaxis=dict(showgrid=True), margin=dict(t=20,b=20,l=20,r=130))
        st.plotly_chart(fig2, use_container_width=True)

    if not single_yr:
        trend = df_plot.groupby(['Tahun','Kategori'])['Kasus'].sum().reset_index()
        st.markdown('<div class="sec">Tren Kasus per Tahun</div>', unsafe_allow_html=True)
        fig3 = go.Figure()
        for i, cat in enumerate(sorted(df_plot['Kategori'].unique())):
            cd = trend[trend['Kategori']==cat].sort_values('Tahun')
            col_c = COLORS[i % len(COLORS)]
            fig3.add_trace(go.Scatter(
                x=cd['Tahun'], y=cd['Kasus'], name=cat, mode='lines+markers',
                line=dict(color=col_c, width=2.5, shape='spline'),
                marker=dict(size=9, color=col_c, line=dict(color='#020c18', width=2.5),
                    symbol='circle'),
                fill='tozeroy', fillcolor=hex_to_rgba(col_c, 0.04),
                hovertemplate=f"<b>{cat}</b><br>Tahun: %{{x}}<br>Kasus: %{{y:,}}<extra></extra>"
            ))
        styled_chart(fig3, height=440)
        fig3.update_layout(xaxis=dict(dtick=1))
        st.plotly_chart(fig3, use_container_width=True)

        t3l, t3r = st.columns(2)
        with t3l:
            st.markdown('<div class="sec">Komposisi Stacked per Tahun</div>', unsafe_allow_html=True)
            fig4 = go.Figure()
            for i, cat in enumerate(sorted(df_plot['Kategori'].unique())):
                cd = trend[trend['Kategori']==cat].sort_values('Tahun')
                fig4.add_trace(go.Bar(
                    x=cd['Tahun'], y=cd['Kasus'], name=cat,
                    marker_color=COLORS[i % len(COLORS)],
                    marker_line_width=0,
                    hovertemplate=f"<b>{cat}</b><br>%{{y:,}}<extra></extra>"
                ))
            fig4.update_layout(barmode='stack')
            styled_chart(fig4, height=380)
            fig4.update_layout(xaxis=dict(dtick=1))
            st.plotly_chart(fig4, use_container_width=True)

        with t3r:
            st.markdown('<div class="sec">Heatmap Kasus (Program × Tahun)</div>', unsafe_allow_html=True)
            hm_p = (df_plot.groupby(['Kategori','Tahun'])['Kasus'].sum()
                    .reset_index().pivot(index='Kategori',columns='Tahun',values='Kasus').fillna(0))
            fig5 = go.Figure(go.Heatmap(
                z=hm_p.values, x=[str(c) for c in hm_p.columns], y=list(hm_p.index),
                colorscale=[[0,'#020c18'],[0.3,'#0f2744'],[0.7,'#1d4ed8'],[1,'#38bdf8']],
                text=[[f'{v:,.0f}' for v in row] for row in hm_p.values],
                texttemplate='%{text}', textfont=dict(size=10, color='#e2e8f0'),
                hovertemplate='<b>%{y}</b> · %{x}<br>Kasus: %{z:,}<extra></extra>',
                colorbar=dict(tickfont=dict(color='#334155',size=9),
                    title=dict(text='Kasus',font=dict(color='#475569',size=10)),
                    thickness=10, len=0.8),
            ))
            fig5.update_layout(**DARK, height=380, margin=dict(t=20,b=20,l=20,r=80))
            st.plotly_chart(fig5, use_container_width=True)

        st.markdown('<div class="sec">Year-over-Year Growth per Program</div>', unsafe_allow_html=True)
        yoy = []
        for cat in active_progs:
            cd = df_plot[df_plot['Kategori']==cat].sort_values('Tahun')
            for i in range(1, len(cd)):
                prev=cd.iloc[i-1]['Kasus']; curr=cd.iloc[i]['Kasus']
                yoy.append({'Kategori':cat,'Tahun':int(cd.iloc[i]['Tahun']),'Growth (%)':round((curr/(prev+1e-9)-1)*100,2)})
        if yoy:
            ydf = pd.DataFrame(yoy)
            fig_y = go.Figure()
            for i, cat in enumerate(sorted(active_progs)):
                cd_y = ydf[ydf['Kategori']==cat].sort_values('Tahun')
                col_c = COLORS[i % len(COLORS)]
                fig_y.add_trace(go.Bar(
                    x=cd_y['Tahun'], y=cd_y['Growth (%)'], name=cat,
                    marker_color=col_c, marker_line_width=0,
                    text=cd_y['Growth (%)'].apply(lambda v: f'{v:+.1f}%'),
                    textposition='outside', textfont=dict(size=9, color='#475569', family='DM Mono'),
                    hovertemplate=f"<b>{cat}</b><br>%{{x}}: %{{y:+.1f}}%<extra></extra>"
                ))
            fig_y.add_hline(y=0, line_color='rgba(30,58,95,.8)', line_width=1.5)
            fig_y.update_layout(barmode='group')
            styled_chart(fig_y, height=380)
            fig_y.update_layout(xaxis=dict(dtick=1))
            st.plotly_chart(fig_y, use_container_width=True)

        if len(years) >= 3:
            st.markdown('<div class="sec">Distribusi & Variabilitas per Program</div>', unsafe_allow_html=True)
            fig_box = go.Figure()
            for i, cat in enumerate(sorted(active_progs)):
                cd = df_plot[df_plot['Kategori']==cat]['Kasus'].values
                col_c = COLORS[i % len(COLORS)]
                fig_box.add_trace(go.Box(
                    y=cd, name=cat,
                    marker_color=col_c, line_color=col_c,
                    fillcolor=hex_to_rgba(col_c, 0.12), boxmean='sd',
                    hovertemplate=f"<b>{cat}</b><br>%{{y:,}}<extra></extra>"
                ))
            styled_chart(fig_box, height=360, legend_bottom=False)
            fig_box.update_layout(showlegend=False, yaxis_title='Kasus', margin=dict(t=20,b=20,l=60,r=20))
            st.plotly_chart(fig_box, use_container_width=True)

    if has_nom:
        st.markdown('<div class="sec">Analisis Nominal (Rp)</div>', unsafe_allow_html=True)
        nc1, nc2 = st.columns(2)
        with nc1:
            np_d = df_plot.groupby('Kategori')['Nominal'].sum().reset_index()
            np_d['Nominal_B'] = np_d['Nominal']/1e9
            total_nom = np_d['Nominal_B'].sum()
            fp = go.Figure(go.Pie(
                labels=np_d['Kategori'], values=np_d['Nominal'], hole=0.6,
                textinfo='label+percent', textposition='outside',
                marker=dict(colors=COLORS[:len(np_d)], line=dict(color='#020c18', width=2)),
                hovertemplate='<b>%{label}</b><br>Rp%{value:,.0f}<extra></extra>',
            ))
            fp.add_annotation(text=f"<b>Rp{total_nom:,.1f}B</b><br>Total",
                showarrow=False, font=dict(size=12, color='#e2e8f0', family='Syne'), align='center')
            fp.update_layout(**DARK, showlegend=False, height=380, margin=dict(t=20,b=20,l=10,r=10))
            st.plotly_chart(fp, use_container_width=True)
        with nc2:
            if not single_yr:
                nt = df_plot.groupby(['Tahun','Kategori'])['Nominal'].sum().reset_index()
                nt['Nominal_B'] = nt['Nominal']/1e9
                fn = go.Figure()
                for i, cat in enumerate(sorted(df_plot['Kategori'].unique())):
                    cd = nt[nt['Kategori']==cat].sort_values('Tahun')
                    col_c = COLORS[i % len(COLORS)]
                    fn.add_trace(go.Scatter(
                        x=cd['Tahun'], y=cd['Nominal_B'], name=cat,
                        mode='lines+markers', stackgroup='one',
                        line=dict(color=col_c, width=1.5),
                        fillcolor=hex_to_rgba(col_c, 0.25),
                        hovertemplate=f"<b>{cat}</b><br>Rp%{{y:,.1f}}B<extra></extra>"
                    ))
                styled_chart(fn, height=380)
                fn.update_layout(xaxis=dict(dtick=1), yaxis_title='Rp Miliar')
                st.plotly_chart(fn, use_container_width=True)
            else:
                nb = df_plot.groupby('Kategori')['Nominal'].sum().reset_index()
                nb['Nominal_B'] = nb['Nominal']/1e9
                fn = go.Figure()
                for i, (_, row) in enumerate(nb.sort_values('Nominal_B',ascending=True).iterrows()):
                    fn.add_trace(go.Bar(
                        x=[row['Nominal_B']], y=[row['Kategori']], orientation='h',
                        showlegend=False, marker_color=COLORS[i%len(COLORS)], marker_line_width=0,
                        text=f"Rp{row['Nominal_B']:,.1f}B", textposition='outside',
                        textfont=dict(size=10.5, color='#64748b', family='DM Mono'),
                    ))
                styled_chart(fn, height=380, legend_bottom=False)
                fn.update_layout(margin=dict(t=10,b=10,l=10,r=100))
                st.plotly_chart(fn, use_container_width=True)

        if not single_yr:
            st.markdown('<div class="sec">Korelasi Kasus vs Nominal</div>', unsafe_allow_html=True)
            corr_data = df_plot.groupby(['Kategori','Tahun']).agg(Kasus=('Kasus','sum'),Nominal=('Nominal','sum')).reset_index()
            corr_data['Nominal_B'] = corr_data['Nominal']/1e9
            fig_sc = px.scatter(corr_data, x='Kasus', y='Nominal_B', color='Kategori',
                size='Kasus', text='Tahun', color_discrete_sequence=COLORS,
                labels={'Nominal_B':'Nominal (Rp Miliar)','Kasus':'Jumlah Kasus'})
            fig_sc.update_traces(textposition='top center', textfont=dict(size=9, family='DM Mono'))
            styled_chart(fig_sc, height=400)
            st.plotly_chart(fig_sc, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ML ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    col_s, _ = st.columns([1, 3])
    with col_s:
        target_ml = st.selectbox("Target Prediksi", targets, key='ml_target')
        run_btn   = st.button("🚀  Jalankan Analisis ML", type="primary", use_container_width=True)

    ck     = f"{target_ml}_lags{n_lags}_test{test_pct}"
    ml_res = results_cache.get(ck)

    if run_btn:
        with st.spinner(f"Melatih model untuk {len(active_progs)} program…"):
            ml_res, err = run_ml(df, target_ml, n_lags, test_pct/100)
        if err:
            st.error(f"Error: {err}"); ml_res = None
        else:
            results_cache[ck] = ml_res
            st.session_state.active_results = results_cache
            with st.spinner("Menganalisis model per program…"):
                st.session_state[f'per_prog_{target_ml}'] = ml_res
            data_hash = hashlib.md5(df.to_csv().encode()).hexdigest()[:8]
            eid   = f"{data_hash}_{target_ml}_L{n_lags}_T{test_pct}"
            label = f"📁 {datetime.now().strftime('%d/%m %H:%M')} | {target_ml} | {len(years)}yr | {len(active_progs)} prog"
            extra_snapshot = {k: st.session_state[k] for k in
                ['raw_monthly','forecast_Kasus','forecast_Nominal','forecast_monthly_Kasus',
                 'forecast_monthly_Nominal','forecast_annual_Kasus','forecast_annual_Nominal',
                 'last_forecast','last_forecast_monthly']
                if k in st.session_state and st.session_state[k] is not None}
            add_to_history(label, eid, df.copy(), dict(results_cache), extra_snapshot)
            st.session_state.active_entry_id = eid

    if ml_res:
        bpp      = ml_res.get('best_per_program', pd.DataFrame())
        per_prog = ml_res.get('per_prog', {})
        n_yrs    = len(sorted(df['Tahun'].unique()))
        avg_mape = float(bpp['MAPE (%)'].mean()) if not bpp.empty and 'MAPE (%)' in bpp.columns else 0.0
        mape_grade = ("🟢 Sangat Akurat" if avg_mape<10 else "🔵 Akurat" if avg_mape<20
                      else "🟡 Cukup" if avg_mape<50 else "🔴 Perlu Perbaikan")
        data_note  = f"&nbsp;|&nbsp; ⚠️ {n_yrs} tahun → gunakan MAPE" if n_yrs<8 else ""
        mode_note  = '&nbsp;|&nbsp; ⚠️ Mode 1 Tahun' if single_yr else ''
        st.markdown(
            f'<div class="badge">🎯 <b>Per-Program Best Model</b>'
            f'&nbsp;|&nbsp; Avg MAPE = <b>{avg_mape:.2f}%</b> ({mape_grade})'
            f'&nbsp;|&nbsp; <b>{len(active_progs)} program</b>{data_note}{mode_note}</div>',
            unsafe_allow_html=True)

        mtab1, mtab2, mtab3, mtab4 = st.tabs(["📊  Perbandingan", "🎯  Model per Program", "📝  Conclusion", "🔮  Prophet + Kalender"])

        with mtab1:
            st.markdown('<div class="sec">Model Terbaik per Program</div>', unsafe_allow_html=True)
            if not bpp.empty:
                def badge_r2(v):
                    return "🟢" if v and v>0.8 else "🔵" if v and v>0.6 else "🟡" if v and v>0.3 else "🔴"
                def badge_mape(v):
                    return "🟢" if v and v<10 else "🔵" if v and v<20 else "🟡" if v and v<50 else "🔴"
                bpp_disp = bpp.copy()
                bpp_disp['R²'] = bpp_disp['R2'].apply(badge_r2)
                bpp_disp['MAPE Grade'] = bpp_disp['MAPE (%)'].apply(badge_mape)
                st.dataframe(
                    bpp_disp.style
                       .highlight_max(subset=['R2'], color='#0a2a12')
                       .highlight_min(subset=['MAPE (%)'], color='#0a2a12')
                       .format({'R2':'{:.4f}','MAPE (%)':'{:.2f}','MAE':'{:,.0f}','RMSE':'{:,.0f}'}),
                    use_container_width=True, height=280)

                ml_ta, ml_tb = st.columns(2)
                with ml_ta:
                    bpp_sorted = bpp.sort_values('MAPE (%)').copy() if not bpp.empty else bpp
                    fig_bm = go.Figure()
                    for i, (_, row) in enumerate(bpp_sorted.iterrows()):
                        mv   = row['MAPE (%)']
                        col_c = ('#34d399' if mv<10 else '#38bdf8' if mv<20 else '#fbbf24' if mv<50 else '#f87171')
                        fig_bm.add_trace(go.Bar(
                            x=[mv], y=[row['Program']], orientation='h', showlegend=False,
                            marker=dict(color=col_c, line=dict(width=0)),
                            text=f"{mv:.1f}%  ·  {row['Model']}",
                            textposition='outside', textfont=dict(size=9.5, color='#64748b', family='DM Mono'),
                            hovertemplate=f"<b>{row['Program']}</b><br>MAPE: {mv:.2f}%<br>Model: {row['Model']}<extra></extra>"
                        ))
                    fig_bm.add_vline(x=20, line_dash='dash', line_color='rgba(251,191,36,.4)', line_width=1.5,
                        annotation_text='Target <20%', annotation_font=dict(color='#fbbf24', size=9, family='DM Mono'),
                        annotation_position='top')
                    styled_chart(fig_bm, height=380, title='MAPE per Program (lebih rendah = lebih baik)', legend_bottom=False)
                    fig_bm.update_layout(xaxis_title='MAPE (%)', margin=dict(t=50,b=20,l=20,r=140))
                    st.plotly_chart(fig_bm, use_container_width=True)

                with ml_tb:
                    if not bpp.empty:
                        type_counts = bpp['Model'].value_counts().reset_index()
                        type_counts.columns = ['Model','Count']
                        fig_mt = go.Figure(go.Pie(
                            labels=type_counts['Model'], values=type_counts['Count'],
                            hole=0.55, textinfo='label+percent', textposition='outside',
                            marker=dict(colors=COLORS[:len(type_counts)], line=dict(color='#020c18',width=2)),
                        ))
                        fig_mt.update_layout(**DARK, height=380, showlegend=False,
                            title=dict(text='Distribusi Model Terbaik',font=dict(size=12,color='#64748b'),x=0),
                            margin=dict(t=50,b=20,l=20,r=20))
                        st.plotly_chart(fig_mt, use_container_width=True)

                mc1, mc2 = st.columns(2)
                bpp_plot = bpp.dropna(subset=['R2','MAPE (%)'])
                with mc1:
                    if not bpp_plot.empty:
                        fig_r2 = go.Figure()
                        for i, (_, row) in enumerate(bpp_plot.iterrows()):
                            fig_r2.add_trace(go.Bar(
                                x=[row['Program']], y=[row['R2']], name=row['Model'],
                                marker_color=COLORS[i%len(COLORS)], marker_line_width=0,
                                text=f"{row['R2']:.3f}", textposition='outside',
                                textfont=dict(size=9, color='#64748b', family='DM Mono'),
                            ))
                        fig_r2.add_hline(y=0.8, line_dash='dot', line_color='rgba(52,211,153,.4)',
                            annotation_text='0.8', annotation_font=dict(color='#34d399',size=9,family='DM Mono'))
                        styled_chart(fig_r2, height=360, title='R² per Program (LOO-CV)')
                        st.plotly_chart(fig_r2, use_container_width=True)
                with mc2:
                    if not bpp_plot.empty:
                        fig_mp = go.Figure()
                        for i, (_, row) in enumerate(bpp_plot.iterrows()):
                            fig_mp.add_trace(go.Bar(
                                x=[row['Program']], y=[row['MAPE (%)']], name=row['Model'],
                                marker_color=COLORS[i%len(COLORS)], marker_line_width=0,
                                text=f"{row['MAPE (%)']:.1f}%", textposition='outside',
                                textfont=dict(size=9, color='#64748b', family='DM Mono'),
                            ))
                        fig_mp.add_hline(y=20, line_dash='dot', line_color='rgba(251,191,36,.4)',
                            annotation_text='20%', annotation_font=dict(color='#fbbf24',size=9,family='DM Mono'))
                        styled_chart(fig_mp, height=360, title='MAPE % per Program')
                        st.plotly_chart(fig_mp, use_container_width=True)

                if not single_yr and per_prog:
                    st.markdown('<div class="sec">Tren Historis per Program</div>', unsafe_allow_html=True)
                    fig_av = go.Figure()
                    for i, (cat, info) in enumerate(per_prog.items()):
                        hist = info.get('history', [])
                        if len(hist) < 2: continue
                        col_c = COLORS[i % len(COLORS)]
                        fig_av.add_trace(go.Scatter(
                            y=hist, name=cat, mode='lines+markers',
                            line=dict(color=col_c, width=2.5, shape='spline'),
                            marker=dict(size=7, color=col_c, line=dict(color='#020c18',width=2)),
                            fill='tozeroy', fillcolor=hex_to_rgba(col_c, 0.05),
                        ))
                    styled_chart(fig_av, height=380)
                    fig_av.update_layout(yaxis_title=target_ml, xaxis_title='Index Tahun')
                    st.plotly_chart(fig_av, use_container_width=True)

                    tree_models = ('Random Forest','Gradient Boosting','Decision Tree','Extra Trees','XGBoost','LightGBM')
                    lnames = [f'Lag_{j}' for j in range(1, n_lags+1)]
                    extras = ['MA3','Std3','MA6','Std6','Trend','Trend2','Min6','Max6','cat_id']
                    for i, (cat, info) in enumerate(per_prog.items()):
                        mdl_name = info.get('best_name','')
                        mdl_obj  = info.get('best_model', None)
                        if mdl_name in tree_models and mdl_obj is not None:
                            try:
                                fi_vals = mdl_obj.feature_importances_
                                nf = len(fi_vals)
                                fnames = (lnames + extras)[:nf]
                                while len(fnames)<nf: fnames.append(f'feat_{len(fnames)}')
                                fi_df = pd.DataFrame({'Feature':fnames,'Importance':fi_vals})\
                                          .sort_values('Importance',ascending=False).head(8)
                                fig_fi = go.Figure(go.Bar(
                                    x=fi_df['Importance'], y=fi_df['Feature'], orientation='h',
                                    marker=dict(color=COLORS[i%len(COLORS)], line=dict(width=0)),
                                    text=fi_df['Importance'].apply(lambda v: f'{v:.3f}'),
                                    textposition='outside', textfont=dict(size=9, color='#64748b', family='DM Mono'),
                                ))
                                styled_chart(fig_fi, height=300, title=f'Feature Importance — {cat} ({mdl_name})', legend_bottom=False)
                                fig_fi.update_layout(margin=dict(l=100,t=50,b=20))
                                st.plotly_chart(fig_fi, use_container_width=True)
                            except: pass

        with mtab2:
            det = ml_res.get('detail', pd.DataFrame())
            if bpp.empty:
                st.info("Klik **Jalankan Analisis ML** untuk melihat model terbaik per program.")
            else:
                st.markdown('<div class="sec">Model Terbaik per Program</div>', unsafe_allow_html=True)
                st.dataframe(
                    bpp.style
                       .highlight_max(subset=['R2'], color='#0a2a12')
                       .highlight_min(subset=['MAPE (%)'], color='#0a2a12')
                       .format({'R2':'{:.4f}','MAPE (%)':'{:.2f}','MAE':'{:,.0f}','RMSE':'{:,.0f}'}),
                    use_container_width=True, height=280)

                st.markdown('<div class="sec">Heatmap R² — Semua Model × Semua Program</div>', unsafe_allow_html=True)
                heat = det.pivot_table(index='Model',columns='Program',values='R2',aggfunc='mean').fillna(0)
                fig_heat = go.Figure(go.Heatmap(
                    z=heat.values, x=list(heat.columns), y=list(heat.index),
                    colorscale=[[0,'#020c18'],[0.3,'#0f2744'],[0.7,'#1d4ed8'],[1,'#38bdf8']],
                    text=[[f'{v:.3f}' for v in row] for row in heat.values],
                    texttemplate='%{text}', textfont=dict(size=10.5,color='#e2e8f0'),
                    hovertemplate='<b>%{y}</b> × <b>%{x}</b><br>R²: %{z:.4f}<extra></extra>',
                    colorbar=dict(tickfont=dict(color='#334155',size=9),title=dict(text='R²',font=dict(color='#475569',size=10)),thickness=10),
                ))
                fig_heat.update_layout(**DARK, height=420, margin=dict(t=20,b=20,l=20,r=60))
                st.plotly_chart(fig_heat, use_container_width=True)

                mc_a, mc_b = st.columns(2)
                with mc_a:
                    fig_bpp = go.Figure()
                    for i,(_, row) in enumerate(bpp.iterrows()):
                        fig_bpp.add_trace(go.Bar(
                            x=[row['Program']], y=[row['R2']], name=row['Model'],
                            marker_color=COLORS[i%len(COLORS)], marker_line_width=0,
                            text=row['Model'], textposition='outside',
                            textfont=dict(size=9,color='#64748b',family='DM Mono'),
                        ))
                    fig_bpp.add_hline(y=0.8, line_dash='dot', line_color='rgba(52,211,153,.4)')
                    styled_chart(fig_bpp, height=380, title='Model Terbaik & R²')
                    st.plotly_chart(fig_bpp, use_container_width=True)
                with mc_b:
                    fig_mape = go.Figure()
                    for i,(_, row) in enumerate(bpp.iterrows()):
                        fig_mape.add_trace(go.Bar(
                            x=[row['Program']], y=[row['MAPE (%)']], name=row['Model'],
                            marker_color=COLORS[i%len(COLORS)], marker_line_width=0,
                        ))
                    fig_mape.add_hline(y=20, line_dash='dot', line_color='rgba(251,191,36,.4)')
                    styled_chart(fig_mape, height=380, title='MAPE % per Program')
                    st.plotly_chart(fig_mape, use_container_width=True)

                with st.expander("📋 Tabel Detail Semua Model × Semua Program"):
                    st.dataframe(
                        det.sort_values(['Program','R2'],ascending=[True,False])
                           .style.format({'R2':'{:.4f}','MAPE (%)':'{:.2f}','MAE':'{:,.0f}','RMSE':'{:,.0f}'}),
                        use_container_width=True, height=400)

        with mtab3:
            conclusions = build_conclusion(ml_res, ml_res, df, target_ml, n_future)
            if conclusions:
                st.markdown('<div class="sec">Kesimpulan & Rekomendasi</div>', unsafe_allow_html=True)

                if not bpp.empty and 'MAPE (%)' in bpp.columns:
                    best_prog  = bpp.loc[bpp['MAPE (%)'].idxmin(),'Program']
                    best_mape  = bpp['MAPE (%)'].min()
                    worst_prog = bpp.loc[bpp['MAPE (%)'].idxmax(),'Program']
                    worst_mape = bpp['MAPE (%)'].max()
                    avg_m = bpp['MAPE (%)'].mean()
                    overall_grade = ("🟢 Sangat Baik" if avg_m<10 else "🔵 Baik" if avg_m<20
                                     else "🟡 Cukup" if avg_m<50 else "🔴 Perlu Data Lebih")
                    data_note = (f"Catatan: {n_yrs} tahun → metode statistik. " if n_yrs<8 else f"Data {n_yrs} tahun → ML tersedia. ")
                    st.markdown(f"""<div class="success-box">
                    🔍 <b>Auto-Insight:</b> Kualitas keseluruhan: <b>{overall_grade}</b> (Avg MAPE {avg_m:.1f}%).
                    Terbaik: <b>{best_prog}</b> ({best_mape:.1f}%) · Perlu perhatian: <b>{worst_prog}</b> ({worst_mape:.1f}%).
                    {data_note}</div>""", unsafe_allow_html=True)

                for icon, title, text in conclusions:
                    st.markdown(f"""<div class="concl-card">
                    <div class="ct">{icon} {title}</div>
                    <div class="cv">{text}</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown('<div class="sec">Radar — Profil Kualitas per Program</div>', unsafe_allow_html=True)
                if not bpp.empty and 'MAPE (%)' in bpp.columns:
                    rdf_r = bpp.dropna(subset=['MAPE (%)','MAE','RMSE']).copy()
                    rdf_r['MAPE_n']  = (1-(rdf_r['MAPE (%)']/100).clip(0,1))
                    rdf_r['MAE_n']   = 1-(rdf_r['MAE']/(rdf_r['MAE'].max()+1e-9))
                    rdf_r['RMSE_n']  = 1-(rdf_r['RMSE']/(rdf_r['RMSE'].max()+1e-9))
                    mape_med         = rdf_r['MAPE (%)'].median()
                    rdf_r['STAB_n']  = (1-np.abs(rdf_r['MAPE (%)']-mape_med)/(mape_med+1e-9)).clip(0,1)
                    cats_radar = ['Akurasi (1−MAPE)','Presisi (1−MAE)','Konsistensi (1−RMSE)','Stabilitas']
                    fig_radar = go.Figure()
                    for i, row in rdf_r.iterrows():
                        vals  = [row['MAPE_n'],row['MAE_n'],row['RMSE_n'],row['STAB_n']]
                        vals += [vals[0]]
                        col_c = COLORS[i % len(COLORS)]
                        label = f"{row['Program']} ({row['Model']})"
                        fig_radar.add_trace(go.Scatterpolar(
                            r=vals, theta=cats_radar+[cats_radar[0]],
                            fill='toself', name=label, opacity=0.6,
                            line=dict(color=col_c, width=2),
                            fillcolor=hex_to_rgba(col_c, 0.12)
                        ))
                    fig_radar.update_layout(
                        **DARK, height=520,
                        polar=dict(
                            radialaxis=dict(visible=True, range=[0,1],
                                gridcolor='#0a1a2e', tickfont=dict(color='#334155',size=9,family='DM Mono'),
                                tickvals=[0.2,0.4,0.6,0.8,1.0]),
                            angularaxis=dict(gridcolor='#0a1a2e', tickfont=dict(size=11,color='#64748b',family='Instrument Sans')),
                            bgcolor='rgba(2,12,24,0)',
                        ),
                        legend=dict(orientation='h',y=-0.15,font=dict(size=11,family='Instrument Sans'),bgcolor='rgba(0,0,0,0)'),
                        margin=dict(t=30,b=80)
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)

                st.markdown('<div class="sec">Scorecard per Program</div>', unsafe_allow_html=True)
                if n_yrs < 8:
                    st.info(f"ℹ️ **Catatan data kecil ({n_yrs} tahun):** Gunakan **MAPE** sebagai acuan utama akurasi.")

                def grade_mape(v):
                    if v is None or (isinstance(v,float) and np.isnan(v)): return "⚪ N/A"
                    return "🟢 Sangat Akurat (<10%)" if v<10 else "🔵 Akurat (10-20%)" if v<20 else "🟡 Cukup (20-50%)" if v<50 else "🔴 Tidak Akurat"

                sc_df = bpp.copy() if not bpp.empty else pd.DataFrame()
                if not sc_df.empty:
                    sc_df['Grade MAPE'] = sc_df['MAPE (%)'].apply(grade_mape)
                    cols_show = ['Program','Model','MAPE (%)','MAE','RMSE','Grade MAPE']
                    if n_yrs >= 8: cols_show = ['Program','Model','R2','MAPE (%)','MAE','RMSE','Grade MAPE']
                    fmt = {'MAPE (%)':'{:.2f}','MAE':'{:,.0f}','RMSE':'{:,.0f}'}
                    if 'R2' in cols_show: fmt['R2']='{:.4f}'
                    st.dataframe(
                        sc_df[cols_show].style.highlight_min(subset=['MAPE (%)'],color='#0a2a12').format(fmt),
                        use_container_width=True, height=280)

        # ── Sub-tab 4: Prophet ────────────────────────────────────────────────
        with mtab4:
            df_raw_m_p = st.session_state.get('raw_monthly', None)
            if not PROPHET_OK:
                st.warning("**Prophet belum terinstall.** Tambahkan `prophet` ke `requirements.txt`.")
            elif df_raw_m_p is None or len(df_raw_m_p)==0:
                st.warning("Upload dataset bulanan terlebih dahulu untuk menggunakan Prophet.")
            else:
                if 'holiday_df' not in st.session_state or st.session_state.get('holiday_df') is None:
                    with st.spinner("Mengambil kalender hari libur Indonesia dari Google Calendar API…"):
                        h_df, h_status, h_ok = build_holiday_df()
                        st.session_state['holiday_df']     = h_df
                        st.session_state['holiday_status'] = h_status
                        st.session_state['holiday_ok']     = h_ok
                else:
                    h_df     = st.session_state['holiday_df']
                    h_status = st.session_state.get('holiday_status','')
                    h_ok     = st.session_state.get('holiday_ok',False)

                n_holidays = len(h_df) if h_ok and len(h_df)>0 else 0
                n_htypes   = h_df['holiday'].nunique() if n_holidays>0 else 0

                col_hdr, col_refresh = st.columns([5,1])
                with col_refresh:
                    if st.button("🔄 Refresh", help="Ambil ulang dari Google Calendar API"):
                        fetch_google_holidays.clear()
                        for k in ('holiday_df','holiday_status','holiday_ok'):
                            st.session_state.pop(k, None)
                        st.rerun()

                if h_ok and n_holidays>0:
                    sample_names = sorted(h_df['holiday'].unique())[:10]
                    sample_str   = ' · '.join(f'<code>{n}</code>' for n in sample_names)
                    if h_df['holiday'].nunique()>10: sample_str += f' · <i>+{h_df["holiday"].nunique()-10}</i>'
                    st.markdown(f'<div class="success-box">✅ <b>Google Calendar OK.</b> {h_status}<br>🏷️ {sample_str}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="warn">{h_status}<br>Cek GCAL_KEY di Streamlit Secrets. Prophet tetap bisa jalan <b>tanpa holiday effect</b>.</div>', unsafe_allow_html=True)

                st.markdown(f'<div class="info-box">🔮 <b>Prophet</b> menangani efek hari libur secara eksplisit, trend non-stasioner, dan seasonality tahunan.<br>📅 Status kalender: {h_status}</div>', unsafe_allow_html=True)

                target_prophet = target_ml
                st.markdown(f'<div class="info-box" style="padding:10px 16px;margin-bottom:10px">🎯 <b>Target Prophet:</b> <code>{target_prophet}</code> — sinkron dengan target di atas</div>', unsafe_allow_html=True)
                n_months_prophet = st.slider("Prediksi (bulan)", 6, 36, 12, 6)
                holidays_for_prophet = h_df if (h_ok and n_holidays>0) else None
                use_holidays = st.checkbox(
                    f"Gunakan kalender hari libur Indonesia ({n_holidays} hari libur, {n_htypes} jenis)",
                    value=(n_holidays>0), disabled=(n_holidays==0))
                if not use_holidays: holidays_for_prophet = None

                cache_key = f"prophet_{target_prophet}_{n_months_prophet}"
                cached    = st.session_state.get('prophet_cache',{})
                need_run  = cache_key not in cached

                col_btn1, col_btn2 = st.columns([3,1])
                with col_btn1:
                    manual_run = st.button("🔮  Jalankan / Refresh Prophet", type="primary",
                        use_container_width=True, help="Paksa ulang training Prophet")
                with col_btn2:
                    if cache_key in cached:
                        st.markdown('<div style="color:#34d399;font-size:.78rem;padding-top:8px;font-family:\'DM Mono\',monospace;">✅ Cached</div>', unsafe_allow_html=True)

                if need_run or manual_run:
                    run_errors={}; run_results={}
                    with st.spinner(f"Melatih Prophet — {target_prophet}…"):
                        for cp in active_progs:
                            pr, pe = run_prophet(df_raw_m_p, target_prophet, cp, n_months_prophet, holidays_for_prophet)
                            if pe: run_errors[cp]=pe
                            else:  run_results[cp]=pr
                    if run_errors: st.warning("Beberapa program gagal: "+", ".join(f"{k}: {v}" for k,v in run_errors.items()))
                    if run_results:
                        cached[cache_key]={'results':run_results,'meta':{'target':target_prophet,'use_holidays':use_holidays,'n_months':n_months_prophet}}
                        st.session_state['prophet_cache']=cached

                entry=cached.get(cache_key,{}); all_p_results=entry.get('results',{})

                if all_p_results:
                    tgt_label = target_prophet
                    st.markdown(f'<div class="sec">Forecast Prophet — Semua Program ({tgt_label})</div>', unsafe_allow_html=True)

                    def hex_rgba(hex_c, alpha):
                        hex_c=hex_c.lstrip('#'); r,g,b=int(hex_c[0:2],16),int(hex_c[2:4],16),int(hex_c[4:6],16)
                        return f'rgba({r},{g},{b},{alpha})'

                    all_y_vals=[]
                    for cp,pr in all_p_results.items(): all_y_vals+=list(pr['history']['y'].dropna())
                    global_ymin = max(0, min(all_y_vals)*0.85) if all_y_vals else 0

                    fig_p_all=go.Figure(); last_hist_ds=None
                    for i,(cp,pr) in enumerate(all_p_results.items()):
                        fc_df=pr['forecast'].copy(); hist_df=pr['history'].copy()
                        col_c=COLORS[i%len(COLORS)]; cutoff=hist_df['ds'].max(); last_hist_ds=cutoff
                        fc_future=fc_df[fc_df['ds']>cutoff].copy()
                        for col in ['yhat','yhat_lower','yhat_upper']: fc_future[col]=fc_future[col].clip(lower=0)

                        fig_p_all.add_trace(go.Scatter(
                            x=hist_df['ds'], y=hist_df['y'], name=f'{cp} Aktual',
                            mode='lines+markers', legendgroup=cp,
                            line=dict(color=col_c, width=2.5, shape='spline'),
                            marker=dict(size=5),
                            fill='tozeroy', fillcolor=hex_rgba(col_c,0.04),
                            hovertemplate=f'<b>{cp}</b><br>%{{x|%b %Y}}<br>{tgt_label}: %{{y:,.0f}}<extra></extra>'))

                        fig_p_all.add_trace(go.Scatter(
                            x=list(fc_future['ds'])+list(fc_future['ds'][::-1]),
                            y=list(fc_future['yhat_upper'])+list(fc_future['yhat_lower'][::-1]),
                            fill='toself', fillcolor=hex_rgba(col_c,0.07),
                            line=dict(color='rgba(0,0,0,0)'),
                            legendgroup=cp, showlegend=False, hoverinfo='skip'))

                        fig_p_all.add_trace(go.Scatter(
                            x=fc_future['ds'], y=fc_future['yhat'],
                            name=f'{cp} Prediksi', mode='lines+markers', legendgroup=cp,
                            line=dict(color=col_c, width=2, dash='dash'),
                            marker=dict(size=7, symbol='diamond'),
                            hovertemplate=f'<b>{cp} (Prediksi)</b><br>%{{x|%b %Y}}<br>{tgt_label}: %{{y:,.0f}}<extra></extra>'))

                    if last_hist_ds is not None:
                        fig_p_all.add_vline(x=last_hist_ds.timestamp()*1000,
                            line_dash='dot', line_color='rgba(100,116,139,.3)', line_width=1.5,
                            annotation_text='← Aktual | Prediksi →',
                            annotation_font=dict(size=10,color='#475569',family='Instrument Sans'),
                            annotation_position='top')

                    fig_p_all.update_layout(**DARK, height=540, hovermode='x unified',
                        legend=dict(orientation='h',y=-0.22,font=dict(size=10.5,family='Instrument Sans'),groupclick='toggleitem',bgcolor='rgba(0,0,0,0)'),
                        margin=dict(t=30,b=130),
                        xaxis=dict(showgrid=True,gridcolor='rgba(14,30,56,.8)'),
                        yaxis=dict(showgrid=True,gridcolor='rgba(10,20,40,.9)',rangemode='nonnegative',range=[global_ymin,None]),
                        xaxis_title='Periode', yaxis_title=tgt_label)
                    st.plotly_chart(fig_p_all, use_container_width=True)

                    st.markdown('<div class="sec">Tabel Prediksi per Program</div>', unsafe_allow_html=True)
                    prog_tabs = st.tabs(list(all_p_results.keys()))
                    for tab_i,(cp,pr) in zip(prog_tabs, all_p_results.items()):
                        with tab_i:
                            fc_df=pr['forecast']; hist_df=pr['history']
                            fc_fut=fc_df[fc_df['ds']>hist_df['ds'].max()][['ds','yhat','yhat_lower','yhat_upper']].copy()
                            fc_fut.columns=['Periode','Prediksi','Batas Bawah','Batas Atas']
                            fc_fut['Periode']=fc_fut['Periode'].dt.strftime('%Y-%m')
                            for col in ['Prediksi','Batas Bawah','Batas Atas']:
                                fc_fut[col]=fc_fut[col].apply(lambda x: f"{max(0,x):,.0f}")
                            st.dataframe(fc_fut, use_container_width=True, height=320)

                    # ── Holiday Effects ───────────────────────────────────────
                    st.markdown('<div class="sec">Efek Hari Libur per Program</div>', unsafe_allow_html=True)

                    def _sanitize_prophet_name(name: str):
                        return re.sub(r'[^\w]','_',str(name))

                    def _extract_holiday_effects(pr):
                        model=pr.get('model'); hist_df=pr.get('history'); fc_df=pr.get('forecast',pd.DataFrame())
                        if model is None or hist_df is None or fc_df is None or len(fc_df)==0: return {}
                        avg_y=float(hist_df['y'].mean()) if len(hist_df)>0 else 1.0
                        s_mode=getattr(model,'seasonality_mode','additive')
                        holiday_names=[]
                        try: holiday_names=list(model.train_holiday_names)
                        except:
                            try: holiday_names=list(model.holidays['holiday'].unique())
                            except: pass
                        if not holiday_names: return {}
                        forecast_cols=set(fc_df.columns); effects={}
                        for orig_name in holiday_names:
                            col=None
                            candidates=[orig_name,re.sub(r'[^\w]','_',orig_name),orig_name.replace("'",'').replace(' ','_')]
                            for c in candidates:
                                if c in forecast_cols and not c.endswith('_lower') and not c.endswith('_upper'): col=c; break
                            if col is None: continue
                            col_vals=fc_df[col]; active_mask=col_vals.abs()>1e-9
                            if active_mask.sum()==0: continue
                            raw_eff=float(col_vals[active_mask].mean())
                            pct=raw_eff*100.0 if s_mode=='multiplicative' else (raw_eff/(avg_y+1e-9))*100.0
                            effects[orig_name]=pct
                        return effects

                    def _cat_holiday(name):
                        nl=name.lower()
                        if any(k in nl for k in ['idul fitri','lebaran','eid al-fitr']): return 'Idul Fitri'
                        if any(k in nl for k in ['idul adha','eid al-adha']): return 'Idul Adha'
                        if any(k in nl for k in ['ramad','puasa']): return 'Ramadhan'
                        if any(k in nl for k in ['natal','christmas']): return 'Natal'
                        if any(k in nl for k in ["new year's",'tahun baru masehi']) and not any(x in nl for x in ['islam','imlek','chinese','hijri','lunar']): return 'Tahun Baru'
                        if any(k in nl for k in ['imlek','chinese new year','lunar new year']): return 'Imlek'
                        if any(k in nl for k in ['nyepi','day of silence']): return 'Nyepi'
                        if any(k in nl for k in ["isra","mi'raj",'miraj']): return "Isra Mi'raj"
                        if any(k in nl for k in ['waisak','vesak']): return 'Waisak'
                        if any(k in nl for k in ['good friday','wafat','easter','paskah']): return 'Paskah/Wafat'
                        if any(k in nl for k in ['maulid','mawlid']): return 'Maulid Nabi'
                        if any(k in nl for k in ['muharram','islamic new year']): return 'Tahun Baru Islam'
                        if any(k in nl for k in ['buruh','labor day','may day']): return 'Hari Buruh'
                        if any(k in nl for k in ['pancasila']): return 'Hari Pancasila'
                        if any(k in nl for k in ['kemerdekaan','independence']): return 'HUT RI'
                        if any(k in nl for k in ['cuti bersama']): return 'Cuti Bersama'
                        if any(k in nl for k in ['election','pemilu']): return 'Pemilu'
                        if any(k in nl for k in ['tahun baru','new year']) and not any(x in nl for x in ['islam','imlek','chinese']): return 'Tahun Baru'
                        return name[:30]

                    heff_rows=[]
                    for cp,pr in all_p_results.items():
                        hist_df_p=pr.get('history',pd.DataFrame())
                        avg_y_p=float(hist_df_p['y'].mean()) if len(hist_df_p)>0 else 1.0
                        eff_map=_extract_holiday_effects(pr)
                        for hname,pct in eff_map.items():
                            heff_rows.append({'Program':cp,'Holiday':hname,'Efek_pct':pct,'avg_y':avg_y_p})

                    if not heff_rows:
                        if not use_holidays: st.info("Holiday effect tidak aktif. Aktifkan checkbox kalender.")
                        elif n_holidays==0: st.markdown('<div class="warn">⚠️ Tidak ada data hari libur. Pastikan GCAL_KEY di Streamlit Secrets.</div>', unsafe_allow_html=True)
                        else: st.info("Efek hari libur tidak terdeteksi. Data bulanan minimal 12–24 bulan.")
                    else:
                        heff_df=pd.DataFrame(heff_rows)
                        heff_df['Kategori']=heff_df['Holiday'].apply(_cat_holiday)
                        heff_grp=(heff_df.groupby(['Program','Kategori'])
                                  .agg(Efek_pct=('Efek_pct','mean'),avg_y=('avg_y','first'),n_events=('Efek_pct','count'))
                                  .reset_index())
                        top_cats=(heff_grp.groupby('Kategori')['Efek_pct'].apply(lambda x:x.abs().mean())
                                  .nlargest(12).index.tolist())
                        heff_grp=heff_grp[heff_grp['Kategori'].isin(top_cats)].copy()
                        programs_list=sorted(heff_grp['Program'].unique())
                        max_abs_pct=heff_grp['Efek_pct'].abs().max()

                        if len(heff_grp)==0 or max_abs_pct<0.01:
                            st.markdown('<div class="warn">⚠️ Efek sangat kecil. Tambah data bulanan minimal 24 bulan.</div>', unsafe_allow_html=True)
                        else:
                            n_cats_shown=len(top_cats); n_total_hols=heff_df['Kategori'].nunique()
                            st.markdown(f'<div class="info-box">📊 Menampilkan <b>{n_cats_shown} jenis hari libur</b> (dari {n_total_hols} total). Efek diekstrak dari parameter posterior Prophet.</div>', unsafe_allow_html=True)

                            cat_order=(heff_grp.groupby('Kategori')['Efek_pct'].apply(lambda x:x.abs().mean())
                                       .sort_values(ascending=False).index.tolist())

                            # ── Clamp outliers for readability, keep full in hover ──
                            P95 = float(heff_grp['Efek_pct'].abs().quantile(0.90))
                            clamp_limit = max(P95 * 1.5, 80.0)   # show full range if small
                            heff_clamped = heff_grp.copy()
                            heff_clamped['Efek_display'] = heff_clamped['Efek_pct'].clip(-clamp_limit, clamp_limit)
                            n_outliers = int((heff_grp['Efek_pct'].abs() > clamp_limit).sum())

                            # ── 1. ANNOTATED HEATMAP (primary – best readability) ────────
                            st.markdown('<div class="sec">Heatmap Efek Hari Libur per Program</div>', unsafe_allow_html=True)
                            hm_pivot = (heff_grp.pivot_table(index='Kategori', columns='Program',
                                         values='Efek_pct', aggfunc='mean')
                                        .reindex(cat_order).fillna(0))
                            zmax = float(hm_pivot.abs().quantile(0.90).max())
                            zmax = max(zmax, 5.0)

                            # Build cell text with color-coded arrows
                            cell_text = []
                            for row in hm_pivot.values:
                                cell_text.append([
                                    f"{'▲' if v>0.5 else ('▼' if v<-0.5 else '–')} {abs(v):.0f}%"
                                    for v in row
                                ])

                            fig_hm2 = go.Figure(go.Heatmap(
                                z=hm_pivot.values,
                                x=list(hm_pivot.columns),
                                y=list(hm_pivot.index),
                                colorscale=[
                                    [0,   '#7f1d1d'],
                                    [0.2, '#dc2626'],
                                    [0.45,'#1a0505'],
                                    [0.5, '#061120'],
                                    [0.55,'#052e16'],
                                    [0.8, '#16a34a'],
                                    [1,   '#14532d'],
                                ],
                                zmid=0, zmin=-zmax*1.1, zmax=zmax*1.1,
                                text=cell_text,
                                texttemplate='%{text}',
                                textfont=dict(size=11, color='rgba(255,255,255,0.9)', family='DM Mono'),
                                hovertemplate='<b>%{y}</b> — <b>%{x}</b><br>Efek: <b>%{z:+.2f}%</b><br><i style="color:#475569">merah=turun · hijau=naik</i><extra></extra>',
                                colorbar=dict(
                                    ticksuffix='%',
                                    tickfont=dict(color='#64748b', size=9, family='DM Mono'),
                                    thickness=10,
                                    title=dict(text='Efek', font=dict(color='#475569', size=10)),
                                    len=0.7,
                                ),
                            ))
                            row_h = max(38, min(60, 700 // max(len(cat_order), 1)))
                            fig_hm2.update_layout(
                                **DARK,
                                height=max(380, len(cat_order)*row_h + 120),
                                margin=dict(t=20, b=50, l=175, r=80),
                                xaxis=dict(
                                    tickfont=dict(size=12, color='#7dd3fc', family='Instrument Sans'),
                                    side='top',
                                ),
                                yaxis=dict(
                                    categoryorder='array', categoryarray=cat_order[::-1],
                                    tickfont=dict(size=11, color='#e2e8f0', family='Instrument Sans'),
                                ),
                            )
                            st.plotly_chart(fig_hm2, use_container_width=True)

                            # ── 2. DOT PLOT — per program, clamped, sorted ────────────
                            st.markdown('<div class="sec">Dot Plot — Efek per Hari Libur & Program</div>', unsafe_allow_html=True)
                            if n_outliers > 0:
                                st.markdown(
                                    f'<div class="info-box">⚠️ <b>{n_outliers} nilai</b> dipotong di ±{clamp_limit:.0f}% untuk keterbacaan. '
                                    f'Hover untuk melihat nilai asli.</div>', unsafe_allow_html=True)

                            fig_dot = go.Figure()
                            # Alternating row bands for readability
                            for ri, cat in enumerate(cat_order):
                                if ri % 2 == 0:
                                    fig_dot.add_hrect(
                                        y0=ri-0.5, y1=ri+0.5,
                                        fillcolor='rgba(255,255,255,0.02)', line_width=0, layer='below',
                                    )

                            for pi, prog in enumerate(programs_list):
                                pdata = heff_grp[heff_grp['Program']==prog].set_index('Kategori')
                                pdata_c = heff_clamped[heff_clamped['Program']==prog].set_index('Kategori')
                                col = COLORS[pi % len(COLORS)]
                                x_disp, x_real, y_cats = [], [], []
                                for cat in cat_order:
                                    if cat in pdata.index:
                                        real_v = float(pdata.loc[cat,'Efek_pct'])
                                        disp_v = float(pdata_c.loc[cat,'Efek_display'])
                                        x_disp.append(disp_v)
                                        x_real.append(real_v)
                                        y_cats.append(cat)
                                    else:
                                        x_disp.append(None)
                                        x_real.append(None)
                                        y_cats.append(cat)

                                # Connecting lines (zero → dot)
                                for yi, (xd, xr) in enumerate(zip(x_disp, x_real)):
                                    if xd is not None:
                                        line_col = 'rgba(22,163,74,0.27)' if xd >= 0 else 'rgba(220,38,38,0.27)'
                                        fig_dot.add_shape(
                                            type='line',
                                            x0=0, x1=xd, y0=y_cats[yi], y1=y_cats[yi],
                                            line=dict(color=line_col, width=1.5),
                                            layer='below',
                                        )

                                hover_texts = [
                                    f'<b>{prog}</b><br>{cat}<br>'
                                    f'Efek: <b style="color:{"#34d399" if (xr or 0)>=0 else "#f87171"}">{(xr or 0):+.2f}%</b>'
                                    + (f'<br><i style="color:#64748b">ditampilkan: {(xd or 0):+.1f}%</i>'
                                       if xr is not None and abs((xr or 0)) > clamp_limit else '')
                                    for cat, xd, xr in zip(y_cats, x_disp, x_real)
                                ]

                                fig_dot.add_trace(go.Scatter(
                                    name=prog,
                                    x=x_disp, y=y_cats,
                                    mode='markers',
                                    marker=dict(
                                        color=[col if (v or 0)>=0 else col for v in x_disp],
                                        size=11,
                                        symbol=['circle' if (v or 0)>=0 else 'diamond' for v in x_disp],
                                        line=dict(color='rgba(255,255,255,0.2)', width=1),
                                        opacity=0.92,
                                    ),
                                    text=hover_texts,
                                    hovertemplate='%{text}<extra></extra>',
                                    hoverlabel=dict(bgcolor='rgba(6,18,36,.96)', font_size=12, font_family='DM Mono'),
                                ))

                            fig_dot.add_vline(x=0, line_color='rgba(56,189,248,0.25)', line_width=1.5, line_dash='dot')
                            # Clamp indicator lines
                            if n_outliers > 0:
                                for sign in [1, -1]:
                                    fig_dot.add_vline(
                                        x=sign*clamp_limit,
                                        line_color='rgba(251,146,60,0.3)', line_width=1, line_dash='dash',
                                    )

                            dot_h = max(420, len(cat_order)*32 + 140)
                            fig_dot.update_layout(
                                **DARK,
                                height=dot_h,
                                hovermode='closest',
                                xaxis=dict(
                                    range=[-(clamp_limit*1.15), clamp_limit*1.15],
                                    ticksuffix='%',
                                    tickfont=dict(size=10, color='#475569', family='DM Mono'),
                                    title=dict(text='Efek terhadap rata-rata klaim (%)', font=dict(size=11, color='#475569')),
                                    showgrid=True, gridcolor='rgba(14,30,56,0.8)',
                                    zeroline=False,
                                ),
                                yaxis=dict(
                                    categoryorder='array', categoryarray=cat_order,
                                    tickfont=dict(size=11, color='#e2e8f0', family='Instrument Sans'),
                                    showgrid=True, gridcolor='rgba(10,20,40,0.6)',
                                ),
                                legend=dict(
                                    orientation='h', y=-0.1,
                                    font=dict(size=11, family='Instrument Sans'),
                                    bgcolor='rgba(0,0,0,0)',
                                    itemsizing='constant',
                                ),
                                margin=dict(t=20, b=80, l=175, r=30),
                                annotations=[
                                    dict(x=-clamp_limit*0.6, y=len(cat_order)-0.3,
                                         text='◀ klaim turun',
                                         showarrow=False, font=dict(size=10, color='#7f1d1d', family='Instrument Sans')),
                                    dict(x=clamp_limit*0.6, y=len(cat_order)-0.3,
                                         text='klaim naik ▶',
                                         showarrow=False, font=dict(size=10, color='#14532d', family='Instrument Sans')),
                                ],
                            )
                            st.plotly_chart(fig_dot, use_container_width=True)

                            # Cards
                            st.markdown('<div class="sec">Ringkasan Efek per Program</div>', unsafe_allow_html=True)
                            card_cols=st.columns(len(programs_list))
                            for ci,prog in enumerate(programs_list):
                                prog_data=heff_grp[heff_grp['Program']==prog].copy()
                                avg_y_p=float(prog_data['avg_y'].iloc[0]) if len(prog_data)>0 else 1.0
                                col_c=COLORS[ci%len(COLORS)]
                                pos_data=prog_data[prog_data['Efek_pct']>0.1].sort_values('Efek_pct',ascending=False)
                                neg_data=prog_data[prog_data['Efek_pct']<-0.1].sort_values('Efek_pct')
                                net_prog=float(prog_data['Efek_pct'].sum())

                                def _pill_p(v):
                                    return (f'<span style="color:#34d399;font-weight:700;font-family:\'DM Mono\',monospace">{v:+.1f}%</span>'
                                            if v>0 else f'<span style="color:#f87171;font-weight:700;font-family:\'DM Mono\',monospace">{v:+.1f}%</span>')
                                def _dk(v):
                                    dk=abs(v/100.0*avg_y_p)
                                    return f'<span style="color:#334155;font-size:.68rem;font-family:\'DM Mono\',monospace"> (~{dk:,.0f})</span>' if dk>=1 else ''

                                up_html=''.join(
                                    f'<div style="display:flex;justify-content:space-between;align-items:center;margin:5px 0;font-size:.78rem;gap:4px;">'
                                    f'<span><span style="color:#34d399;margin-right:4px">▲</span><span style="color:#cbd5e1;font-family:\'Instrument Sans\',sans-serif">{row.Kategori}</span>{_dk(row.Efek_pct)}</span>'
                                    f'{_pill_p(row.Efek_pct)}</div>'
                                    for row in pos_data.head(3).itertuples()
                                ) if len(pos_data)>0 else '<div style="color:#1e3a5f;font-size:.78rem;font-style:italic">—</div>'

                                dn_html=''.join(
                                    f'<div style="display:flex;justify-content:space-between;align-items:center;margin:5px 0;font-size:.78rem;gap:4px;">'
                                    f'<span><span style="color:#f87171;margin-right:4px">▼</span><span style="color:#cbd5e1;font-family:\'Instrument Sans\',sans-serif">{row.Kategori}</span>{_dk(row.Efek_pct)}</span>'
                                    f'{_pill_p(row.Efek_pct)}</div>'
                                    for row in neg_data.head(3).itertuples()
                                ) if len(neg_data)>0 else '<div style="color:#1e3a5f;font-size:.78rem;font-style:italic">—</div>'

                                if abs(net_prog)<0.5: badge=f'<span style="background:#0a1a2e;color:#475569;padding:2px 8px;border-radius:6px;font-size:.7rem;font-family:\'DM Mono\',monospace">Netral</span>'
                                elif net_prog>0: badge=f'<span style="background:#052e16;color:#34d399;padding:2px 8px;border-radius:6px;font-size:.7rem;font-family:\'DM Mono\',monospace">Net +{net_prog:.1f}%</span>'
                                else: badge=f'<span style="background:#1a0505;color:#f87171;padding:2px 8px;border-radius:6px;font-size:.7rem;font-family:\'DM Mono\',monospace">Net {net_prog:.1f}%</span>'

                                with card_cols[ci]:
                                    st.markdown(f'''
                                    <div style="background:#061120;border:1px solid {col_c}30;
                                    border-top:2px solid {col_c};border-radius:14px;padding:16px 18px;">
                                      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
                                        <span style="font-size:.62rem;color:#334155;font-weight:700;text-transform:uppercase;letter-spacing:1.5px;font-family:\'Instrument Sans\',sans-serif;">{prog}</span>
                                        {badge}
                                      </div>
                                      <div style="font-size:.58rem;color:#1e3a5f;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:6px;font-family:\'Instrument Sans\',sans-serif;">📈 Klaim Naik Saat</div>
                                      {up_html}
                                      <div style="border-top:1px solid #0a1a2e;margin:10px 0;"></div>
                                      <div style="font-size:.58rem;color:#1e3a5f;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:6px;font-family:\'Instrument Sans\',sans-serif;">📉 Klaim Turun Saat</div>
                                      {dn_html}
                                    </div>''', unsafe_allow_html=True)

                            with st.expander("📋 Tabel Detail Efek Semua Holiday × Program"):
                                detail_tbl=(heff_grp.copy()
                                            .sort_values(['Kategori','Efek_pct'],ascending=[True,False])
                                            .rename(columns={'Kategori':'Hari Libur','Efek_pct':'Efek (%)','n_events':'Jumlah Event'}))
                                detail_tbl['Efek (%)']=detail_tbl['Efek (%)'].round(2)
                                detail_tbl['Arah']=detail_tbl['Efek (%)'].apply(lambda v:'▲ Naik' if v>0.1 else ('▼ Turun' if v<-0.1 else '– Netral'))
                                st.dataframe(
                                    detail_tbl[['Program','Hari Libur','Efek (%)','Arah','Jumlah Event']]
                                    .style.format({'Efek (%)':'{:+.2f}%'}),
                                    use_container_width=True, height=360)

                            st.markdown('<div class="info-box">📊 <b>Cara baca:</b> Efek = % perubahan klaim vs rata-rata bulan normal. Efek diekstrak dari parameter posterior Prophet. Nama hari libur dari Google Calendar API Indonesia.</div>', unsafe_allow_html=True)

    else:
        st.info("Klik **Jalankan Analisis ML** untuk memulai.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PREDIKSI
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    p1, _ = st.columns([1, 3])
    with p1:
        target_pred = st.selectbox("Target", targets, key='pred_tgt')
        run_pred    = st.button("🔮  Hitung Prediksi", type="primary", use_container_width=True)

    ck_p    = f"{target_pred}_lags{n_lags}_test{test_pct}"
    ml_pred = results_cache.get(ck_p)

    if run_pred:
        if ml_pred is None:
            with st.spinner("Melatih model…"):
                ml_pred, err = run_ml(df, target_pred, n_lags, test_pct/100)
            if err: st.error(err); ml_pred=None
            else:
                results_cache[ck_p]=ml_pred
                st.session_state.active_results=results_cache

    fut=None; fut_monthly=None
    if ml_pred:
        with st.spinner("Menghitung proyeksi…"):
            fut = forecast(df, ml_pred, n_future)
        df_raw_m = st.session_state.get('raw_monthly', None)
        if df_raw_m is not None and len(df_raw_m)>0:
            if target_pred in df_raw_m.columns:
                try: fut_monthly=compute_monthly_breakdown(df_raw_m, fut, target_pred)
                except Exception as e: st.warning(f"Gagal hitung prediksi bulanan: {e}"); fut_monthly=None

        st.session_state['last_forecast']=fut
        st.session_state['last_forecast_monthly']=fut_monthly
        st.session_state[f'forecast_{target_pred}']=fut
        st.session_state[f'forecast_monthly_{target_pred}']=fut_monthly
        st.session_state[f'forecast_annual_{target_pred}']=fut

        data_hash=hashlib.md5(df.to_csv().encode()).hexdigest()[:8]
        eid_pred=f"{data_hash}_{target_pred}_L{n_lags}_T{test_pct}"
        label_pred=f"📁 {datetime.now().strftime('%d/%m %H:%M')} | {target_pred} | {len(years)}yr | {len(active_progs)} prog"
        extra_snapshot={k:st.session_state[k] for k in
            ['raw_monthly','forecast_Kasus','forecast_Nominal','forecast_monthly_Kasus',
             'forecast_monthly_Nominal','forecast_annual_Kasus','forecast_annual_Nominal',
             'last_forecast','last_forecast_monthly']
            if k in st.session_state and st.session_state[k] is not None}
        add_to_history(label_pred, eid_pred, df.copy(), dict(results_cache), extra_snapshot)

        future_yrs=sorted(fut['Tahun'].unique()); yr_range=f"{future_yrs[0]}-{future_yrs[-1]}"
        per_prog_info=ml_pred.get('per_prog',{})
        if per_prog_info:
            model_parts=" | ".join(f"<b>{cat}</b>→{info['best_name']}" for cat,info in per_prog_info.items())
            st.markdown(f'<div class="badge">🎯 <b>Per-Program:</b> {model_parts} &nbsp;|&nbsp; Proyeksi <b>{n_future} tahun</b> ({yr_range})</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="badge">Model: <b>{ml_pred["best_name"]}</b> &nbsp;|&nbsp; Proyeksi <b>{n_future} tahun</b> ({yr_range})</div>', unsafe_allow_html=True)

        ptab_yr, ptab_mo = st.tabs(["📅  Prediksi Tahunan", "📆  Prediksi Bulanan"])

        with ptab_yr:
            hist = df_plot.groupby(['Tahun','Kategori'])[target_pred].sum().reset_index()
            fut_yr = fut.copy(); fut_yr['Jenis']='Prediksi'

            st.markdown('<div class="sec">Tren Aktual vs Prediksi</div>', unsafe_allow_html=True)
            fig_main=go.Figure()
            cat_color={c:COLORS[i%len(COLORS)] for i,c in enumerate(sorted(hist['Kategori'].unique()))}

            for cat in sorted(hist['Kategori'].unique()):
                col=cat_color[cat]
                h=hist[hist['Kategori']==cat].sort_values('Tahun')
                if len(h):
                    fig_main.add_trace(go.Scatter(
                        x=h['Tahun'], y=h[target_pred], name=cat+" (Aktual)",
                        mode='lines+markers', line=dict(color=col, width=2.5, shape='spline'),
                        marker=dict(size=9, color=col, line=dict(color='#020c18',width=2.5)),
                        legendgroup=cat,
                        fill='tozeroy', fillcolor=hex_to_rgba(col, 0.05),
                    ))
                p=fut_yr[fut_yr['Kategori']==cat].sort_values('Tahun')
                if len(p):
                    x_p=list(p['Tahun']); y_p=list(p[target_pred])
                    if len(h):
                        last_h=h.sort_values('Tahun').iloc[-1]
                        x_p=[int(last_h['Tahun'])]+x_p; y_p=[float(last_h[target_pred])]+y_p
                    fig_main.add_trace(go.Scatter(
                        x=x_p, y=y_p, name=cat+" (Prediksi)",
                        mode='lines+markers', line=dict(color=col, width=2.5, dash='dash'),
                        marker=dict(size=10, color=col, symbol='diamond', line=dict(color='#020c18',width=2)),
                        legendgroup=cat,
                    ))

            future_yrs2=sorted(fut['Tahun'].unique())
            fig_main.add_vrect(x0=latest_year+0.3, x1=future_yrs2[-1]+0.7,
                fillcolor='rgba(59,130,246,0.04)', line_width=0,
                annotation_text="▶ Zona Prediksi",
                annotation_position="top left",
                annotation_font=dict(color='#334155',size=10,family='Instrument Sans'))
            fig_main.add_vline(x=latest_year+0.5, line_dash='dot',
                line_color='rgba(59,130,246,.3)', line_width=1.5,
                annotation_text=f"← {latest_year} | {future_yrs2[0]} →",
                annotation_font=dict(color='#334155',size=9,family='DM Mono'),
                annotation_position="bottom right")

            styled_chart(fig_main, height=560, legend_bottom=True, margin_b=150)
            fig_main.update_layout(xaxis=dict(dtick=1), yaxis_title=target_pred, xaxis_title='Tahun')
            st.plotly_chart(fig_main, use_container_width=True)

            if per_prog_info:
                mape_grade_fn = lambda m: 'mpill-green' if m and m<10 else 'mpill-blue' if m and m<20 else 'mpill-yellow' if m and m<50 else 'mpill-red'
                cards_html='<div style="display:flex;flex-wrap:wrap;gap:10px;margin:14px 0;">'
                for cat,info in per_prog_info.items():
                    mape_v=info.get('metrics',{}).get('MAPE (%)',None)
                    pill_cls=mape_grade_fn(mape_v)
                    mape_txt=f"MAPE: {mape_v:.1f}%" if mape_v is not None else ""
                    cards_html+=(f'<div class="prog-card">'
                                 f'<div class="pc-name">{cat}</div>'
                                 f'<div class="pc-model">{info["best_name"]}</div>'
                                 f'<span class="mpill {pill_cls}">{mape_txt}</span>'
                                 f'</div>')
                cards_html+='</div>'
                st.markdown(cards_html, unsafe_allow_html=True)

            st.markdown('<div class="sec">Nilai Prediksi per Tahun per Program</div>', unsafe_allow_html=True)
            fig_bar=go.Figure()
            for i,cat in enumerate(sorted(fut['Kategori'].unique())):
                cd=fut[fut['Kategori']==cat].sort_values('Tahun')
                fig_bar.add_trace(go.Bar(
                    x=cd['Tahun'], y=cd[target_pred], name=cat,
                    marker=dict(color=COLORS[i%len(COLORS)], line=dict(width=0)),
                    text=cd[target_pred].apply(lambda v: f'{v:,.0f}'),
                    textposition='outside', textfont=dict(size=9,color='#64748b',family='DM Mono'),
                ))
            fig_bar.update_layout(barmode='group')
            styled_chart(fig_bar, height=420)
            fig_bar.update_layout(xaxis=dict(dtick=1))
            st.plotly_chart(fig_bar, use_container_width=True)

            if len(future_yrs2)>=2:
                st.markdown('<div class="sec">Waterfall — Total Klaim Prediksi per Tahun</div>', unsafe_allow_html=True)
                tot_by_yr=fut.groupby('Tahun')[target_pred].sum().reset_index()
                tot_by_yr['delta']=tot_by_yr[target_pred].diff().fillna(0)
                measures=['absolute']+['relative']*(len(tot_by_yr)-1)
                wf_y_vals=[tot_by_yr[target_pred].iloc[0]]+tot_by_yr['delta'].iloc[1:].tolist()
                fig_wf=go.Figure(go.Waterfall(
                    orientation='v', measure=measures,
                    x=[str(y) for y in tot_by_yr['Tahun'].tolist()], y=wf_y_vals,
                    textposition='outside',
                    text=[f"{v:+,.0f}" if i>0 else f"{v:,.0f}" for i,v in enumerate(wf_y_vals)],
                    connector=dict(line=dict(color='#0a1a2e',width=1.5)),
                    increasing=dict(marker_color='#34d399'),
                    decreasing=dict(marker_color='#f87171'),
                    totals=dict(marker_color='#38bdf8')
                ))
                fig_wf.update_layout(**DARK, height=360, showlegend=False,
                    yaxis_title=target_pred, margin=dict(t=20,b=40,l=60,r=20))
                st.plotly_chart(fig_wf, use_container_width=True)

            st.markdown('<div class="sec">Distribusi per Tahun Prediksi</div>', unsafe_allow_html=True)
            ncols=min(len(future_yrs2),3)
            pcols=st.columns(ncols)
            for i,fy in enumerate(future_yrs2):
                with pcols[i%ncols]:
                    fy_d=fut[fut['Tahun']==fy]
                    fp=go.Figure(go.Pie(
                        labels=fy_d['Kategori'], values=fy_d[target_pred], hole=0.55,
                        textinfo='label+percent', textposition='outside',
                        marker=dict(colors=COLORS[:len(fy_d)], line=dict(color='#020c18',width=2)),
                    ))
                    fp.update_layout(**DARK, showlegend=False, height=320,
                        title=dict(text=str(fy),font=dict(size=14,color='#64748b',family='Syne'),x=0.5),
                        margin=dict(t=40,b=10,l=10,r=10))
                    st.plotly_chart(fp, use_container_width=True)

            if len(future_yrs2)>1:
                st.markdown('<div class="sec">Estimasi Pertumbuhan Total (%)</div>', unsafe_allow_html=True)
                grow=[]
                for cat in ml_pred['active_programs']:
                    cd=fut[fut['Kategori']==cat][target_pred].values
                    if len(cd)>=2: grow.append({'Kategori':cat,'Pertumbuhan (%)':round((cd[-1]/(cd[0]+1e-9)-1)*100,2)})
                if grow:
                    gdf=pd.DataFrame(grow).sort_values('Pertumbuhan (%)',ascending=True)
                    fig_g=go.Figure()
                    for i,(_, row) in enumerate(gdf.iterrows()):
                        fig_g.add_trace(go.Bar(
                            x=[row['Pertumbuhan (%)']], y=[row['Kategori']], orientation='h',
                            name=row['Kategori'], showlegend=False,
                            marker=dict(color='#34d399' if row['Pertumbuhan (%)']>=0 else '#f87171', line=dict(width=0)),
                            text=f"{row['Pertumbuhan (%)']:+.1f}%",
                            textposition='outside', textfont=dict(size=10,color='#64748b',family='DM Mono'),
                        ))
                    fig_g.add_vline(x=0, line_color='rgba(30,58,95,.8)', line_width=1.5)
                    styled_chart(fig_g, height=max(280, len(active_progs)*50+80), legend_bottom=False)
                    fig_g.update_layout(margin=dict(l=10,t=10,b=10,r=100))
                    st.plotly_chart(fig_g, use_container_width=True)

            st.markdown('<div class="sec">Tabel Prediksi Tahunan</div>', unsafe_allow_html=True)
            tbl=fut[['Kategori','Tahun',target_pred]].copy().sort_values(['Kategori','Tahun'])
            fmt='Rp {:,.0f}' if target_pred=='Nominal' else '{:,.0f}'
            tbl[target_pred]=tbl[target_pred].apply(lambda x: fmt.format(x))
            st.dataframe(tbl, use_container_width=True)

        with ptab_mo:
            if fut_monthly is not None and len(fut_monthly)>0:
                st.markdown('<div class="info-box">Prediksi bulanan dihitung dengan mendistribusikan total tahunan menggunakan pola musiman historis.</div>', unsafe_allow_html=True)

                st.markdown('<div class="sec">Tren Bulanan: Aktual vs Prediksi</div>', unsafe_allow_html=True)
                df_raw_m2=st.session_state.get('raw_monthly',None)
                fig_mo=go.Figure()
                for i,cat in enumerate(sorted(ml_pred['active_programs'])):
                    col=COLORS[i%len(COLORS)]
                    if df_raw_m2 is not None and target_pred in df_raw_m2.columns:
                        cat_raw=(df_raw_m2[df_raw_m2['Kategori']==cat].sort_values(['Tahun','Bulan']).copy())
                        if len(cat_raw):
                            cat_raw['Periode']=cat_raw['Tahun'].astype(str)+'-'+cat_raw['Bulan'].astype(str).str.zfill(2)
                            fig_mo.add_trace(go.Scatter(x=cat_raw['Periode'],y=cat_raw[target_pred],
                                name=cat+" (Aktual)",mode='lines+markers',legendgroup=cat,
                                line=dict(color=col,width=2),marker=dict(size=5)))
                    cat_pred_mo=(fut_monthly[fut_monthly['Kategori']==cat].sort_values(['Tahun','Bulan']))
                    if len(cat_pred_mo):
                        fig_mo.add_trace(go.Scatter(x=cat_pred_mo['Periode'],y=cat_pred_mo[target_pred],
                            name=cat+" (Prediksi)",mode='lines+markers',legendgroup=cat,
                            line=dict(color=col,width=2,dash='dash'),marker=dict(size=7,symbol='diamond')))
                fig_mo.update_layout(**DARK, height=540, hovermode='x unified',
                    xaxis_tickangle=-45,
                    legend=dict(orientation='h',y=-0.35,groupclick='toggleitem',font=dict(size=11,family='Instrument Sans'),bgcolor='rgba(0,0,0,0)'),
                    margin=dict(b=150,t=20,l=70,r=20),
                    yaxis_title=target_pred, xaxis_title='Periode (YYYY-MM)',
                    xaxis=dict(showgrid=True,gridcolor='rgba(14,30,56,.8)'),
                    yaxis=dict(showgrid=True,gridcolor='rgba(10,20,40,.9)'))
                st.plotly_chart(fig_mo, use_container_width=True)

                st.markdown('<div class="sec">Detail Prediksi Bulanan per Program</div>', unsafe_allow_html=True)
                sel_cat=st.selectbox("Pilih Program", sorted(ml_pred['active_programs']), key='mo_cat_sel')
                cat_mo=fut_monthly[fut_monthly['Kategori']==sel_cat].sort_values(['Tahun','Bulan'])
                fig_cat=go.Figure()
                col_sel=COLORS[sorted(ml_pred['active_programs']).index(sel_cat)%len(COLORS)]
                if df_raw_m2 is not None and target_pred in df_raw_m2.columns:
                    h_mo=(df_raw_m2[df_raw_m2['Kategori']==sel_cat].sort_values(['Tahun','Bulan']).copy())
                    if len(h_mo):
                        h_mo['Periode']=h_mo['Tahun'].astype(str)+'-'+h_mo['Bulan'].astype(str).str.zfill(2)
                        fig_cat.add_trace(go.Bar(x=h_mo['Periode'],y=h_mo[target_pred],
                            name='Aktual',marker=dict(color=col_sel,opacity=0.6,line=dict(width=0))))
                fig_cat.add_trace(go.Bar(x=cat_mo['Periode'],y=cat_mo[target_pred],
                    name='Prediksi',marker=dict(color=col_sel,opacity=1,line=dict(width=0),
                    pattern=dict(shape='/'))))
                fig_cat.update_layout(**DARK, height=420, barmode='group', xaxis_tickangle=-45,
                    legend=dict(orientation='h',bgcolor='rgba(0,0,0,0)'),
                    margin=dict(b=100,t=20),
                    title=dict(text=f"{sel_cat} — Aktual & Prediksi Bulanan",font=dict(size=12,color='#64748b'),x=0))
                st.plotly_chart(fig_cat, use_container_width=True)

                st.markdown('<div class="sec">Tabel Prediksi Bulanan Lengkap</div>', unsafe_allow_html=True)
                tbl_mo=(fut_monthly[['Kategori','Tahun','Bulan','Periode',target_pred]]
                        .copy().sort_values(['Kategori','Tahun','Bulan']))
                fmt='Rp {:,.0f}' if target_pred=='Nominal' else '{:,.0f}'
                tbl_mo[target_pred]=tbl_mo[target_pred].apply(lambda x: fmt.format(x))
                st.dataframe(tbl_mo, use_container_width=True, height=420)
            else:
                df_raw_debug=st.session_state.get('raw_monthly',None)
                if df_raw_debug is None: st.warning("Data bulanan belum tersimpan. Upload ulang file dataset.")
                elif target_pred not in df_raw_debug.columns: st.warning(f"Kolom {target_pred} tidak ditemukan di data bulanan.")
                else: st.warning("Terjadi kesalahan. Coba klik Hitung Prediksi ulang.")
    else:
        st.info("Klik **Hitung Prediksi** — model ML akan dilatih otomatis jika belum ada.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — EXPORT
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="sec">Export Laporan</div>', unsafe_allow_html=True)
    ec1, ec2 = st.columns(2)

    best_ml         = next((v for v in results_cache.values() if v), None)
    last_fut        = st.session_state.get('last_forecast', None)
    fut_ann_kasus   = st.session_state.get('forecast_annual_Kasus',   None)
    fut_ann_nominal = st.session_state.get('forecast_annual_Nominal', None)
    fut_mo_kasus    = st.session_state.get('forecast_monthly_Kasus',   None)
    fut_mo_nominal  = st.session_state.get('forecast_monthly_Nominal', None)

    if fut_ann_kasus is None and fut_ann_nominal is None and last_fut is not None:
        tc_lf=[c for c in last_fut.columns if c not in ['Kategori','Tahun','Type']]
        if tc_lf:
            if tc_lf[0]=='Kasus': fut_ann_kasus=last_fut
            else: fut_ann_nominal=last_fut

    if fut_mo_kasus is None and fut_mo_nominal is None:
        lm=st.session_state.get('last_forecast_monthly',None)
        if lm is not None and len(lm)>0:
            tc_lm=[c for c in lm.columns if c not in ['Kategori','Tahun','Bulan','Periode','Type']]
            if tc_lm:
                if tc_lm[0]=='Kasus': fut_mo_kasus=lm
                else: fut_mo_nominal=lm

    has_ann_k  = fut_ann_kasus   is not None and len(fut_ann_kasus)>0
    has_ann_n  = fut_ann_nominal is not None and len(fut_ann_nominal)>0
    has_mo_k   = fut_mo_kasus   is not None and len(fut_mo_kasus)>0
    has_mo_n   = fut_mo_nominal is not None and len(fut_mo_nominal)>0

    with ec1:
        st.markdown("**📊 Excel dengan Chart Terintegrasi**")
        st.caption("Sheet: Data Gabungan · Pivot Kasus · Prediksi Tahunan · Prediksi Bulanan · Bulanan Detail · ML Results")
        st.markdown(f"""<div class="export-box">
        <b>Prediksi Tahunan:</b><br>
        <span class="{'ok' if has_ann_k else 'nok'}">{'✅' if has_ann_k else '⬜'} Kasus tahunan {'siap ('+str(len(fut_ann_kasus))+' baris)' if has_ann_k else 'belum ada'}</span><br>
        <span class="{'ok' if has_ann_n else 'nok'}">{'✅' if has_ann_n else '⬜'} Nominal tahunan {'siap ('+str(len(fut_ann_nominal))+' baris)' if has_ann_n else 'belum ada'}</span><br>
        <br><b>Prediksi Bulanan:</b><br>
        <span class="{'ok' if has_mo_k else 'nok'}">{'✅' if has_mo_k else '⬜'} Kasus bulanan {'siap ('+str(len(fut_mo_kasus))+' baris)' if has_mo_k else 'belum ada'}</span><br>
        <span class="{'ok' if has_mo_n else 'nok'}">{'✅' if has_mo_n else '⬜'} Nominal bulanan {'siap ('+str(len(fut_mo_nominal))+' baris)' if has_mo_n else 'belum ada'}</span>
        </div>""", unsafe_allow_html=True)

        if st.button("⚙️  Generate Excel", use_container_width=True):
            with st.spinner("Membuat Excel…"):
                xlsx=export_excel(df, best_ml, last_fut,
                    fut_kasus=fut_ann_kasus, fut_nominal=fut_ann_nominal,
                    fut_monthly_kasus=fut_mo_kasus, fut_monthly_nominal=fut_mo_nominal)
            st.download_button("⬇️  Download Excel", data=xlsx,
                file_name=f"BPJS_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True)

    with ec2:
        st.markdown("**📄 CSV Data Gabungan**")
        df_sorted=df.sort_values(['Tahun','Kategori']).reset_index(drop=True)
        st.download_button("⬇️  Download CSV", data=df_sorted.to_csv(index=False),
            file_name=f"BPJS_Data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv", use_container_width=True)

        if has_mo_k or has_mo_n:
            st.markdown("<br>**📄 CSV Prediksi Bulanan**")
            frames_csv=[]
            if has_mo_k: frames_csv.append(fut_mo_kasus.assign(Target='Kasus'))
            if has_mo_n: frames_csv.append(fut_mo_nominal.assign(Target='Nominal'))
            combined_csv=(pd.concat(frames_csv,ignore_index=True)
                          .sort_values(['Tahun','Bulan','Kategori','Target']).reset_index(drop=True))
            st.download_button("⬇️  Download CSV Bulanan", data=combined_csv.to_csv(index=False),
                file_name=f"BPJS_Bulanan_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv", use_container_width=True)

    st.markdown('<div class="sec">Preview Data Aktif</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="info-box"><b>{len(df)} baris</b> · <b>{len(active_progs)} program aktif</b> ({", ".join(active_progs)}) · <b>Tahun: {", ".join(map(str, years))}</b></div>', unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True, height=360)