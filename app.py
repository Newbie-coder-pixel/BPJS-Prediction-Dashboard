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
        st.markdown("## 🔒 BPJS ML Dashboard — Login")
        password = st.text_input("Masukkan password:", type="password")
        if st.button("Login"):
            if password == "bpjs2026":
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Password salah!")
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

st.set_page_config(page_title="BPJS ML Dashboard", layout="wide", page_icon="📊", initial_sidebar_state="expanded")

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

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
*{box-sizing:border-box;}
html,body,[class*="css"]{font-family:'Inter',sans-serif;}

/* ── LIGHT BACKGROUND (poin 7) ── */
.stApp{background:#f1f5f9 !important;color:#1e293b !important;}
.main .block-container{background:#f1f5f9;padding-top:1.5rem;}
section[data-testid="stSidebar"]{background:#ffffff !important;border-right:1px solid #e2e8f0 !important;}
section[data-testid="stSidebar"] > div{background:#ffffff !important;}

/* Scrollbar */
::-webkit-scrollbar{width:5px;height:5px;}
::-webkit-scrollbar-track{background:#f1f5f9;}
::-webkit-scrollbar-thumb{background:#cbd5e1;border-radius:4px;}
::-webkit-scrollbar-thumb:hover{background:#94a3b8;}

/* ── KPI CARDS ── */
.kpi{
  background:#ffffff;border:1px solid #e2e8f0;border-radius:14px;
  padding:20px 14px 16px;text-align:center;
  box-shadow:0 1px 4px rgba(0,0,0,.06);
  transition:transform .2s,box-shadow .2s;cursor:default;
}
.kpi:hover{transform:translateY(-2px);box-shadow:0 4px 16px rgba(0,0,0,.10);}
.kpi::before{content:'';position:absolute;display:none;}
.kpi .val{font-size:1.7rem;font-weight:800;color:#0f172a;font-family:'JetBrains Mono',monospace;line-height:1.2;}
.kpi .lbl{font-size:.68rem;color:#64748b;text-transform:uppercase;letter-spacing:1.2px;margin-top:6px;font-weight:600;}
.kpi .delta{font-size:.75rem;margin-top:5px;font-weight:500;}
.delta-pos{color:#16a34a;} .delta-neg{color:#dc2626;} .delta-neu{color:#64748b;}

/* ── SECTION HEADERS ── */
.sec{
  font-size:.7rem;font-weight:700;color:#64748b;
  text-transform:uppercase;letter-spacing:1.5px;
  margin:24px 0 10px;padding-bottom:6px;
  border-bottom:2px solid #e2e8f0;
}

/* ── INFO/BADGE BOXES ── */
.badge{background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;
  padding:12px 16px;font-size:.84rem;line-height:1.8;
  border-left:3px solid #3b82f6;color:#1e40af;}
.badge b{color:#1d4ed8;}
.warn{background:#fffbeb;border:1px solid #fde68a;border-radius:10px;
  padding:12px 16px;color:#92400e;font-size:.84rem;margin:8px 0;
  border-left:3px solid #f59e0b;line-height:1.7;}
.info-box{background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;
  padding:12px 16px;font-size:.84rem;color:#166534;
  margin:8px 0;line-height:1.8;border-left:3px solid #22c55e;}
.success-box{background:#f0fdf4;border:1px solid #86efac;border-radius:10px;
  padding:12px 16px;font-size:.84rem;color:#166534;
  margin:8px 0;border-left:3px solid #16a34a;line-height:1.8;}
.insight-note{background:#fafafa;border:1px solid #e2e8f0;border-radius:8px;
  padding:10px 14px;font-size:.82rem;color:#475569;line-height:1.7;margin:6px 0;}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"]{
  gap:2px;background:#e2e8f0;border-radius:10px;padding:3px;border:none;
}
.stTabs [data-baseweb="tab"]{
  border-radius:8px;padding:8px 18px;font-size:.84rem;font-weight:500;color:#475569;
}
.stTabs [aria-selected="true"]{background:#ffffff !important;color:#0f172a !important;
  box-shadow:0 1px 4px rgba(0,0,0,.12) !important;}

/* ── BUTTONS ── */
.stButton>button{border-radius:8px;font-weight:600;font-size:.85rem;transition:all .2s;}
.stButton>button[kind="primary"]{
  background:#2563eb !important;border:none !important;color:#fff !important;
  box-shadow:0 2px 8px rgba(37,99,235,.35);}
.stButton>button[kind="primary"]:hover{background:#1d4ed8 !important;}

/* ── INPUT ── */
.stSelectbox [data-baseweb="select"] > div{
  background:#ffffff !important;border-color:#e2e8f0 !important;color:#1e293b !important;}
.stTextInput input{background:#ffffff !important;border:1px solid #e2e8f0 !important;
  border-radius:8px !important;color:#1e293b !important;}

/* ── DATAFRAME ── */
.stDataFrame{border-radius:10px;overflow:hidden;border:1px solid #e2e8f0;}

/* ── EXPANDER ── */
[data-testid="stExpander"]{border:1px solid #e2e8f0;border-radius:10px;background:#ffffff;}
[data-testid="stExpander"] summary{background:#ffffff;color:#475569;font-size:.85rem;}

/* ── SIDEBAR ELEMENTS ── */
section[data-testid="stSidebar"] label{color:#475569 !important;font-size:.82rem !important;}
section[data-testid="stSidebar"] .stMarkdown{color:#475569;}
section[data-testid="stSidebar"] .stButton>button{
  background:#f8fafc;border:1px solid #e2e8f0;color:#475569;border-radius:8px;
  font-size:.82rem;transition:all .2s;}
section[data-testid="stSidebar"] .stButton>button:hover{border-color:#2563eb;color:#2563eb;}

/* ── HERO HEADER ── */
.hero-wrap{padding:20px 24px;background:#ffffff;border:1px solid #e2e8f0;
  border-radius:14px;border-left:4px solid #2563eb;margin-bottom:8px;
  box-shadow:0 1px 4px rgba(0,0,0,.05);}
.hero-logo{font-size:1.35rem;font-weight:800;color:#0f172a;line-height:1.2;margin-bottom:2px;}
.hero-sub{font-size:.78rem;color:#64748b;letter-spacing:.3px;}

/* ── PROGRAM FILTER BAR ── */
.filter-bar{background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;
  padding:12px 18px;margin-bottom:16px;display:flex;align-items:center;gap:12px;
  box-shadow:0 1px 3px rgba(0,0,0,.05);}

/* ── MODEL PILLS ── */
.mpill{display:inline-block;padding:2px 10px;border-radius:20px;font-size:.75rem;font-weight:600;margin:2px;}
.mpill-green{background:#dcfce7;color:#15803d;border:1px solid #86efac;}
.mpill-blue{background:#dbeafe;color:#1d4ed8;border:1px solid #93c5fd;}
.mpill-yellow{background:#fef9c3;color:#854d0e;border:1px solid #fde047;}
.mpill-red{background:#fee2e2;color:#b91c1c;border:1px solid #fca5a5;}

/* ── CONCLUSION CARDS ── */
.concl-card{background:#ffffff;border:1px solid #e2e8f0;border-radius:10px;
  padding:16px 20px;margin:8px 0;border-left:3px solid #2563eb;}
.concl-card .ct{font-size:.7rem;color:#64748b;text-transform:uppercase;
  letter-spacing:1px;font-weight:600;margin-bottom:5px;}
.concl-card .cv{color:#334155;font-size:.88rem;line-height:1.8;}

/* ── PEAK ANNOTATION BOX ── */
.peak-box{background:#fffbeb;border:1px solid #fde68a;border-radius:8px;
  padding:10px 14px;font-size:.8rem;color:#713f12;line-height:1.7;margin:6px 0;}
.trough-box{background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;
  padding:10px 14px;font-size:.8rem;color:#1e3a8a;line-height:1.7;margin:6px 0;}

/* ── PROG CARDS ── */
.prog-card{background:#ffffff;border:1px solid #e2e8f0;border-radius:10px;
  padding:10px 14px;min-width:120px;display:inline-block;}
.prog-card .pc-name{font-size:.68rem;color:#64748b;text-transform:uppercase;letter-spacing:.8px;font-weight:600;margin-bottom:3px;}
.prog-card .pc-model{font-size:.88rem;font-weight:700;color:#0f172a;margin-bottom:4px;}

/* ── DOWNLOAD BUTTON ── */
[data-testid="stDownloadButton"] button{background:#059669 !important;
  border:1px solid #047857 !important;color:#fff !important;
  border-radius:8px;font-weight:600;}

/* ── TAGS ── */
.tag-add{display:inline-block;background:#dcfce7;color:#15803d;
  border:1px solid #86efac;border-radius:5px;padding:2px 8px;font-size:.76rem;margin:2px;font-weight:600;}
.tag-rem{display:inline-block;background:#fee2e2;color:#b91c1c;
  border:1px solid #fca5a5;border-radius:5px;padding:2px 8px;font-size:.76rem;margin:2px;font-weight:600;}
.tag-stable{display:inline-block;background:#f1f5f9;color:#475569;
  border:1px solid #cbd5e1;border-radius:5px;padding:2px 8px;font-size:.76rem;margin:2px;font-weight:600;}

/* ── INSIGHT CARD ── */
.insight-card{background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;padding:18px 20px;
  box-shadow:0 1px 4px rgba(0,0,0,.05);}
.insight-card .ic-title{font-size:.7rem;color:#64748b;text-transform:uppercase;
  letter-spacing:1px;font-weight:600;margin-bottom:6px;}
.insight-card .ic-val{font-size:1.3rem;font-weight:700;color:#0f172a;}
.insight-card .ic-sub{font-size:.8rem;color:#64748b;margin-top:3px;}

/* ── ALERT ── */
[data-testid="stAlert"]{border-radius:10px;}

/* ── FEAT CARDS ── */
.feat-card{background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;padding:22px 18px;
  text-align:center;height:180px;border-top:3px solid #2563eb;
  box-shadow:0 1px 4px rgba(0,0,0,.05);transition:transform .2s;}
.feat-card:hover{transform:translateY(-3px);}
.feat-icon{font-size:1.8rem;margin-bottom:8px;}
.feat-title{font-weight:700;color:#0f172a;margin-bottom:6px;font-size:.92rem;}
.feat-desc{color:#64748b;font-size:.8rem;line-height:1.6;}

/* ── EXPORT BOX ── */
.export-box{background:#ffffff;border:1px solid #e2e8f0;border-radius:10px;
  padding:16px 18px;line-height:2;font-size:.84rem;}
.export-box .ok{color:#16a34a;} .export-box .nok{color:#94a3b8;}

/* ── EMPTY STATE ── */
.empty-state{text-align:center;padding:70px 0 50px;}
.empty-icon{font-size:3.5rem;margin-bottom:16px;}
.empty-title{font-size:1.8rem;font-weight:800;color:#0f172a;margin-bottom:12px;}
.empty-sub{color:#64748b;max-width:500px;margin:auto;font-size:.9rem;line-height:1.8;}
</style>""", unsafe_allow_html=True)


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

DARK = dict(template='plotly_dark', paper_bgcolor='rgba(15,23,42,0.95)', plot_bgcolor='rgba(15,23,42,0.95)',
            font_color='#94a3b8', font_family='Inter')
COLORS = ['#60a5fa','#34d399','#fb923c','#a78bfa','#f87171',
          '#fbbf24','#38bdf8','#f472b6','#4ade80','#e879f9']
COLORS_ALPHA = {c: c for c in COLORS}

def hex_to_rgba(hex_c, alpha=1.0):
    h = hex_c.lstrip('#')
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f'rgba({r},{g},{b},{alpha})'

def styled_chart(fig, height=400, legend_bottom=True, margin_b=80):
    fig.update_layout(
        **DARK, height=height,
        hovermode='x unified',
        hoverlabel=dict(bgcolor='#0d1f35', font_size=12, bordercolor='#1e3a5f'),
        legend=dict(orientation='h', y=-0.22, font=dict(size=10.5),
                    groupclick='toggleitem') if legend_bottom else {},
        margin=dict(b=margin_b if legend_bottom else 40, t=20, l=60, r=20),
        xaxis=dict(showgrid=True, gridcolor='#0f1923', gridwidth=1,
                   zeroline=False, linecolor='#1e2d45'),
        yaxis=dict(showgrid=True, gridcolor='#0f1923', gridwidth=1,
                   zeroline=False, linecolor='#1e2d45'),
    )
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# DATA PARSING
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
        y0 = years[i]
        y1 = years[i + 1]
        p0 = set(df[df['Tahun'] == y0]['Kategori'].unique())
        p1 = set(df[df['Tahun'] == y1]['Kategori'].unique())
        changes[(y0, y1)] = {
            'added':   sorted(p1 - p0),
            'removed': sorted(p0 - p1),
            'stable':  sorted(p0 & p1),
        }
    return changes

def get_active_programs(df):
    latest = df['Tahun'].max()
    return sorted(df[df['Tahun'] == latest]['Kategori'].unique())

# ══════════════════════════════════════════════════════════════════════════════
# ML CORE
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
                lvl[0] = y[0]; trnd[0] = y[1] - y[0] if n > 1 else 0
                for i in range(1, n):
                    lvl[i]  = a * y[i] + (1-a) * (lvl[i-1] + trnd[i-1])
                    trnd[i] = b * (lvl[i] - lvl[i-1]) + (1-b) * trnd[i-1]
                fitted = lvl[:-1] + trnd[:-1]
                mape = np.mean(np.abs((y[1:] - fitted) / (np.abs(y[1:]) + 1e-9))) * 100
                if mape < best_mape:
                    best_mape = mape; best_a = a; best_b = b
            except:
                pass

    lvl = np.zeros(n); trnd = np.zeros(n)
    lvl[0] = y[0]; trnd[0] = y[1] - y[0] if n > 1 else 0
    for i in range(1, n):
        lvl[i]  = best_a * y[i] + (1-best_a) * (lvl[i-1] + trnd[i-1])
        trnd[i] = best_b * (lvl[i] - lvl[i-1]) + (1-best_b) * trnd[i-1]
    preds = np.array([lvl[-1] + (s+1)*trnd[-1] for s in range(n_steps)])
    return preds, best_a, best_b, lvl, trnd


def forecast_ses(history, n_steps):
    y = np.array(history, dtype=float)
    best_mape = np.inf; best_a = 0.3
    for a in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        lvl = y[0]
        fitted = []
        for i in range(1, len(y)):
            fitted.append(lvl)
            lvl = a * y[i] + (1-a) * lvl
        if fitted:
            mape = np.mean(np.abs((y[1:len(fitted)+1] - np.array(fitted)) / (np.abs(y[1:len(fitted)+1]) + 1e-9))) * 100
            if mape < best_mape:
                best_mape = mape; best_a = a
    lvl = y[0]
    for v in y[1:]:
        lvl = best_a * v + (1-best_a) * lvl
    return np.array([lvl] * n_steps), best_a


def forecast_moving_avg(history, n_steps, window=3):
    y = np.array(history, dtype=float)
    w = min(window, len(y))
    weights = np.arange(1, w+1, dtype=float)
    weights /= weights.sum()
    base = float(np.dot(y[-w:], weights))
    if len(y) >= 2:
        trend = float(y[-1] - y[-2])
    else:
        trend = 0.0
    trend = np.clip(trend, -abs(base)*0.3, abs(base)*0.3)
    return np.array([base + trend*(s+1)*0.5 for s in range(n_steps)])


def loo_cv_stat(history, method_fn, n_steps=1):
    y = np.array(history, dtype=float)
    n = len(y)
    yt_all, yp_all = [], []
    for leave in range(1, n):
        train = y[:leave]
        actual = y[leave]
        try:
            preds = method_fn(train, n_steps)
            if isinstance(preds, tuple): preds = preds[0]
            yp_all.append(float(preds[0]))
            yt_all.append(actual)
        except:
            pass
    if len(yt_all) < 1:
        return {'MAE': np.inf, 'RMSE': np.inf, 'R2': -999, 'MAPE (%)': np.inf}
    return score_model(np.array(yt_all), np.array(yp_all))


def build_features(series, n_lags=1, cat_id=0.0):
    pad = list(series)
    while len(pad) <= n_lags:
        pad.insert(0, pad[0])
    X_all, y_all = [], []
    for i in range(n_lags, len(pad)):
        lags = [pad[i - l] for l in range(1, n_lags + 1)]
        win  = pad[max(0, i-3):i]
        feat = lags + [
            np.mean(win),
            np.std(win) if len(win) > 1 else 0.0,
            pad[i-1] - pad[i-2] if i >= 2 else 0.0,
            cat_id
        ]
        X_all.append(feat)
        y_all.append(pad[i])
    return np.array(X_all), np.array(y_all)


SCALED_MODELS = {'SVR', 'KNN', 'Ridge', 'Lasso', 'ElasticNet', 'Linear Regression', 'Huber'}


def get_ml_models(n_train):
    k = min(3, max(1, n_train - 1))
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
        models['XGBoost'] = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1,
                                          n_jobs=-1, random_state=42, verbosity=0)
    if LGBM_OK:
        models['LightGBM'] = LGBMRegressor(n_estimators=100, max_depth=3, learning_rate=0.1,
                                             n_jobs=-1, random_state=42, verbose=-1)
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
        except:
            pass
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

    per_prog    = {}
    detail_rows = []

    for cat in active:
        sub = (df[df['Kategori'] == cat]
               .sort_values('Tahun')[target].dropna().values.astype(float))
        if len(sub) == 0:
            continue

        if len(sub) < 2:
            per_prog[cat] = {
                'best_name': 'Holt Smoothing', 'method_type': 'stat',
                'history': list(sub), 'single': True,
                'metrics': {'R2': None, 'MAPE (%)': None, 'MAE': None, 'RMSE': None},
                'cat_id': cat_enc.get(cat, 0.0),
            }
            continue

        use_ml = (len(sub) >= 8)

        stat_candidates = {
            'Holt Smoothing':   lambda h, s: forecast_holt(h, s),
            'Exp Smoothing':    lambda h, s: (forecast_ses(h, s)[0],),
            'Weighted MA':      lambda h, s: (forecast_moving_avg(h, s),),
        }
        stat_scores = {}
        for mname, fn in stat_candidates.items():
            sc = loo_cv_stat(list(sub), fn)
            stat_scores[mname] = sc
            detail_rows.append({'Program': cat, 'Model': mname, **sc})

        ml_scores = {}
        ml_models_fitted = {}
        if use_ml:
            Xc, yc = build_features(sub, min(n_lags, 2), cat_enc.get(cat, 0.0))
            sc_full = StandardScaler().fit(Xc)
            ml_defs = get_ml_models(len(Xc) - 1)
            for mname, mdl_obj in ml_defs.items():
                sc = loo_cv_ml(Xc, yc, mname, mdl_obj)
                ml_scores[mname] = sc
                detail_rows.append({'Program': cat, 'Model': mname, **sc})
                try:
                    mdl_full = copy.deepcopy(mdl_obj)
                    Xc_s = sc_full.transform(Xc)
                    mdl_full.fit(Xc_s if mname in SCALED_MODELS else Xc, yc)
                    ml_models_fitted[mname] = {'model': mdl_full, 'scaler': sc_full, 'Xc': Xc, 'yc': yc}
                except:
                    pass

        all_scores = {**stat_scores, **ml_scores}
        valid = {m: s for m, s in all_scores.items() if s['MAPE (%)'] < 200}
        pool  = valid if valid else all_scores
        if not pool:
            pool = stat_scores

        best_name = min(pool, key=lambda m: pool[m]['MAPE (%)'])
        best_sc   = all_scores[best_name]

        is_stat = best_name in stat_candidates
        entry = {
            'best_name':   best_name,
            'method_type': 'stat' if is_stat else 'ml',
            'history':     list(sub),
            'single':      False,
            'metrics':     best_sc,
            'all_scores':  all_scores,
            'cat_id':      cat_enc.get(cat, 0.0),
        }
        if is_stat:
            entry['stat_fn_name'] = best_name
        else:
            if best_name in ml_models_fitted:
                info_ml = ml_models_fitted[best_name]
                entry['best_model'] = info_ml['model']
                entry['scaler']     = info_ml['scaler']
                entry['n_lags_used'] = min(n_lags, 2)

        per_prog[cat] = entry

    bpp_rows = []
    for cat, info in per_prog.items():
        m = info.get('metrics', {})
        bpp_rows.append({
            'Program': cat, 'Model': info['best_name'],
            'Tipe': '📊 Statistik' if info.get('method_type') == 'stat' else '🤖 ML',
            'R2':       m.get('R2'), 'MAPE (%)': m.get('MAPE (%)'),
            'MAE':      m.get('MAE'), 'RMSE':    m.get('RMSE'),
        })
    best_per_prog = pd.DataFrame(bpp_rows)
    detail_df     = pd.DataFrame(detail_rows) if detail_rows else pd.DataFrame()

    valid_bpp = [r for r in bpp_rows if r['MAPE (%)'] is not None and r['MAPE (%)'] < 200]
    avg_r2   = float(np.mean([r['R2'] for r in valid_bpp if r['R2'] is not None])) if valid_bpp else None
    avg_mape = float(np.mean([r['MAPE (%)'] for r in valid_bpp])) if valid_bpp else 0.0
    avg_mae  = float(np.mean([r['MAE']      for r in valid_bpp])) if valid_bpp else 0.0

    from collections import Counter
    overall_best = Counter(r['Model'] for r in bpp_rows).most_common(1)[0][0] if bpp_rows else 'N/A'

    return {
        'per_prog': per_prog, 'best_per_program': best_per_prog,
        'detail': detail_df,  'results_df': pd.DataFrame(),
        'best_name': overall_best, 'best_r2': avg_r2,
        'best_mape': avg_mape, 'best_mae': avg_mae,
        'cat_enc': cat_enc, 'single': single,
        'n_lags': n_lags, 'target': target,
        'active_programs': active,
    }, None


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
    if ml_result is None:
        return lines

    bpp = ml_result.get('best_per_program', pd.DataFrame())
    per_prog = ml_result.get('per_prog', {})

    if not bpp.empty and 'R2' in bpp.columns:
        r2_val   = float(bpp['R2'].mean())
        mape_val = float(bpp['MAPE (%)'].mean())
    else:
        r2_val, mape_val = ml_result.get('best_r2', 0.0), ml_result.get('best_mape', 0.0)

    r2_grade   = ("Sangat Baik (>0.9)" if r2_val > 0.9 else
                  "Baik (0.8–0.9)"     if r2_val > 0.8 else
                  "Cukup (0.6–0.8)"    if r2_val > 0.6 else "Lemah (<0.6)")
    mape_grade = ("Sangat Akurat (<10%)" if mape_val < 10 else
                  "Akurat (10–20%)"      if mape_val < 20 else
                  "Cukup (20–50%)"       if mape_val < 50 else "Tidak Akurat (>50%)")

    lines.append(('🎯', 'Pendekatan Prediksi',
        "Setiap program menggunakan **model terbaiknya sendiri** (per-program best model). "
        f"Rata-rata R² = **{r2_val:.4f}** ({r2_grade}), rata-rata MAPE = **{mape_val:.2f}%** ({mape_grade})."))

    if not bpp.empty:
        prog_str = ', '.join(f"{r['Program']} → **{r['Model']}**" for _, r in bpp.iterrows())
        lines.append(('📊', 'Model Terbaik per Program', prog_str))
        if len(bpp) > 1:
            worst  = bpp.sort_values('R2').iloc[0]
            best_p = bpp.sort_values('R2', ascending=False).iloc[0]
            lines.append(('🔍', 'Akurasi per Program',
                f"Program **{best_p['Program']}** paling mudah diprediksi (R²={best_p['R2']:.3f}, MAPE={best_p['MAPE (%)']:.1f}%). "
                f"Program **{worst['Program']}** paling sulit (R²={worst['R2']:.3f}, MAPE={worst['MAPE (%)']:.1f}%) — "
                "pertimbangkan menambah data historis atau fitur eksternal."))

    base_yr = int(df['Tahun'].max())
    lines.append(('📅', 'Horizon Prediksi',
        f"Model dilatih pada data s/d **{base_yr}** dan mampu memproyeksikan hingga "
        f"**{base_yr + n_future}** ({n_future} tahun ke depan). "
        "Akurasi menurun semakin jauh horizon waktu — gunakan prediksi jangka pendek untuk keputusan kritis."))

    yrs = sorted(df['Tahun'].unique())
    lines.append(('📁', 'Kualitas Data',
        f"Dataset mencakup **{len(yrs)} tahun** ({yrs[0]}–{yrs[-1]}) "
        f"dengan **{len(get_active_programs(df))} program aktif**. "
        + ("✅ Jumlah tahun cukup untuk model lag." if len(yrs) >= 4
           else "⚠️ Tambah data historis untuk meningkatkan akurasi model.")))

    if r2_val >= 0.8 and mape_val <= 20:
        rec = "✅ Model layak digunakan untuk perencanaan anggaran dan proyeksi klaim BPJS."
    elif r2_val >= 0.6:
        rec = "⚠️ Model cukup untuk proyeksi kasar. Validasi manual disarankan sebelum keputusan strategis."
    else:
        rec = "❌ Akurasi belum optimal. Tambah data historis minimal 5 tahun, atau gunakan Prophet untuk data bulanan."
    lines.append(('💡', 'Rekomendasi', rec))
    return lines


# ══════════════════════════════════════════════════════════════════════════════
# INDONESIAN HOLIDAYS — Google Calendar API
# ══════════════════════════════════════════════════════════════════════════════

GCAL_ID  = "en.indonesian%23holiday%40group.v.calendar.google.com"
# Gunakan .get() agar tidak crash jika key belum diset
GCAL_KEY = st.secrets.get("GCAL_KEY", "")

_WINDOW_RULES = {
    'idul fitri'          : (-7,  7),
    'lebaran'             : (-7,  7),
    'ramadan'             : ( 0, 29),
    'ramadhan'            : ( 0, 29),
    'puasa'               : ( 0, 29),
    'idul adha'           : (-3,  3),
    'natal'               : (-2,  2),
    'christmas'           : (-2,  2),
    'tahun baru'          : (-1,  2),
    'new year'            : (-1,  2),
    'cuti bersama'        : (-1,  1),
    'default'             : (-1,  1),
}

def _get_window(name: str):
    nl = name.lower()
    for keyword, (lo, hi) in _WINDOW_RULES.items():
        if keyword != 'default' and keyword in nl:
            return lo, hi
    return _WINDOW_RULES['default']


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_google_holidays(year_start: int = 2019, year_end: int = 2028) -> list:
    """
    Ambil semua hari libur Indonesia dari Google Calendar API.
    Return [] jika API tidak tersedia atau GCAL_KEY kosong.
    """
    import urllib.request, urllib.parse, json

    if not GCAL_KEY:
        return []

    base_url = f"https://www.googleapis.com/calendar/v3/calendars/{GCAL_ID}/events"
    all_rows  = []
    seen_keys = set()

    for year in range(year_start, year_end + 1):
        page_token = None
        while True:
            params = {
                'key'          : GCAL_KEY,
                'timeMin'      : f'{year}-01-01T00:00:00Z',
                'timeMax'      : f'{year}-12-31T23:59:59Z',
                'maxResults'   : '2500',
                'singleEvents' : 'true',
                'orderBy'      : 'startTime',
            }
            if page_token:
                params['pageToken'] = page_token

            url = base_url + '?' + urllib.parse.urlencode(params)
            try:
                with urllib.request.urlopen(url, timeout=10) as r:
                    data = json.loads(r.read().decode('utf-8'))
            except Exception:
                break

            if 'error' in data:
                return []

            for item in data.get('items', []):
                start_raw = (item.get('start', {}).get('date')
                             or item.get('start', {}).get('dateTime', '')[:10])
                name = item.get('summary', '').strip()
                if not start_raw or not name:
                    continue
                try:
                    ds = pd.Timestamp(start_raw)
                except Exception:
                    continue
                key = (str(ds.date()), name)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                lo, hi = _get_window(name)
                all_rows.append({
                    'ds'           : ds,
                    'holiday'      : name,
                    'lower_window' : lo,
                    'upper_window' : hi,
                })

            page_token = data.get('nextPageToken')
            if not page_token:
                break

    return all_rows


@st.cache_data(ttl=86400, show_spinner=False)
def build_holiday_df() -> pd.DataFrame:
    """
    Build DataFrame hari libur dari Google Calendar API.
    Jika API gagal atau GCAL_KEY tidak diset, kembalikan DataFrame kosong.
    Prophet tetap berjalan tanpa holiday — tidak ada hardcode apapun.
    """
    rows = fetch_google_holidays()
    if not rows:
        return pd.DataFrame(columns=['ds', 'holiday', 'lower_window', 'upper_window'])

    df_h = pd.DataFrame(rows)
    df_h['ds'] = pd.to_datetime(df_h['ds'])
    df_h = (df_h
            .drop_duplicates(subset=['ds', 'holiday'])
            .sort_values('ds')
            .reset_index(drop=True))
    return df_h


INDONESIAN_HOLIDAYS = build_holiday_df()


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: Prophet holiday column name sanitization
# Prophet mengubah nama holiday saat membuat kolom di forecast:
#   spasi dan karakter non-alphanumeric → underscore
# Fungsi ini membuat mapping: nama_kolom_prophet → nama_asli_holiday
# ══════════════════════════════════════════════════════════════════════════════

def _sanitize_prophet_name(name: str) -> str:
    """Replicate Prophet's internal holiday name sanitization."""
    return re.sub(r'[^\w]', '_', str(name))


def _build_holiday_col_map(holidays_df: pd.DataFrame) -> dict:
    """
    Buat mapping dari nama kolom Prophet (sanitized) ke nama asli holiday.
    Contoh: 'Idul_Fitri' -> 'Idul Fitri'
    """
    if holidays_df is None or len(holidays_df) == 0:
        return {}
    col_map = {}
    for orig_name in holidays_df['holiday'].unique():
        sanitized = _sanitize_prophet_name(orig_name)
        # Jika ada duplikat sanitized name, simpan semua (ambil yang pertama saja untuk display)
        if sanitized not in col_map:
            col_map[sanitized] = orig_name
    return col_map


def _get_prophet_holiday_columns(forecast_df: pd.DataFrame, col_map: dict) -> list:
    """
    Temukan semua kolom holiday di forecast Prophet.
    Return list of (kolom_name, nama_asli_holiday).
    """
    found = []
    for col in forecast_df.columns:
        if col in col_map:
            found.append((col, col_map[col]))
    return found


# ══════════════════════════════════════════════════════════════════════════════
# PROPHET
# ══════════════════════════════════════════════════════════════════════════════

def run_prophet(df_monthly_raw, target, cat, n_months, use_holidays=True):
    if not PROPHET_OK:
        return None, "Prophet tidak terinstall. Tambahkan 'prophet' ke requirements.txt."
    cat_df = df_monthly_raw[df_monthly_raw['Kategori'] == cat].copy()
    if len(cat_df) < 6:
        return None, f"Data {cat} kurang dari 6 bulan."
    cat_df = cat_df.sort_values(['Tahun','Bulan'])
    cat_df['ds'] = pd.to_datetime(
        cat_df['Tahun'].astype(str) + '-' + cat_df['Bulan'].astype(str).str.zfill(2) + '-01')
    cat_df = cat_df.groupby('ds')[target].sum().reset_index()
    cat_df.columns = ['ds', 'y']
    cat_df = cat_df[cat_df['y'] > 0].sort_values('ds').reset_index(drop=True)

    if len(cat_df) < 6:
        return None, f"Data {cat} setelah filtering kurang dari 6 bulan."

    holidays_df = INDONESIAN_HOLIDAYS.copy() if use_holidays and len(INDONESIAN_HOLIDAYS) > 0 else None

    y_floor = 0.0
    y_cap   = float(cat_df['y'].max()) * 3.0

    try:
        n_data = len(cat_df)
        if n_data >= 24:
            s_mode = 'multiplicative'
            cp_scale = 0.05
        else:
            s_mode = 'additive'
            cp_scale = 0.03

        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            holidays=holidays_df,
            seasonality_mode=s_mode,
            interval_width=0.80,
            changepoint_prior_scale=cp_scale,
            seasonality_prior_scale=5.0,
            holidays_prior_scale=5.0,
            growth='flat' if n_data < 12 else 'linear',
        )
        m.fit(cat_df)
        future = m.make_future_dataframe(periods=n_months, freq='MS')
        fc = m.predict(future)

        fc['yhat']       = fc['yhat'].clip(lower=y_floor)
        fc['yhat_lower'] = fc['yhat_lower'].clip(lower=y_floor)
        fc['yhat_upper'] = fc['yhat_upper'].clip(lower=y_floor)
        fc['yhat']       = fc['yhat'].clip(upper=y_cap)
        fc['yhat_upper'] = fc['yhat_upper'].clip(upper=y_cap * 1.2)

        hist_pred = fc[fc['ds'].isin(cat_df['ds'])]
        if len(hist_pred) > 0:
            yt = cat_df.set_index('ds').loc[hist_pred['ds'], 'y'].values
            yp = hist_pred['yhat'].values
            r2_is   = float(r2_score(yt, yp)) if len(yt) > 1 else 0.0
            mape_is = float(np.mean(np.abs((yt - yp)/(np.abs(yt)+1e-9)))*100)
        else:
            r2_is = mape_is = 0.0

        n_holidays = len(holidays_df) if holidays_df is not None else 0

        # Build holiday column map untuk digunakan saat analisis efek
        h_col_map = _build_holiday_col_map(holidays_df) if holidays_df is not None else {}

        return {'model': m, 'forecast': fc, 'history': cat_df,
                'r2_insample': r2_is, 'mape_insample': mape_is,
                'n_holidays': n_holidays, 'gcal_used': n_holidays > 0,
                'holiday_col_map': h_col_map}, None
    except Exception as e:
        return None, str(e)


def forecast(df, ml, n_years):
    target   = ml['target']
    nlags    = ml['n_lags']
    active   = ml['active_programs']
    per_prog = ml.get('per_prog', {})
    base_yr  = int(df['Tahun'].max())
    rows     = []

    STAT_METHODS = {'Holt Smoothing', 'Exp Smoothing', 'Weighted MA'}

    for cat in active:
        info    = per_prog.get(cat, None)
        history = list(df[df['Kategori'] == cat]
                       .sort_values('Tahun')[target]
                       .dropna().values.astype(float))
        if not history: continue

        best_nm = info.get('best_name', 'Holt Smoothing') if info else 'Holt Smoothing'

        if info is None or info.get('single', True) or best_nm in STAT_METHODS:
            for fy in range(1, n_years + 1):
                try:
                    if best_nm == 'Holt Smoothing' or info is None:
                        result = forecast_holt(history, 1)
                        pred = float(result[0][0])
                    elif best_nm == 'Exp Smoothing':
                        result = forecast_ses(history, 1)
                        pred = float(result[0][0])
                    elif best_nm == 'Weighted MA':
                        result = forecast_moving_avg(history, 1)
                        pred = float(result[0])
                    else:
                        result = forecast_holt(history, 1)
                        pred = float(result[0][0])
                except:
                    pred = history[-1] * 1.05
                pred = max(0.0, pred)
                rows.append({'Kategori': cat, 'Tahun': base_yr + fy,
                             target: pred, 'Type': f'Prediksi ({best_nm})'})
                history.append(pred)
            continue

        mdl       = info.get('best_model')
        sc        = info.get('scaler')
        cat_id    = info.get('cat_id', 0.0)
        nlags_use = info.get('n_lags_used', min(nlags, 2))
        last_actual = history[-1]

        for fy in range(1, n_years + 1):
            Xc, _ = build_features(history, nlags_use, cat_id)
            pred = None
            if len(Xc) > 0 and mdl is not None:
                feat = Xc[-1].reshape(1, -1)
                try:
                    feat_use = sc.transform(feat) if best_nm in SCALED_MODELS else feat
                    pred = float(mdl.predict(feat_use)[0])
                    if pred < last_actual * 0.5 or pred > last_actual * 2.0:
                        holt_pred = float(forecast_holt(history[:fy + len(history) - 1], 1)[0][0])
                        pred = (pred + holt_pred) / 2
                except:
                    pred = None
            if pred is None:
                try:
                    pred = float(forecast_holt(history, 1)[0][0])
                except:
                    pred = history[-1] * 1.05
            pred = max(0.0, pred)
            rows.append({'Kategori': cat, 'Tahun': base_yr + fy,
                         target: pred, 'Type': 'Prediksi'})
            history.append(pred)

    return pd.DataFrame(rows)

def compute_monthly_breakdown(df_raw_monthly, yearly_pred_df, target):
    rows = []
    for cat in yearly_pred_df['Kategori'].unique():
        cat_hist = df_raw_monthly[df_raw_monthly['Kategori'] == cat]
        if len(cat_hist) >= 12:
            mo = cat_hist.groupby(['Tahun','Bulan'])[target].sum().reset_index()
            yr = mo.groupby('Tahun')[target].sum().reset_index()
            yr.columns = ['Tahun','YrTotal']
            mo = mo.merge(yr, on='Tahun')
            mo['W'] = mo[target] / (mo['YrTotal'] + 1e-9)
            weights = mo.groupby('Bulan')['W'].mean()
        else:
            weights = pd.Series({m: 1/12 for m in range(1,13)})

        for m in range(1,13):
            if m not in weights.index:
                weights[m] = 0.0
        weights = weights.sort_index()
        wsum = weights.sum()
        if wsum > 0:
            weights = weights / wsum
        else:
            weights = pd.Series({m: 1/12 for m in range(1,13)})

        cat_pred = yearly_pred_df[yearly_pred_df['Kategori'] == cat]
        for _, row in cat_pred.iterrows():
            yr_total = float(row[target])
            yr_int   = int(row['Tahun'])
            for bulan, w in weights.items():
                rows.append({
                    'Kategori': cat,
                    'Tahun':    yr_int,
                    'Bulan':    int(bulan),
                    'Periode':  f"{yr_int}-{int(bulan):02d}",
                    target:     max(0.0, yr_total * w),
                    'Type':     'Prediksi Bulanan',
                })
    return pd.DataFrame(rows)

# ══════════════════════════════════════════════════════════════════════════════
# EXPORT EXCEL
# ══════════════════════════════════════════════════════════════════════════════

def _write_monthly_block(ws, wb, df_mo, target_col, hdr_fmt, num_fmt,
                         start_row, sheet_name, chart_title):
    if df_mo is None or len(df_mo) == 0:
        return start_row, None

    piv = (df_mo.pivot_table(index='Periode', columns='Kategori',
                             values=target_col, aggfunc='sum')
           .reset_index()
           .sort_values('Periode')
           .reset_index(drop=True))

    n_rows = len(piv)
    n_cats = len(piv.columns) - 1

    for col_idx, col_name in enumerate(piv.columns):
        ws.write(start_row, col_idx, str(col_name), hdr_fmt)
        ws.set_column(col_idx, col_idx, 16)

    for row_idx in range(n_rows):
        ws.write(start_row + 1 + row_idx, 0, piv.iloc[row_idx, 0])
        for col_idx in range(1, n_cats + 1):
            ws.write(start_row + 1 + row_idx, col_idx,
                     piv.iloc[row_idx, col_idx], num_fmt)

    ch = wb.add_chart({'type': 'line'})
    for col_idx in range(1, n_cats + 1):
        ch.add_series({
            'name':       [sheet_name, start_row,     col_idx],
            'categories': [sheet_name, start_row + 1, 0,        start_row + n_rows, 0],
            'values':     [sheet_name, start_row + 1, col_idx,  start_row + n_rows, col_idx],
            'marker':     {'type': 'circle', 'size': 4},
        })
    ch.set_title({'name': chart_title})
    ch.set_x_axis({'name': 'Periode (YYYY-MM)', 'num_font': {'rotation': -45}})
    ch.set_y_axis({'name': target_col})
    ch.set_legend({'position': 'bottom'})
    ch.set_size({'width': 760, 'height': 420})

    next_row = start_row + n_rows + 3
    return next_row, ch, start_row + n_rows


def export_excel(df, ml_result, fut_df,
                 fut_kasus=None, fut_nominal=None,
                 fut_monthly_kasus=None, fut_monthly_nominal=None):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
        wb      = writer.book
        hdr     = wb.add_format({'bold': True, 'bg_color': '#1e3a5f',
                                  'font_color': 'white', 'border': 1})
        num_fmt = wb.add_format({'num_format': '#,##0', 'border': 1})
        sec_fmt = wb.add_format({'bold': True, 'bg_color': '#0f2744',
                                  'font_color': '#93c5fd', 'font_size': 12,
                                  'border': 0})

        df_sorted = df.sort_values(['Tahun', 'Kategori']).reset_index(drop=True)
        df_sorted.to_excel(writer, sheet_name='Data Gabungan', index=False)
        ws1 = writer.sheets['Data Gabungan']
        for i, c in enumerate(df_sorted.columns):
            ws1.write(0, i, c, hdr)
            ws1.set_column(i, i, 22)

        if 'Kasus' in df.columns:
            piv = df.pivot_table(index='Kategori', columns='Tahun',
                                 values='Kasus', aggfunc='sum', fill_value=0)
            piv.to_excel(writer, sheet_name='Pivot Kasus')
            ws2 = writer.sheets['Pivot Kasus']
            for i in range(len(piv.columns) + 1):
                ws2.set_column(i, i, 18)
            nr, nc2 = len(piv), len(piv.columns)
            chart_pk = wb.add_chart({'type': 'line'})
            for i in range(nr):
                chart_pk.add_series({
                    'name':       ['Pivot Kasus', i+1, 0],
                    'categories': ['Pivot Kasus', 0, 1, 0, nc2],
                    'values':     ['Pivot Kasus', i+1, 1, i+1, nc2],
                })
            chart_pk.set_title({'name': 'Tren Kasus per Program'})
            chart_pk.set_size({'width': 600, 'height': 350})
            ws2.insert_chart(f'A{nr+4}', chart_pk)

        has_fut_k = fut_kasus   is not None and len(fut_kasus)   > 0
        has_fut_n = fut_nominal is not None and len(fut_nominal) > 0

        if not has_fut_k and not has_fut_n and fut_df is not None and len(fut_df) > 0:
            tc_fb = [c for c in fut_df.columns if c not in ['Kategori','Tahun','Type']]
            if tc_fb:
                if tc_fb[0] == 'Kasus':
                    fut_kasus   = fut_df; has_fut_k = True
                else:
                    fut_nominal = fut_df; has_fut_n = True

        if has_fut_k or has_fut_n:
            ws3_name = 'Prediksi Tahunan'
            writer.book.add_worksheet(ws3_name)
            ws3 = writer.sheets[ws3_name]

            hdr_sec  = wb.add_format({'bold': True, 'bg_color': '#0f2744',
                                       'font_color': '#93c5fd', 'font_size': 12})
            hdr_yr   = wb.add_format({'bold': True, 'bg_color': '#1e3a5f',
                                       'font_color': 'white', 'border': 1,
                                       'align': 'center'})
            num_k    = wb.add_format({'num_format': '#,##0',   'border': 1})
            num_hist = wb.add_format({'num_format': '#,##0',   'border': 1,
                                       'bg_color': '#1a1a2e'})

            cursor = 0

            def _write_annual_block(ws, wb, df_hist, df_pred, value_col,
                                    title_txt, cursor, ws_name):
                ph = (df_hist.groupby(['Tahun','Kategori'])[value_col].sum()
                      .reset_index()
                      .pivot(index='Tahun', columns='Kategori', values=value_col)
                      .reset_index())
                ph['_type'] = 'Aktual'

                pp = (df_pred.groupby(['Tahun','Kategori'])[value_col].sum()
                      .reset_index()
                      .pivot(index='Tahun', columns='Kategori', values=value_col)
                      .reset_index())
                pp['_type'] = 'Prediksi'

                combined = pd.concat([ph, pp], ignore_index=True)\
                             .sort_values('Tahun').reset_index(drop=True)

                cats = [c for c in combined.columns if c not in ['Tahun','_type']]
                n_rows = len(combined)
                n_cats = len(cats)

                ws.merge_range(cursor, 0, cursor, n_cats + 1, title_txt, hdr_sec)
                hdr_row = cursor + 1

                ws.write(hdr_row, 0, 'Tahun',   hdr_yr); ws.set_column(0, 0, 10)
                ws.write(hdr_row, 1, 'Tipe',    hdr_yr); ws.set_column(1, 1, 12)
                for ci, cat in enumerate(cats):
                    ws.write(hdr_row, ci + 2, cat, hdr_yr)
                    ws.set_column(ci + 2, ci + 2, 20)

                data_start = hdr_row + 1
                for ri, row in combined.iterrows():
                    r_abs = data_start + ri
                    fmt_use = num_hist if row['_type'] == 'Aktual' else num_k
                    ws.write(r_abs, 0, int(row['Tahun']))
                    ws.write(r_abs, 1, row['_type'])
                    for ci, cat in enumerate(cats):
                        val = row.get(cat, 0)
                        ws.write(r_abs, ci + 2, float(val) if pd.notna(val) else 0, fmt_use)

                last_data_row = data_start + n_rows - 1

                ch = wb.add_chart({'type': 'line'})

                aktual_rows  = [data_start + i for i, r in combined.iterrows()
                                if r['_type'] == 'Aktual']
                pred_rows    = [data_start + i for i, r in combined.iterrows()
                                if r['_type'] == 'Prediksi']

                CHART_COLORS = ['#4472C4','#ED7D31','#A9D18E','#FF0000',
                                '#7030A0','#00B0F0','#92D050','#FFC000']

                for ci, cat in enumerate(cats):
                    col_excel = ci + 2
                    color     = CHART_COLORS[ci % len(CHART_COLORS)]

                    if aktual_rows:
                        ch.add_series({
                            'name':   cat + ' Aktual',
                            'categories': [ws_name, aktual_rows[0],  0,
                                           aktual_rows[-1], 0],
                            'values':     [ws_name, aktual_rows[0],  col_excel,
                                           aktual_rows[-1], col_excel],
                            'line':   {'color': color, 'width': 2.25},
                            'marker': {'type': 'circle', 'size': 6,
                                       'fill': {'color': color},
                                       'border': {'color': color}},
                        })

                    if pred_rows:
                        ch.add_series({
                            'name':   cat + ' Prediksi',
                            'categories': [ws_name, pred_rows[0],  0,
                                           pred_rows[-1], 0],
                            'values':     [ws_name, pred_rows[0],  col_excel,
                                           pred_rows[-1], col_excel],
                            'line':   {'color': color, 'width': 2.25, 'dash_type': 'dash'},
                            'marker': {'type': 'diamond', 'size': 7,
                                       'fill': {'color': color},
                                       'border': {'color': color}},
                        })

                ch.set_title({'name': title_txt})
                ch.set_x_axis({
                    'name': 'Tahun',
                    'major_gridlines': {'visible': True, 'line': {'color': '#e0e0e0'}},
                    'num_font': {'size': 10},
                })
                ch.set_y_axis({
                    'name': value_col,
                    'major_gridlines': {'visible': True, 'line': {'color': '#e0e0e0'}},
                    'num_format': '#,##0',
                })
                ch.set_legend({'position': 'bottom', 'font': {'size': 9}})
                ch.set_size({'width': 800, 'height': 450})
                ch.set_style(10)

                chart_col = xl_col_to_name(n_cats + 3)
                ws.insert_chart(f'{chart_col}{hdr_row + 1}', ch)

                return last_data_row + 4

            df_hist_k = df[df['Kategori'].isin(
                df['Kategori'].unique())][['Tahun','Kategori','Kasus']].copy() \
                if 'Kasus' in df.columns else None
            df_hist_n = df[df['Kategori'].isin(
                df['Kategori'].unique())][['Tahun','Kategori','Nominal']].copy() \
                if 'Nominal' in df.columns else None

            if has_fut_k and df_hist_k is not None:
                cursor = _write_annual_block(
                    ws3, wb, df_hist_k, fut_kasus, 'Kasus',
                    '📊 PREDIKSI KASUS (TAHUNAN) — Aktual vs Proyeksi',
                    cursor, ws3_name
                )

            if has_fut_n and df_hist_n is not None:
                cursor = _write_annual_block(
                    ws3, wb, df_hist_n, fut_nominal, 'Nominal',
                    '💰 PREDIKSI NOMINAL (TAHUNAN) — Aktual vs Proyeksi',
                    cursor, ws3_name
                )

        has_kasus   = fut_monthly_kasus   is not None and len(fut_monthly_kasus)   > 0
        has_nominal = fut_monthly_nominal is not None and len(fut_monthly_nominal) > 0

        if has_kasus or has_nominal:
            ws4_name = 'Prediksi Bulanan'
            writer.book.add_worksheet(ws4_name)
            ws4 = writer.sheets[ws4_name]

            cursor = 0

            if has_kasus:
                ws4.merge_range(cursor, 0, cursor, 7,
                                '📊 PREDIKSI KASUS (BULANAN)', sec_fmt)
                cursor += 1

                piv_k = (fut_monthly_kasus
                         .sort_values(['Tahun','Bulan','Kategori'])
                         .pivot_table(index='Periode', columns='Kategori',
                                      values='Kasus', aggfunc='sum')
                         .reset_index()
                         .sort_values('Periode')
                         .reset_index(drop=True))
                nrow_k = len(piv_k)
                ncat_k = len(piv_k.columns) - 1

                header_row_k = cursor
                for ci, cn in enumerate(piv_k.columns):
                    ws4.write(cursor, ci, str(cn), hdr)
                    ws4.set_column(ci, ci, 16)
                cursor += 1

                for ri in range(nrow_k):
                    ws4.write(cursor + ri, 0, piv_k.iloc[ri, 0])
                    for ci in range(1, ncat_k + 1):
                        ws4.write(cursor + ri, ci, piv_k.iloc[ri, ci], num_fmt)
                last_data_row_k = cursor + nrow_k - 1

                ch_k = wb.add_chart({'type': 'line'})
                for ci in range(1, ncat_k + 1):
                    ch_k.add_series({
                        'name':       [ws4_name, header_row_k, ci],
                        'categories': [ws4_name, header_row_k + 1, 0,
                                       last_data_row_k, 0],
                        'values':     [ws4_name, header_row_k + 1, ci,
                                       last_data_row_k, ci],
                        'marker':     {'type': 'circle', 'size': 4},
                    })
                ch_k.set_title({'name': 'Prediksi Kasus per Program (Bulanan)'})
                ch_k.set_x_axis({'name': 'Periode (YYYY-MM)',
                                  'num_font': {'rotation': -45}})
                ch_k.set_y_axis({'name': 'Kasus'})
                ch_k.set_legend({'position': 'bottom'})
                ch_k.set_size({'width': 760, 'height': 420})
                chart_col_k = xl_col_to_name(ncat_k + 2)
                ws4.insert_chart(f'{chart_col_k}{header_row_k + 1}', ch_k)

                cursor = last_data_row_k + 3

            if has_nominal:
                ws4.merge_range(cursor, 0, cursor, 7,
                                '💰 PREDIKSI NOMINAL (BULANAN)', sec_fmt)
                cursor += 1

                piv_n = (fut_monthly_nominal
                         .sort_values(['Tahun','Bulan','Kategori'])
                         .pivot_table(index='Periode', columns='Kategori',
                                      values='Nominal', aggfunc='sum')
                         .reset_index()
                         .sort_values('Periode')
                         .reset_index(drop=True))
                nrow_n = len(piv_n)
                ncat_n = len(piv_n.columns) - 1

                header_row_n = cursor
                for ci, cn in enumerate(piv_n.columns):
                    ws4.write(cursor, ci, str(cn), hdr)
                    ws4.set_column(ci, ci, 16)
                cursor += 1

                for ri in range(nrow_n):
                    ws4.write(cursor + ri, 0, piv_n.iloc[ri, 0])
                    for ci in range(1, ncat_n + 1):
                        ws4.write(cursor + ri, ci, piv_n.iloc[ri, ci], num_fmt)
                last_data_row_n = cursor + nrow_n - 1

                ch_n = wb.add_chart({'type': 'line'})
                for ci in range(1, ncat_n + 1):
                    ch_n.add_series({
                        'name':       [ws4_name, header_row_n, ci],
                        'categories': [ws4_name, header_row_n + 1, 0,
                                       last_data_row_n, 0],
                        'values':     [ws4_name, header_row_n + 1, ci,
                                       last_data_row_n, ci],
                        'marker':     {'type': 'circle', 'size': 4},
                    })
                ch_n.set_title({'name': 'Prediksi Nominal per Program (Bulanan)'})
                ch_n.set_x_axis({'name': 'Periode (YYYY-MM)',
                                  'num_font': {'rotation': -45}})
                ch_n.set_y_axis({'name': 'Nominal (Rp)'})
                ch_n.set_legend({'position': 'bottom'})
                ch_n.set_size({'width': 760, 'height': 420})
                chart_col_n = xl_col_to_name(ncat_n + 2)
                ws4.insert_chart(f'{chart_col_n}{header_row_n + 1}', ch_n)

        detail_frames = []
        if has_kasus:
            tmp_k = fut_monthly_kasus[
                ['Periode','Tahun','Bulan','Kategori','Kasus']].copy()
            detail_frames.append(tmp_k)
        if has_nominal:
            tmp_n = fut_monthly_nominal[
                ['Periode','Tahun','Bulan','Kategori','Nominal']].copy()
            detail_frames.append(tmp_n)

        if detail_frames:
            if len(detail_frames) == 2:
                detail_all = detail_frames[0].merge(
                    detail_frames[1], on=['Periode','Tahun','Bulan','Kategori'],
                    how='outer')
            else:
                detail_all = detail_frames[0]

            detail_all = (detail_all
                          .sort_values(['Tahun','Bulan','Kategori'])
                          .reset_index(drop=True))
            detail_all.to_excel(writer, sheet_name='Bulanan Detail', index=False)
            ws5 = writer.sheets['Bulanan Detail']
            for i, c in enumerate(detail_all.columns):
                ws5.write(0, i, c, hdr)
                ws5.set_column(i, i, 18)
            for ci, col_name in enumerate(detail_all.columns):
                if col_name in ('Kasus', 'Nominal'):
                    for ri in range(len(detail_all)):
                        ws5.write(ri + 1, ci, detail_all.iloc[ri][col_name], num_fmt)

        if ml_result:
            rdf = ml_result['results_df']
            rdf.to_excel(writer, sheet_name='ML Results', index=False)
            ws6 = writer.sheets['ML Results']
            for i, c in enumerate(rdf.columns):
                ws6.write(0, i, c, hdr)
                ws6.set_column(i, i, 18)
            n = len(rdf)
            ch_ml = wb.add_chart({'type': 'column'})
            ch_ml.add_series({
                'name':       'R² Score',
                'categories': ['ML Results', 1, 0, n, 0],
                'values':     ['ML Results', 1, 3, n, 3],
                'fill':       {'color': '#3b82f6'},
            })
            ch_ml.set_title({'name': 'Model Comparison – R²'})
            ch_ml.set_size({'width': 500, 'height': 300})
            ws6.insert_chart('H2', ch_ml)

    buf.seek(0)
    return buf.getvalue()


def xl_col_to_name(col_idx):
    name = ''
    col_idx += 1
    while col_idx:
        col_idx, remainder = divmod(col_idx - 1, 26)
        name = chr(65 + remainder) + name
    return name

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="padding:16px 0 8px;text-align:center;">
      <div style="font-size:1.3rem;font-weight:800;
        background:linear-gradient(135deg,#60a5fa,#a78bfa);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        📊 BPJS ML
      </div>
      <div style="font-size:.68rem;color:#334155;letter-spacing:1px;
        text-transform:uppercase;margin-top:2px;">Prediction Dashboard</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    uploaded = st.file_uploader(
        "Upload dataset (nama file tidak harus mengandung tahun)",
        type=['csv', 'xlsx', 'xls'],
        accept_multiple_files=True,
        help="Contoh nama file: data_tahun_2021.xlsx, bpjs.csv"
    )
    st.markdown("""
    <div style="font-size:.78rem;color:#64748b;line-height:1.7;margin-top:6px">
    Kolom yang dibaca:<br>
    <code>PROGRAM</code> → Kategori<br>
    <code>KASUS</code> → Jumlah kasus<br>
    <code>NOMINAL</code> → Nominal (Rp)<br>
    <code>DATE</code> → Periode (diagregasi/tahun)<br>
    Prediksi hanya untuk program yang <b>aktif di tahun terbaru</b>.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    n_lags   = st.slider("Lag features", 1, 4, 2)
    test_pct = st.slider("Test split (%)", 10, 40, 25, 5)
    n_future = st.slider("Prediksi tahun ke depan", 1, 5, 3)

    st.markdown("---")
    st.markdown("**🕑 Riwayat Analisis**")
    st.caption("Riwayat tersimpan permanen — tidak hilang saat restart.")
    history_meta = load_history_meta()
    if history_meta:
        for h in reversed(history_meta):
            col_h, col_del = st.columns([5, 1])
            with col_h:
                if st.button(h['label'], key=f"hbtn_{h['id']}", width='stretch'):
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
                if st.button("🗑", key=f"hdel_{h['id']}", help="Hapus riwayat ini"):
                    delete_history_entry(h['id'])
                    meta = load_history_meta()
                    meta = [m for m in meta if m['id'] != h['id']]
                    save_history_meta(meta)
                    st.rerun()
        if st.button("🗑 Hapus Semua Riwayat", width='stretch'):
            for h in history_meta:
                delete_history_entry(h['id'])
            save_history_meta([])
            st.rerun()
    else:
        st.caption("Belum ada riwayat tersimpan.")



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
            st.sidebar.success(f"✅ {f.name} → {year_hint}")
        else:
            st.sidebar.error(f"❌ {f.name}: gagal dibaca")

    if files_info:
        merged, errs = merge_all(files_info)
        for e in errs:
            st.sidebar.warning(e)
        if merged is not None and len(merged) > 0:
            dh  = hashlib.md5(merged.to_csv().encode()).hexdigest()[:8]
            cur = (hashlib.md5(st.session_state.active_data.to_csv().encode())
                   .hexdigest()[:8] if st.session_state.active_data is not None else None)
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
                st.sidebar.success(f"📆 {len(rm_combined)} baris monthly tersimpan")
            else:
                st.session_state['raw_monthly'] = None
        else:
            st.error("Gagal memproses data. Pastikan file punya kolom PROGRAM dan KASUS.")

df            = st.session_state.active_data
results_cache = st.session_state.active_results
df_raw_monthly = st.session_state.get('raw_monthly', None)

if df is None:
    st.markdown("""
    <div style="text-align:center;padding:80px 0 40px;">
      <div style="font-size:4rem;margin-bottom:16px;">📊</div>
      <div style="font-size:2rem;font-weight:800;
        background:linear-gradient(135deg,#60a5fa,#a78bfa);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        margin-bottom:12px;">
        Dashboard Prediksi Klaim BPJS Ketenagakerjaan
      </div>
      <div style="color:#475569;max-width:600px;margin:auto;font-size:.95rem;line-height:1.8;">
        Upload file dataset Anda untuk memulai analisis prediktif klaim.
      </div>
    </div>""", unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)
    for col, icon, title, desc in [
        (f1, "🤖", "Metode Adaptif", "Holt Smoothing, SES, WMA untuk data kecil. ML (XGBoost, RF) otomatis aktif untuk data ≥ 8 tahun."),
        (f2, "📅", "Kalender Indonesia", "Prophet + Google Calendar Indonesia. Semua hari libur nasional otomatis diambil dari API resmi Google — Nyepi, Paskah, Lebaran, Natal, dan lainnya."),
        (f3, "📥", "Export Excel", "Export prediksi tahunan & bulanan ke Excel dengan chart otomatis, siap untuk presentasi."),
    ]:
        with col:
            st.markdown(f'''<div style="background:#0a1628;border:1px solid #1e2d45;
            border-radius:14px;padding:24px;text-align:center;height:160px;
            border-top:3px solid #3b82f6;">
            <div style="font-size:2rem;margin-bottom:8px;">{icon}</div>
            <div style="font-weight:700;color:#e2e8f0;margin-bottom:8px;font-size:.95rem;">{title}</div>
            <div style="color:#475569;font-size:.82rem;line-height:1.6;">{desc}</div>
            </div>''', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="info-box">📋 <b>Format kolom yang didukung:</b> <code>PROGRAM</code> (nama jaminan) · <code>KASUS</code> (jumlah klaim) · <code>NOMINAL</code> (nilai Rp) · <code>DATE</code> (periode)<br>Upload 1+ file CSV/Excel. Nama file tidak harus mengandung tahun.</div>', unsafe_allow_html=True)
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

# ══════════════════════════════════════════════════════════════════════════════
# DEBUG EXPANDER
# ══════════════════════════════════════════════════════════════════════════════

with st.expander("🔍 Info Parsing & Program Aktif (klik untuk cek)", expanded=False):
    change_html = ""
    for (y0, y1), ch in prog_changes.items():
        change_html += f"<b>{y0} → {y1}:</b> &nbsp;"
        for p in ch['added']:
            change_html += f'<span class="tag-add">+ {p}</span> '
        for p in ch['removed']:
            change_html += f'<span class="tag-rem">– {p}</span> '
        for p in ch['stable']:
            change_html += f'<span class="tag-stable">{p}</span> '
        change_html += "<br>"

    st.markdown(f"""
    <div class="info-box">
    📋 <b>Kolom terbaca:</b> {', '.join(df.columns.tolist())}<br>
    📅 <b>Tahun terdeteksi:</b> {', '.join(map(str, years))}<br>
    🏷️ <b>Semua program ({len(all_progs)}):</b> {', '.join(all_progs)}<br>
    ✅ <b>Program aktif (tahun {latest_year}) → diprediksi ({len(active_progs)}):</b>
       {', '.join(active_progs)}<br>
    📊 <b>Total baris setelah agregasi:</b> {len(df)}<br><br>
    <b>Perubahan Program per Tahun:</b><br>
    {change_html if change_html else 'Hanya 1 tahun data.'}
    </div>""", unsafe_allow_html=True)

    st.markdown("**Verifikasi Nilai per Tahun per Program (setelah agregasi):**")
    verify_cols = ['Tahun', 'Kategori', 'Kasus']
    if 'Nominal' in df.columns:
        verify_cols.append('Nominal')
    vdf = df[verify_cols].copy().sort_values(['Tahun','Kategori'])
    if 'Nominal' in vdf.columns:
        vdf['Nominal (T)'] = (vdf['Nominal'] / 1e12).round(4)
        vdf['Nominal (B)'] = (vdf['Nominal'] / 1e9).round(2)
    st.dataframe(vdf, width='stretch', height=320)

# ══════════════════════════════════════════════════════════════════════════════
# KPIs
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    '<div class="hero-wrap">'
    '<div class="hero-logo">📊 Dashboard Prediksi Klaim BPJS Ketenagakerjaan</div>'
    '<div class="hero-sub">Analisis tren historis &amp; proyeksi — model adaptif statistik &amp; machine learning per program</div>'
    '</div>',
    unsafe_allow_html=True)

if single_yr:
    st.markdown("""<div class="warn">⚠️ <b>Mode 1 Tahun:</b>
    Prediksi menggunakan ekstrapolasi asumsi pertumbuhan 5%/tahun.
    Upload data multi-tahun untuk prediksi ML penuh.</div>""", unsafe_allow_html=True)

if prog_changes:
    last_change = list(prog_changes.items())[-1]
    (y0, y1), ch = last_change
    if ch['added'] or ch['removed']:
        added_str   = ', '.join(ch['added']) if ch['added'] else '–'
        removed_str = ', '.join(ch['removed']) if ch['removed'] else '–'
        st.markdown(f"""<div class="warn">
        📌 <b>Perubahan program {y0}→{y1}:</b>
        &nbsp; Ditambah: <b style="color:#86efac">{added_str}</b>
        &nbsp;|&nbsp; Dihapus: <b style="color:#fca5a5">{removed_str}</b>
        &nbsp;→ Prediksi hanya untuk program aktif tahun {y1}.
        </div>""", unsafe_allow_html=True)

# ── Program filter widget (sticky sidebar, poin 2 & 5) ──────────────────────
with st.sidebar:
    st.markdown("---")
    st.markdown("**🎯 Filter Program**")
    st.caption("Berlaku global untuk semua tab.")
    selected_filter = st.multiselect(
        "Tampilkan program:",
        options=list(active_progs),
        default=list(active_progs),
        key='prog_filter_widget',
        help="Pilih 1+ program. Biarkan kosong = tampilkan semua."
    )
    if not selected_filter:
        selected_filter = list(active_progs)

filtered_progs = selected_filter

df_active_only = df[df['Kategori'].isin(active_progs)]

kpi_delta_k = kpi_delta_n = kpi_avg_growth = ""
if len(years) >= 2:
    yr_kasus = df_active_only.groupby('Tahun')['Kasus'].sum()
    if yr_kasus.iloc[-1] > 0 and yr_kasus.iloc[-2] > 0:
        delta_k_pct = (yr_kasus.iloc[-1] / yr_kasus.iloc[-2] - 1) * 100
        sign_k = "▲" if delta_k_pct >= 0 else "▼"
        cls_k  = "delta-pos" if delta_k_pct >= 0 else "delta-neg"
        kpi_delta_k = f'<div class="delta {cls_k}">{sign_k} {abs(delta_k_pct):.1f}% vs {years[-2]}</div>'
    if has_nom:
        yr_nom = df_active_only.groupby('Tahun')['Nominal'].sum()
        if yr_nom.iloc[-1] > 0 and yr_nom.iloc[-2] > 0:
            delta_n_pct = (yr_nom.iloc[-1] / yr_nom.iloc[-2] - 1) * 100
            sign_n = "▲" if delta_n_pct >= 0 else "▼"
            cls_n  = "delta-pos" if delta_n_pct >= 0 else "delta-neg"
            kpi_delta_n = f'<div class="delta {cls_n}">{sign_n} {abs(delta_n_pct):.1f}% vs {years[-2]}</div>'
    growths = []
    for i in range(1, len(years)):
        k_prev = df_active_only[df_active_only['Tahun']==years[i-1]]['Kasus'].sum()
        k_curr = df_active_only[df_active_only['Tahun']==years[i]]['Kasus'].sum()
        if k_prev > 0:
            growths.append((k_curr/k_prev - 1)*100)
    avg_g = np.mean(growths) if growths else 0
    sign_g = "▲" if avg_g >= 0 else "▼"
    cls_g  = "delta-pos" if avg_g >= 0 else "delta-neg"
    kpi_avg_growth = f'<div class="delta {cls_g}">{sign_g} {abs(avg_g):.1f}%/thn rata-rata</div>'

tk = int(df_active_only['Kasus'].sum())
tk_latest = int(df_active_only[df_active_only['Tahun']==latest_year]['Kasus'].sum())

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(f'''<div class="kpi">
    <div class="val">{len(years)}</div>
    <div class="lbl">📅 Tahun Data</div>
    <div class="delta delta-neu">{years[0]} – {years[-1]}</div>
    </div>''', unsafe_allow_html=True)
with c2:
    st.markdown(f'''<div class="kpi">
    <div class="val">{len(active_progs)}</div>
    <div class="lbl">🏷️ Program Aktif</div>
    <div class="delta delta-neu">{", ".join(active_progs)}</div>
    </div>''', unsafe_allow_html=True)
with c3:
    st.markdown(f'''<div class="kpi">
    <div class="val">{tk_latest:,}</div>
    <div class="lbl">📋 Kasus {latest_year}</div>
    {kpi_delta_k}
    </div>''', unsafe_allow_html=True)
with c4:
    if has_nom:
        tn = df_active_only[df_active_only['Tahun']==latest_year]['Nominal'].sum()/1e9
        st.markdown(f'''<div class="kpi">
        <div class="val">Rp{tn:,.1f}B</div>
        <div class="lbl">💰 Nominal {latest_year}</div>
        {kpi_delta_n}
        </div>''', unsafe_allow_html=True)
    else:
        st.markdown(f'''<div class="kpi">
        <div class="val">{latest_year}</div>
        <div class="lbl">📅 Tahun Terbaru</div>
        </div>''', unsafe_allow_html=True)
with c5:
    total_all = f"{tk:,}"
    st.markdown(f'''<div class="kpi">
    <div class="val">{total_all}</div>
    <div class="lbl">📊 Total Kasus (semua)</div>
    {kpi_avg_growth}
    </div>''', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
df_plot = df[df['Kategori'].isin(filtered_progs)].copy()

# Filter status indicator
if len(filtered_progs) < len(active_progs):
    filter_str = ", ".join(filtered_progs)
    st.markdown(
        f'<div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;'
        f'padding:8px 14px;font-size:.82rem;color:#1e40af;margin-top:4px;">'
        f'🔵 <b>Filter aktif:</b> {len(filtered_progs)} dari {len(active_progs)} program'
        f' &nbsp;|&nbsp; <b>{filter_str}</b></div>',
        unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "🤖 ML Analysis", "🔮 Prediksi", "📥 Export"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    df_lat = df_plot[df_plot['Tahun'] == latest_year]

    if not single_yr:
        top_prog = df_lat.groupby('Kategori')['Kasus'].sum().idxmax()
        top_val  = int(df_lat.groupby('Kategori')['Kasus'].sum().max())
        growth_by_prog = {}
        for cp in active_progs:
            cd = df_plot[df_plot['Kategori']==cp].sort_values('Tahun')
            if len(cd) >= 2 and cd['Kasus'].iloc[-2] > 0:
                growth_by_prog[cp] = (cd['Kasus'].iloc[-1]/cd['Kasus'].iloc[-2]-1)*100
        fastest = max(growth_by_prog, key=growth_by_prog.get) if growth_by_prog else "-"
        fastest_g = growth_by_prog.get(fastest, 0)
        slowest = min(growth_by_prog, key=growth_by_prog.get) if growth_by_prog else "-"
        slowest_g = growth_by_prog.get(slowest, 0)
        total_latest = int(df_lat['Kasus'].sum())

        ia, ib, ic, id_ = st.columns(4)
        with ia:
            st.markdown(f'''<div class="insight-card">
            <div class="ic-title">🏆 Program Terbesar</div>
            <div class="ic-val">{top_prog}</div>
            <div class="ic-sub">{top_val:,} kasus di {latest_year}</div>
            </div>''', unsafe_allow_html=True)
        with ib:
            st.markdown(f'''<div class="insight-card">
            <div class="ic-title">📈 Pertumbuhan Tertinggi</div>
            <div class="ic-val" style="color:#34d399">{fastest}</div>
            <div class="ic-sub">+{fastest_g:.1f}% vs tahun lalu</div>
            </div>''', unsafe_allow_html=True)
        with ic:
            st.markdown(f'''<div class="insight-card">
            <div class="ic-title">📉 Pertumbuhan Terendah</div>
            <div class="ic-val" style="color:#f87171">{slowest}</div>
            <div class="ic-sub">{slowest_g:+.1f}% vs tahun lalu</div>
            </div>''', unsafe_allow_html=True)
        with id_:
            st.markdown(f'''<div class="insight-card">
            <div class="ic-title">📋 Total Kasus {latest_year}</div>
            <div class="ic-val">{total_latest:,}</div>
            <div class="ic-sub">{len(active_progs)} program aktif</div>
            </div>''', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    r1, r2 = st.columns(2)
    with r1:
        st.markdown('<div class="sec">Distribusi Kasus — Semua Tahun (Program Aktif)</div>',
                    unsafe_allow_html=True)
        st.markdown('<div style="font-size:.77rem;color:#64748b;margin-bottom:6px;">Proporsi total klaim per program dari seluruh periode historis. Klik legenda untuk isolasi program.</div>', unsafe_allow_html=True)
        pie_d = df_plot.groupby('Kategori')['Kasus'].sum().reset_index()
        pie_d['pct'] = pie_d['Kasus'] / pie_d['Kasus'].sum() * 100
        fig = px.pie(pie_d, names='Kategori', values='Kasus', hole=0.5,
                     color_discrete_sequence=COLORS,
                     custom_data=['pct'])
        fig.update_traces(
            textinfo='label+percent', textposition='outside',
            hovertemplate='<b>%{label}</b><br>Kasus: %{value:,}<br>%{percent}<extra></extra>',
            textfont_size=11)
        fig.update_layout(**DARK, showlegend=True, height=400,
                          legend=dict(orientation='h', y=-0.1, font=dict(size=10)),
                          margin=dict(t=10, b=60, l=10, r=10))
        total_kasus = int(pie_d['Kasus'].sum())
        fig.add_annotation(text=f"<b>{total_kasus:,}</b><br><span style='font-size:10px'>Total</span>",
            showarrow=False, font=dict(size=13, color='#e2e8f0'), align='center')
        st.plotly_chart(fig, width='stretch')

    with r2:
        st.markdown(f'<div class="sec">Market Share per Program — {latest_year}</div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:.77rem;color:#64748b;margin-bottom:6px;">Jumlah & persentase klaim per program di tahun terbaru ({latest_year}). Bar lebih panjang = porsi klaim lebih besar.</div>', unsafe_allow_html=True)
        bar_d = (df_lat.groupby('Kategori')['Kasus'].sum()
                 .sort_values(ascending=True).reset_index())
        total_bar = bar_d['Kasus'].sum()
        bar_d['Share'] = (bar_d['Kasus']/total_bar*100).round(1)
        
        fig2 = go.Figure()
        for i, row in bar_d.iterrows():
            col_c = COLORS[i % len(COLORS)]
            fig2.add_trace(go.Bar(
                x=[row['Kasus']], y=[row['Kategori']], orientation='h',
                name=row['Kategori'],
                marker_color=col_c,
                marker_line_width=0,
                text=f"{row['Kasus']:,} ({row['Share']}%)",
                textposition='outside',
                textfont=dict(size=11),
                showlegend=False,
                hovertemplate=f"<b>{row['Kategori']}</b><br>{row['Kasus']:,} kasus ({row['Share']}%)<extra></extra>"
            ))
        fig2.update_layout(**DARK, height=400, showlegend=False, barmode='overlay',
                           margin=dict(t=10, b=10, l=10, r=120),
                           xaxis=dict(showgrid=True, gridcolor='#0f1923'))
        st.plotly_chart(fig2, width='stretch')

    if not single_yr:
        trend = df_plot.groupby(['Tahun', 'Kategori'])['Kasus'].sum().reset_index()

        # Konteks ekonomi Indonesia per tahun (poin 4 - hardcode)
        # ── AI-powered economic context (poin 4) ───────────────────────────────────
        def _get_ai_ekon_context(years_list):
            """Fetch Indonesian economic context per year via Anthropic API.
            Cached in session_state — no repeated calls on re-render.
            Works for any year range, future-proof.
            """
            cache_key = f"ekon_ctx_{'_'.join(map(str, sorted(years_list)))}"
            if cache_key in st.session_state:
                return st.session_state[cache_key]

            import urllib.request, json as json_lib

            # ── Get API key — Streamlit Cloud secrets ─────────────────────
            api_key = ""
            try:
                # Streamlit secrets can be accessed as dict or attribute
                secrets = st.secrets
                if hasattr(secrets, '__getitem__'):
                    try:    api_key = secrets["ANTHROPIC_API_KEY"]
                    except: pass
                if not api_key:
                    try:    api_key = secrets["anthropic_api_key"]
                    except: pass
                if not api_key:
                    try:    api_key = str(secrets.ANTHROPIC_API_KEY)
                    except: pass
            except Exception as secrets_err:
                pass

            if not api_key:
                st.session_state[cache_key] = {"_error": "ANTHROPIC_API_KEY tidak ditemukan di Secrets."}
                return {}

            yrs_str = ", ".join(map(str, sorted(years_list)))
            prompt = (
                f"Berikan konteks ekonomi makro Indonesia untuk tahun-tahun ini: {yrs_str}.\n"
                "Fokus: PDB growth (%), inflasi, UMP, kondisi pasar kerja, PHK besar, "
                "dan kebijakan pemerintah yang relevan dengan klaim BPJS Ketenagakerjaan.\n"
                "PENTING: Jawab HANYA dengan JSON valid, tanpa teks tambahan, tanpa markdown fences.\n"
                "Format persis:\n"
                "{\n"
                '  "2020": {"icon": "⚠️ Dampak COVID-19", "desc": "PDB -2.07%. PHK massal 2.56 juta. Klaim JHT melonjak."},\n'
                '  "2021": {"icon": "🔄 Pemulihan", "desc": "PDB +3.69%. PPKM masih aktif. Backlog klaim JHT."}\n'
                "}\n"
                "icon = emoji + label singkat max 4 kata. desc = max 55 kata, padat, ada angka statistik."
            )

            try:
                payload = json_lib.dumps({
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 1200,
                    "messages": [{"role": "user", "content": prompt}]
                }).encode('utf-8')

                req = urllib.request.Request(
                    "https://api.anthropic.com/v1/messages",
                    data=payload,
                    headers={
                        "Content-Type": "application/json",
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                    },
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=25) as r:
                    resp_bytes = r.read()
                resp_json = json_lib.loads(resp_bytes.decode('utf-8'))

                if "error" in resp_json:
                    err_msg = resp_json["error"].get("message", str(resp_json["error"]))
                    st.session_state[cache_key] = {"_error": f"API error: {err_msg}"}
                    return {}

                raw_text = resp_json["content"][0]["text"].strip()

                # Strip any markdown fences
                if "```" in raw_text:
                    import re as _re
                    raw_text = _re.sub(r"```[a-z]*\n?", "", raw_text).replace("```", "").strip()

                # Find JSON object in response (handles extra text)
                json_start = raw_text.find("{")
                json_end   = raw_text.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    raw_text = raw_text[json_start:json_end]

                parsed = json_lib.loads(raw_text)

                result = {}
                for yr_str, val in parsed.items():
                    try:
                        yr_int = int(str(yr_str).strip())
                        icon = val.get("icon", "📊") if isinstance(val, dict) else "📊"
                        desc = val.get("desc", "")  if isinstance(val, dict) else str(val)
                        result[yr_int] = (icon, desc)
                    except Exception:
                        pass

                st.session_state[cache_key] = result
                return result

            except urllib.error.HTTPError as http_err:
                err_body = ""
                try:    err_body = http_err.read().decode('utf-8')[:200]
                except: pass
                st.session_state[cache_key] = {"_error": f"HTTP {http_err.code}: {err_body}"}
                return {}
            except Exception as e:
                st.session_state[cache_key] = {"_error": str(e)[:200]}
                return {}

        all_yrs_data_for_ctx = sorted(trend['Tahun'].unique().tolist())

        # Hapus cache lama yang berisi error agar bisa retry otomatis
        _ck = f"ekon_ctx_{'_'.join(map(str, all_yrs_data_for_ctx))}"
        if _ck in st.session_state and isinstance(st.session_state[_ck], dict) and "_error" in st.session_state[_ck]:
            del st.session_state[_ck]

        _ekon_raw = None
        with st.spinner("🤖 Mengambil konteks ekonomi via AI..."):
            _ekon_raw = _get_ai_ekon_context(all_yrs_data_for_ctx)

        if isinstance(_ekon_raw, dict) and "_error" in _ekon_raw:
            st.markdown(
                f'<div class="warn">⚠️ <b>Konteks AI gagal:</b> {_ekon_raw["_error"]}<br>'
                f'Pastikan <code>ANTHROPIC_API_KEY</code> sudah benar di Streamlit Secrets.</div>',
                unsafe_allow_html=True)
            EKON_CONTEXT = {}
        elif not _ekon_raw:
            st.markdown(
                '<div class="warn">⚠️ API tidak mengembalikan data. Coba refresh halaman.</div>',
                unsafe_allow_html=True)
            EKON_CONTEXT = {}
        else:
            EKON_CONTEXT = _ekon_raw

        st.markdown('<div class="sec">Tren Kasus per Tahun — dengan Konteks Ekonomi AI</div>', unsafe_allow_html=True)
        fig3 = go.Figure()
        for i, cat in enumerate(sorted(df_plot['Kategori'].unique())):
            cd = trend[trend['Kategori']==cat].sort_values('Tahun')
            col_c = COLORS[i % len(COLORS)]
            fig3.add_trace(go.Scatter(
                x=cd['Tahun'], y=cd['Kasus'],
                name=cat, mode='lines+markers',
                line=dict(color=col_c, width=2.5),
                marker=dict(size=9, color=col_c,
                    line=dict(color='rgba(255,255,255,0.7)', width=1.5)),
                fill='tozeroy', fillcolor=hex_to_rgba(col_c, 0.07),
                hovertemplate=f"<b>{cat}</b><br>Tahun: %{{x}}<br>Kasus: %{{y:,}}<extra></extra>"
            ))

        all_yrs_data = all_yrs_data_for_ctx
        tot_yr = trend.groupby('Tahun')['Kasus'].sum()
        yr_max_t = int(tot_yr.max()) if len(tot_yr) > 0 else 1
        if 2020 in all_yrs_data:
            fig3.add_vrect(x0=2019.6, x1=2020.4, fillcolor='rgba(239,68,68,0.07)', line_width=0)
            fig3.add_annotation(x=2020, y=yr_max_t * 0.9,
                text="⚠️ COVID-19<br>PDB -2.07%", showarrow=False,
                font=dict(size=9, color='#dc2626', family='Inter'),
                bgcolor='rgba(254,226,226,0.92)', bordercolor='#fca5a5', borderwidth=1, borderpad=4)
        if 2022 in all_yrs_data:
            fig3.add_annotation(x=2022, y=yr_max_t * 0.82,
                text="🚀 Recovery<br>PDB +5.31%", showarrow=False,
                font=dict(size=9, color='#16a34a', family='Inter'),
                bgcolor='rgba(220,252,231,0.92)', bordercolor='#86efac', borderwidth=1, borderpad=4)
        styled_chart(fig3, height=480)
        fig3.update_layout(xaxis=dict(dtick=1, showgrid=True, gridcolor='rgba(148,163,184,0.3)'))
        st.plotly_chart(fig3, width='stretch')

        st.markdown(
            '<div class="info-box">'
            '💡 <b>Cara baca chart:</b> Area fill = volume klaim per program. '
            'Kotak anotasi menandai peristiwa ekonomi yang signifikan mempengaruhi klaim BPJS TK. '
            'Filter program di sidebar kiri untuk fokus analisis per program.</div>',
            unsafe_allow_html=True)

        # Peak & Trough analysis (poin 4)
        st.markdown('<div class="sec">📍 Analisis Peak & Trough per Program</div>', unsafe_allow_html=True)
        progs_to_analyze = sorted(filtered_progs)
        n_pcols = min(len(progs_to_analyze), 3)
        if n_pcols > 0:
            peak_cols_list = st.columns(n_pcols)
            for pi, prog in enumerate(progs_to_analyze):
                prog_trend = trend[trend['Kategori']==prog].sort_values('Tahun')
                if len(prog_trend) < 2:
                    continue
                peak_yr = int(prog_trend.loc[prog_trend['Kasus'].idxmax(), 'Tahun'])
                trough_yr = int(prog_trend.loc[prog_trend['Kasus'].idxmin(), 'Tahun'])
                peak_val = int(prog_trend['Kasus'].max())
                trough_val = int(prog_trend['Kasus'].min())
                _raw_pk = EKON_CONTEXT.get(peak_yr, ("—", "Tidak ada konteks."))
                peak_ctx = _raw_pk if isinstance(_raw_pk, tuple) and len(_raw_pk)==2 else ("📊", str(_raw_pk))
                _raw_tr = EKON_CONTEXT.get(trough_yr, ("—", "Tidak ada konteks."))
                trough_ctx = _raw_tr if isinstance(_raw_tr, tuple) and len(_raw_tr)==2 else ("📊", str(_raw_tr))
                range_pct = abs((peak_val - trough_val) / (trough_val + 1e-9) * 100)
                with peak_cols_list[pi % n_pcols]:
                    st.markdown(
                        f'<div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;'
                        f'padding:16px 18px;margin-bottom:10px;box-shadow:0 1px 4px rgba(0,0,0,.05);">'
                        f'<div style="font-size:.68rem;font-weight:700;color:#64748b;text-transform:uppercase;'
                        f'letter-spacing:1px;margin-bottom:10px;">📊 {prog}</div>'
                        f'<div style="background:#fef9c3;border-left:3px solid #eab308;border-radius:6px;'
                        f'padding:10px 12px;margin-bottom:8px;">'
                        f'<div style="font-size:.7rem;font-weight:700;color:#854d0e;margin-bottom:3px;">'
                        f'🔺 PEAK — {peak_yr} &nbsp;{peak_ctx[0]}</div>'
                        f'<div style="font-size:.95rem;font-weight:800;color:#713f12">{peak_val:,} kasus</div>'
                        f'<div style="font-size:.75rem;color:#92400e;margin-top:4px;line-height:1.6">{peak_ctx[1]}</div>'
                        f'</div>'
                        f'<div style="background:#eff6ff;border-left:3px solid #3b82f6;border-radius:6px;'
                        f'padding:10px 12px;">'
                        f'<div style="font-size:.7rem;font-weight:700;color:#1e3a8a;margin-bottom:3px;">'
                        f'🔻 TROUGH — {trough_yr} &nbsp;{trough_ctx[0]}</div>'
                        f'<div style="font-size:.95rem;font-weight:800;color:#1e40af">{trough_val:,} kasus</div>'
                        f'<div style="font-size:.75rem;color:#1d4ed8;margin-top:4px;line-height:1.6">{trough_ctx[1]}</div>'
                        f'</div>'
                        f'<div style="margin-top:8px;font-size:.75rem;color:#64748b;">'
                        f'Range peak↔trough: <b style="color:#0f172a">{range_pct:.0f}%</b></div>'
                        f'</div>',
                        unsafe_allow_html=True)

        with st.expander("📰 Referensi Konteks Ekonomi Indonesia per Tahun"):
            if not EKON_CONTEXT:
                st.warning("Konteks ekonomi belum tersedia. Pastikan ANTHROPIC_API_KEY sudah diset di Secrets, lalu refresh halaman.")
            else:
                for yr in sorted(EKON_CONTEXT.keys()):
                    if yr in all_yrs_data:
                        ctx_val = EKON_CONTEXT[yr]
                        if isinstance(ctx_val, tuple) and len(ctx_val) == 2:
                            icon, desc = ctx_val
                        else:
                            icon, desc = "📊", str(ctx_val)
                        st.markdown(
                            f'<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;'
                            f'padding:10px 14px;margin:5px 0;">'
                            f'<span style="font-weight:700;color:#0f172a">{yr} — {icon}</span><br>'
                            f'<span style="font-size:.83rem;color:#475569;line-height:1.7">{desc}</span>'
                            f'</div>',
                            unsafe_allow_html=True)

        t3l, t3r = st.columns(2)
        with t3l:
            st.markdown('<div class="sec">Komposisi Stacked per Tahun</div>', unsafe_allow_html=True)
            fig4 = px.bar(trend, x='Tahun', y='Kasus', color='Kategori',
                          barmode='stack', color_discrete_sequence=COLORS)
            fig4.update_traces(marker_line_width=0)
            styled_chart(fig4, height=360)
            fig4.update_layout(xaxis=dict(dtick=1))
            st.plotly_chart(fig4, width='stretch')

        with t3r:
            st.markdown('<div class="sec">Heatmap Kasus (Program × Tahun)</div>',
                        unsafe_allow_html=True)
            st.markdown('<div style="font-size:.77rem;color:#64748b;margin-bottom:6px;">Warna lebih terang/biru = klaim lebih tinggi. Berguna untuk melihat pola lintas program dan tahun sekaligus.</div>', unsafe_allow_html=True)
            hm_p = (df_plot.groupby(['Kategori', 'Tahun'])['Kasus'].sum()
                    .reset_index()
                    .pivot(index='Kategori', columns='Tahun', values='Kasus')
                    .fillna(0))
            fig5 = px.imshow(hm_p, color_continuous_scale='Blues',
                             aspect='auto', text_auto=',')
            fig5.update_layout(**DARK, height=360, margin=dict(t=10,b=10,l=10,r=10))
            fig5.update_traces(textfont_size=11)
            st.plotly_chart(fig5, width='stretch')

        st.markdown('<div class="sec">Year-over-Year Growth & CAGR per Program</div>',
                    unsafe_allow_html=True)
        st.markdown('<div style="font-size:.77rem;color:#64748b;margin-bottom:6px;">Pertumbuhan kasus vs tahun sebelumnya. Hijau = naik, merah = turun. Garis putus = baseline 0%.</div>', unsafe_allow_html=True)
        yoy = []
        for cat in active_progs:
            cd = df_plot[df_plot['Kategori'] == cat].sort_values('Tahun')
            for i in range(1, len(cd)):
                prev = cd.iloc[i-1]['Kasus']
                curr = cd.iloc[i]['Kasus']
                yoy.append({'Kategori': cat,
                    'Tahun': int(cd.iloc[i]['Tahun']),
                    'Growth (%)': round((curr/(prev+1e-9)-1)*100, 2)})
        if yoy:
            ydf = pd.DataFrame(yoy)
            fig_y = px.bar(ydf, x='Tahun', y='Growth (%)', color='Kategori',
                           barmode='group', color_discrete_sequence=COLORS,
                           text='Growth (%)')
            fig_y.update_traces(texttemplate='%{text:.1f}%', textposition='outside',
                                textfont_size=9, marker_line_width=0)
            fig_y.add_hline(y=0, line_color='#334155', line_width=1.5)
            styled_chart(fig_y, height=360)
            fig_y.update_layout(xaxis=dict(dtick=1))
            st.plotly_chart(fig_y, width='stretch')

        if len(years) >= 3:
            st.markdown('<div class="sec">Distribusi & Variabilitas Kasus per Program</div>',
                        unsafe_allow_html=True)
            fig_box = go.Figure()
            for i, cat in enumerate(sorted(active_progs)):
                cd = df_plot[df_plot['Kategori']==cat]['Kasus'].values
                col_c = COLORS[i % len(COLORS)]
                fig_box.add_trace(go.Box(
                    y=cd, name=cat,
                    marker_color=col_c,
                    line_color=col_c,
                    fillcolor=hex_to_rgba(col_c, 0.15),
                    boxmean='sd',
                    hovertemplate=f"<b>{cat}</b><br>%{{y:,}}<extra></extra>"
                ))
            styled_chart(fig_box, height=340, legend_bottom=False)
            fig_box.update_layout(showlegend=False,
                yaxis_title='Kasus', margin=dict(t=20,b=20,l=60,r=20))
            st.plotly_chart(fig_box, width='stretch')

    if has_nom:
        st.markdown('<div class="sec">Analisis Nominal (Rp)</div>', unsafe_allow_html=True)
        nc1, nc2 = st.columns(2)
        with nc1:
            np_d = df_plot.groupby('Kategori')['Nominal'].sum().reset_index()
            np_d['Nominal_B'] = np_d['Nominal']/1e9
            total_nom = np_d['Nominal_B'].sum()
            fp = px.pie(np_d, names='Kategori', values='Nominal', hole=0.5,
                        color_discrete_sequence=COLORS)
            fp.update_traces(textinfo='label+percent', textposition='outside', textfont_size=11)
            fp.update_layout(**DARK, showlegend=False, height=360,
                             margin=dict(t=10, b=10, l=10, r=10))
            fp.add_annotation(text=f"<b>Rp{total_nom:,.1f}B</b><br><span style='font-size:9px'>Total</span>",
                showarrow=False, font=dict(size=12, color='#e2e8f0'), align='center')
            st.plotly_chart(fp, width='stretch')
        with nc2:
            if not single_yr:
                nt = df_plot.groupby(['Tahun', 'Kategori'])['Nominal'].sum().reset_index()
                nt['Nominal_B'] = nt['Nominal']/1e9
                fn = go.Figure()
                for i, cat in enumerate(sorted(df_plot['Kategori'].unique())):
                    cd = nt[nt['Kategori']==cat].sort_values('Tahun')
                    col_c = COLORS[i % len(COLORS)]
                    fn.add_trace(go.Scatter(
                        x=cd['Tahun'], y=cd['Nominal_B'], name=cat,
                        mode='lines+markers', stackgroup='one',
                        line=dict(color=col_c, width=1.5),
                        fillcolor=hex_to_rgba(col_c, 0.3),
                        hovertemplate=f"<b>{cat}</b><br>Rp%{{y:,.1f}}B<extra></extra>"
                    ))
                styled_chart(fn, height=360)
                fn.update_layout(xaxis=dict(dtick=1), yaxis_title='Rp Miliar')
                st.plotly_chart(fn, width='stretch')
            else:
                nb = df_plot.groupby('Kategori')['Nominal'].sum().reset_index()
                nb['Nominal_B'] = nb['Nominal']/1e9
                fn = go.Figure()
                for i, (_, row) in enumerate(nb.sort_values('Nominal_B',ascending=True).iterrows()):
                    fn.add_trace(go.Bar(
                        x=[row['Nominal_B']], y=[row['Kategori']], orientation='h',
                        name=row['Kategori'], showlegend=False,
                        marker_color=COLORS[i%len(COLORS)], marker_line_width=0,
                        text=f"Rp{row['Nominal_B']:,.1f}B", textposition='outside',
                        hovertemplate=f"<b>{row['Kategori']}</b><br>Rp{row['Nominal_B']:,.1f}B<extra></extra>"
                    ))
                styled_chart(fn, height=360, legend_bottom=False)
                fn.update_layout(margin=dict(t=10,b=10,l=10,r=100))
                st.plotly_chart(fn, width='stretch')

        if not single_yr and has_nom:
            st.markdown('<div class="sec">Korelasi Kasus vs Nominal per Program</div>',
                        unsafe_allow_html=True)
            corr_data = df_plot.groupby(['Kategori','Tahun']).agg(
                Kasus=('Kasus','sum'), Nominal=('Nominal','sum')).reset_index()
            corr_data['Nominal_B'] = corr_data['Nominal']/1e9
            fig_sc = px.scatter(corr_data, x='Kasus', y='Nominal_B',
                color='Kategori', size='Kasus', text='Tahun',
                color_discrete_sequence=COLORS,
                labels={'Nominal_B':'Nominal (Rp Miliar)','Kasus':'Jumlah Kasus'})
            fig_sc.update_traces(textposition='top center', textfont_size=9)
            styled_chart(fig_sc, height=380, legend_bottom=True)
            st.plotly_chart(fig_sc, width='stretch')

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: ML ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    col_s, _ = st.columns([1, 3])
    with col_s:
        target_ml = st.selectbox("Target Prediksi", targets, key='ml_target')
        run_btn   = st.button("🚀 Jalankan Analisis ML", type="primary", width='stretch')

    ck     = f"{target_ml}_lags{n_lags}_test{test_pct}"
    ml_res = results_cache.get(ck)

    if run_btn:
        with st.spinner(f"Melatih model untuk {len(active_progs)} program..."):
            ml_res, err = run_ml(df, target_ml, n_lags, test_pct / 100)
        if err:
            st.error(f"Error: {err}"); ml_res = None
        else:
            results_cache[ck] = ml_res
            st.session_state.active_results = results_cache
            with st.spinner("Menganalisis model per program..."):
                st.session_state[f'per_prog_{target_ml}'] = ml_res
            data_hash = hashlib.md5(df.to_csv().encode()).hexdigest()[:8]
            eid   = f"{data_hash}_{target_ml}_L{n_lags}_T{test_pct}"
            label = (f"📁 {datetime.now().strftime('%d/%m %H:%M')} | "
                     f"{target_ml} | {len(years)}yr | {len(active_progs)} prog")
            extra_snapshot = {
                k: st.session_state[k]
                for k in ['raw_monthly',
                          'forecast_Kasus', 'forecast_Nominal',
                          'forecast_monthly_Kasus', 'forecast_monthly_Nominal',
                          'forecast_annual_Kasus', 'forecast_annual_Nominal',
                          'last_forecast', 'last_forecast_monthly']
                if k in st.session_state and st.session_state[k] is not None
            }
            add_to_history(label, eid, df.copy(), dict(results_cache), extra_snapshot)
            st.session_state.active_entry_id = eid

    if ml_res:
        bpp      = ml_res.get('best_per_program', pd.DataFrame())
        per_prog = ml_res.get('per_prog', {})
        rdf      = ml_res.get('results_df', pd.DataFrame())
        n_yrs    = len(sorted(df['Tahun'].unique()))

        avg_mape = float(bpp['MAPE (%)'].mean()) if not bpp.empty and 'MAPE (%)' in bpp.columns else 0.0
        mape_grade = ("🟢 Sangat Akurat" if avg_mape < 10 else
                      "🔵 Akurat" if avg_mape < 20 else
                      "🟡 Cukup" if avg_mape < 50 else "🔴 Perlu Perbaikan")
        data_note = f"⚠️ {n_yrs} tahun data — R² tidak bermakna, gunakan MAPE" if n_yrs < 8 else ""
        mode_note = '&nbsp;|&nbsp; ⚠️ Mode 1 Tahun' if single_yr else ''
        st.markdown(
            f'<div class="badge">'
            f'🎯 <b>Per-Program Best Model</b> — metode terbaik per program'
            f'&nbsp;|&nbsp; Avg MAPE = <b>{avg_mape:.2f}%</b> ({mape_grade})'
            f'&nbsp;|&nbsp; <b>{len(active_progs)} program</b>'
            f'{"&nbsp;|&nbsp; " + data_note if data_note else ""}'
            f'{mode_note}</div>',
            unsafe_allow_html=True)

        mtab1, mtab2, mtab3, mtab4 = st.tabs([
            "📊 Perbandingan Model", "🎯 Model per Program",
            "📝 Conclusion & Metrics", "🔮 Prophet + Kalender"
        ])

        with mtab1:
            st.markdown('<div class="sec">Model Terbaik per Program (Ringkasan)</div>',
                        unsafe_allow_html=True)
            if not bpp.empty:
                def badge_r2(v):
                    return "🟢" if v>0.8 else "🔵" if v>0.6 else "🟡" if v>0.3 else "🔴"
                def badge_mape(v):
                    return "🟢" if v<10 else "🔵" if v<20 else "🟡" if v<50 else "🔴"
                bpp_disp = bpp.copy()
                bpp_disp['Kualitas R²']   = bpp_disp['R2'].apply(badge_r2)
                bpp_disp['Kualitas MAPE'] = bpp_disp['MAPE (%)'].apply(badge_mape)
                st.dataframe(
                    bpp_disp.style
                       .highlight_max(subset=['R2'], color='#14532d')
                       .highlight_min(subset=['MAPE (%)'], color='#14532d')
                       .format({'R2':'{:.4f}','MAPE (%)':'{:.2f}','MAE':'{:,.0f}','RMSE':'{:,.0f}'}),
                    width='stretch', height=260)

                ml_ta, ml_tb = st.columns(2)
                with ml_ta:
                    if not bpp.empty and 'MAPE (%)' in bpp.columns:
                        bpp_sorted = bpp.sort_values('MAPE (%)')
                        fig_bpp_mp = go.Figure()
                        for i, (_, row) in enumerate(bpp_sorted.iterrows()):
                            mape_v = row['MAPE (%)']
                            col_c  = ('#34d399' if mape_v < 10 else '#60a5fa' if mape_v < 20
                                      else '#fbbf24' if mape_v < 50 else '#f87171')
                            fig_bpp_mp.add_trace(go.Bar(
                                x=[mape_v], y=[row['Program']], orientation='h',
                                name=row['Program'], showlegend=False,
                                marker_color=col_c, marker_line_width=0,
                                text=f"{mape_v:.1f}% ({row['Model']})",
                                textposition='outside', textfont_size=10,
                                hovertemplate=f"<b>{row['Program']}</b><br>MAPE: {mape_v:.2f}%<br>Model: {row['Model']}<extra></extra>"
                            ))
                        fig_bpp_mp.add_vline(x=20, line_dash='dash',
                            line_color='rgba(251,191,36,0.6)', line_width=1.5,
                            annotation_text='Target <20%', annotation_font_color='#fbbf24',
                            annotation_font_size=10)
                        fig_bpp_mp.update_layout(**DARK, height=360, showlegend=False,
                            title='MAPE per Program (lebih rendah = lebih baik)',
                            xaxis_title='MAPE (%)', margin=dict(t=50,b=20,l=20,r=120))
                        st.plotly_chart(fig_bpp_mp, width='stretch')

                with ml_tb:
                    if not bpp.empty:
                        type_counts = bpp['Model'].value_counts().reset_index()
                        type_counts.columns = ['Model', 'Count']
                        fig_mt = px.pie(type_counts, names='Model', values='Count',
                            hole=0.5, color_discrete_sequence=COLORS,
                            title='Distribusi Model Terbaik')
                        fig_mt.update_traces(textinfo='label+percent', textposition='outside',
                            textfont_size=10)
                        fig_mt.update_layout(**DARK, height=360, showlegend=False,
                            margin=dict(t=50,b=20,l=20,r=20))
                        st.plotly_chart(fig_mt, width='stretch')
            else:
                st.info("Jalankan Analisis ML untuk melihat hasil.")

            if not bpp.empty and 'R2' in bpp.columns:
                mc1, mc2 = st.columns(2)
                bpp_plot = bpp.dropna(subset=['R2','MAPE (%)'])
                with mc1:
                    if not bpp_plot.empty:
                        fig_r2 = px.bar(bpp_plot, x='Program', y='R2', color='Model',
                                        color_discrete_sequence=COLORS,
                                        title='R² per Program (LOO-CV)')
                        fig_r2.add_hline(y=0.8, line_dash='dash', line_color='#34d399',
                                         annotation_text='Target 0.8')
                        fig_r2.update_layout(**DARK, height=360, margin=dict(b=60, t=40))
                        st.plotly_chart(fig_r2, width='stretch')
                with mc2:
                    if not bpp_plot.empty:
                        fig_mp = px.bar(bpp_plot, x='Program', y='MAPE (%)', color='Model',
                                        color_discrete_sequence=COLORS,
                                        title='MAPE % per Program (lebih rendah = lebih baik)')
                        fig_mp.add_hline(y=20, line_dash='dash', line_color='#fbbf24',
                                         annotation_text='Threshold 20%')
                        fig_mp.update_layout(**DARK, height=360, margin=dict(b=60, t=40))
                        st.plotly_chart(fig_mp, width='stretch')

            if not single_yr and per_prog:
                st.markdown('<div class="sec">Trend Historis per Program</div>',
                            unsafe_allow_html=True)
                fig_av = go.Figure()
                for i, (cat, info) in enumerate(per_prog.items()):
                    hist = info.get('history', [])
                    if len(hist) < 2:
                        continue
                    fig_av.add_trace(go.Scatter(
                        y=hist, name=cat, mode='lines+markers',
                        line=dict(color=COLORS[i % len(COLORS)], width=2),
                        marker=dict(size=6)
                    ))
                fig_av.update_layout(**DARK, height=360, hovermode='x unified',
                                     legend=dict(orientation='h', y=-0.25),
                                     margin=dict(t=20, b=100),
                                     yaxis_title=target_ml, xaxis_title='Index Tahun')
                st.plotly_chart(fig_av, width='stretch')

                tree_models = ('Random Forest','Gradient Boosting','Decision Tree',
                               'Extra Trees','XGBoost','LightGBM')
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
                            while len(fnames) < nf:
                                fnames.append(f'feat_{len(fnames)}')
                            fi_df = pd.DataFrame({'Feature': fnames, 'Importance': fi_vals})\
                                      .sort_values('Importance', ascending=False).head(8)
                            fig_fi = px.bar(fi_df, x='Importance', y='Feature',
                                            orientation='h', color='Importance',
                                            color_continuous_scale='Viridis',
                                            title=f'Feature Importance — {cat} ({mdl_name})')
                            fig_fi.update_layout(**DARK, height=300,
                                                 coloraxis_showscale=False,
                                                 margin=dict(l=100,t=40,b=20))
                            st.plotly_chart(fig_fi, width='stretch')
                        except:
                            pass

        with mtab2:
            det = ml_res.get('detail', pd.DataFrame())
            if bpp.empty:
                st.info("Klik **Jalankan Analisis ML** untuk melihat model terbaik per program.")
            else:
                st.markdown('<div class="sec">Model Terbaik per Program</div>',
                            unsafe_allow_html=True)
                st.dataframe(
                    bpp.style
                       .highlight_max(subset=['R2'], color='#14532d')
                       .highlight_min(subset=['MAPE (%)'], color='#14532d')
                       .format({'R2': '{:.4f}', 'MAPE (%)': '{:.2f}',
                                'MAE': '{:,.0f}', 'RMSE': '{:,.0f}'}),
                    width='stretch', height=260)

                st.markdown('<div class="sec">Heatmap R² — Semua Model × Semua Program</div>',
                            unsafe_allow_html=True)
                heat = det.pivot_table(index='Model', columns='Program',
                                       values='R2', aggfunc='mean').fillna(0)
                fig_heat = px.imshow(heat, color_continuous_scale='Blues',
                                     aspect='auto', text_auto='.3f',
                                     title='R² per Model per Program (lebih biru = lebih baik)')
                fig_heat.update_layout(**DARK, height=400, margin=dict(t=50, b=20))
                st.plotly_chart(fig_heat, width='stretch')

                st.markdown('<div class="sec">R² Model Terbaik per Program</div>',
                            unsafe_allow_html=True)
                fig_bpp = px.bar(bpp, x='Program', y='R2', color='Model',
                                 text='Model', color_discrete_sequence=COLORS,
                                 title='Model Terbaik & R² per Program')
                fig_bpp.add_hline(y=0.8, line_dash='dash', line_color='#34d399',
                                  annotation_text='Target R²=0.8')
                fig_bpp.update_traces(textposition='outside')
                fig_bpp.update_layout(**DARK, height=400, margin=dict(t=50, b=40))
                st.plotly_chart(fig_bpp, width='stretch')

                st.markdown('<div class="sec">MAPE % Model Terbaik per Program</div>',
                            unsafe_allow_html=True)
                fig_mape = px.bar(bpp, x='Program', y='MAPE (%)', color='Model',
                                  color_discrete_sequence=COLORS,
                                  title='MAPE % per Program (lebih rendah = lebih baik)')
                fig_mape.add_hline(y=20, line_dash='dash', line_color='#fbbf24',
                                   annotation_text='Threshold 20%')
                fig_mape.update_layout(**DARK, height=380, margin=dict(t=50, b=40))
                st.plotly_chart(fig_mape, width='stretch')

                with st.expander("📋 Tabel Detail Semua Model × Semua Program"):
                    st.dataframe(
                        det.sort_values(['Program','R2'], ascending=[True,False])
                           .style.format({'R2': '{:.4f}', 'MAPE (%)': '{:.2f}',
                                          'MAE': '{:,.0f}', 'RMSE': '{:,.0f}'}),
                        width='stretch', height=400)

        with mtab3:
            conclusions = build_conclusion(ml_res, ml_res, df, target_ml, n_future)
            if conclusions:
                st.markdown('<div class="sec">Kesimpulan & Rekomendasi Otomatis</div>',
                            unsafe_allow_html=True)

                if not bpp.empty and 'MAPE (%)' in bpp.columns:
                    best_prog  = bpp.loc[bpp['MAPE (%)'].idxmin(), 'Program']
                    best_mape  = bpp['MAPE (%)'].min()
                    worst_prog = bpp.loc[bpp['MAPE (%)'].idxmax(), 'Program']
                    worst_mape = bpp['MAPE (%)'].max()
                    avg_m = bpp['MAPE (%)'].mean()
                    overall_grade = ("🟢 Sangat Baik" if avg_m < 10 else
                                     "🔵 Baik" if avg_m < 20 else
                                     "🟡 Cukup" if avg_m < 50 else "🔴 Perlu Data Lebih Banyak")
                    data_note = (f"Catatan: {n_yrs} tahun data → metode statistik digunakan (Holt/SES/WMA). "
                                 if n_yrs < 8 else f"Data {n_yrs} tahun → ML tersedia. ")
                    st.markdown(f"""
                    <div class="success-box">
                    🔍 <b>Auto-Insight:</b> Kualitas prediksi keseluruhan: <b>{overall_grade}</b> (Avg MAPE {avg_m:.1f}%). 
                    Terbaik: <b>{best_prog}</b> ({best_mape:.1f}%) · Perlu perhatian: <b>{worst_prog}</b> ({worst_mape:.1f}%). 
                    {data_note}
                    </div>""", unsafe_allow_html=True)

                for icon, title, text in conclusions:
                    st.markdown(f"""
                    <div style="background:linear-gradient(135deg,#0a1628,#0d1a0d);
                                border:1px solid #1e3a5f;border-radius:14px;
                                padding:18px 22px;margin:10px 0;
                                border-left:3px solid #3b82f6;">
                        <div style="font-size:.65rem;font-weight:700;color:#334155;
                                    text-transform:uppercase;letter-spacing:2px;
                                    margin-bottom:8px;">{icon} {title}</div>
                        <div style="color:#e2e8f0;font-size:.9rem;line-height:1.75;">{text}</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown('<div class="sec">Radar Chart — Profil Kualitas per Program</div>',
                            unsafe_allow_html=True)
                if not bpp.empty and 'MAPE (%)' in bpp.columns:
                    rdf_r = bpp.dropna(subset=['MAPE (%)','MAE','RMSE']).copy()
                    rdf_r['MAPE_n']  = (1 - (rdf_r['MAPE (%)'] / 100).clip(0, 1))
                    rdf_r['MAE_n']   = 1 - (rdf_r['MAE'] / (rdf_r['MAE'].max() + 1e-9))
                    rdf_r['RMSE_n']  = 1 - (rdf_r['RMSE'] / (rdf_r['RMSE'].max() + 1e-9))
                    mape_med = rdf_r['MAPE (%)'].median()
                    rdf_r['STAB_n'] = 1 - np.abs(rdf_r['MAPE (%)'] - mape_med) / (mape_med + 1e-9)
                    rdf_r['STAB_n'] = rdf_r['STAB_n'].clip(0, 1)

                    cats_radar = ['Akurasi\n(1−MAPE)', 'Presisi\n(1−MAE)',
                                  'Konsistensi\n(1−RMSE)', 'Stabilitas']
                    fig_radar = go.Figure()
                    for i, row in rdf_r.iterrows():
                        vals  = [row['MAPE_n'], row['MAE_n'], row['RMSE_n'], row['STAB_n']]
                        vals += [vals[0]]
                        col_c = COLORS[i % len(COLORS)]
                        label = f"{row['Program']} ({row['Model']})"
                        fig_radar.add_trace(go.Scatterpolar(
                            r=vals, theta=cats_radar+[cats_radar[0]],
                            fill='toself', name=label, opacity=0.65,
                            line=dict(color=col_c, width=2),
                            fillcolor=hex_to_rgba(col_c, 0.15)
                        ))
                    fig_radar.update_layout(
                        **DARK, height=500,
                        polar=dict(
                            radialaxis=dict(visible=True, range=[0,1],
                                gridcolor='#1e2d45', tickfont=dict(color='#475569', size=9),
                                tickvals=[0.2,0.4,0.6,0.8,1.0]),
                            angularaxis=dict(gridcolor='#1e2d45', tickfont=dict(size=11)),
                            bgcolor='#05090f',
                        ),
                        legend=dict(orientation='h', y=-0.15, font=dict(size=11)),
                        margin=dict(t=30, b=80)
                    )
                    st.plotly_chart(fig_radar, width='stretch')

                st.markdown('<div class="sec">Scorecard per Program</div>',
                            unsafe_allow_html=True)

                if n_yrs < 8:
                    st.info(f"ℹ️ **{n_yrs} tahun data** — gunakan **MAPE** sebagai acuan utama. MAPE < 20% = layak pakai.")

                def grade_mape(v):
                    if v is None or np.isnan(v): return "⚪ N/A"
                    return "🟢 Sangat Akurat (<10%)" if v<10 else "🔵 Akurat (10-20%)" if v<20 else "🟡 Cukup (20-50%)" if v<50 else "🔴 Tidak Akurat (>50%)"

                sc_df = bpp.copy() if not bpp.empty else pd.DataFrame()
                if not sc_df.empty:
                    sc_df['Grade MAPE'] = sc_df['MAPE (%)'].apply(grade_mape)
                    cols_show = ['Program','Model','MAPE (%)','MAE','RMSE','Grade MAPE']
                    if n_yrs >= 8:
                        cols_show = ['Program','Model','R2','MAPE (%)','MAE','RMSE','Grade MAPE']
                    fmt = {'MAPE (%)': '{:.2f}', 'MAE': '{:,.0f}', 'RMSE': '{:,.0f}'}
                    if 'R2' in cols_show:
                        fmt['R2'] = '{:.4f}'
                    st.dataframe(
                        sc_df[cols_show].style
                            .highlight_min(subset=['MAPE (%)'], color='#14532d')
                            .format(fmt),
                        width='stretch', height=280)

        # ── Sub-tab 4: Prophet + Indonesian Calendar ──────────────────────
        with mtab4:
            df_raw_m_p = st.session_state.get('raw_monthly', None)
            if not PROPHET_OK:
                st.warning("""
                **Prophet belum terinstall.** Tambahkan ke `requirements.txt`:
                ```
                prophet
                ```
                """)
            elif df_raw_m_p is None or len(df_raw_m_p) == 0:
                st.warning("Upload dataset dengan data bulanan terlebih dahulu untuk menggunakan Prophet.")
            else:
                n_holidays  = len(INDONESIAN_HOLIDAYS)
                n_htypes    = INDONESIAN_HOLIDAYS['holiday'].nunique() if n_holidays > 0 else 0

                # Status GCAL
                if not GCAL_KEY:
                    gcal_status = "⚠️ <b>GCAL_KEY belum diset</b> di Streamlit Secrets. Tambahkan key Google Calendar API agar hari libur Indonesia bisa dimuat. Prophet tetap jalan tanpa holiday effect."
                elif n_holidays == 0:
                    gcal_status = "⚠️ <b>Google Calendar API tidak mengembalikan data.</b> Periksa validitas GCAL_KEY dan pastikan quota API tidak habis. Prophet tetap jalan tanpa holiday effect."
                else:
                    gcal_status = f"✅ <b>{n_holidays} hari libur</b>, <b>{n_htypes} jenis</b> berhasil dimuat dari Google Calendar API Indonesia."

                st.markdown(f"""<div class="info-box">
                🔮 <b>Prophet</b> dipilih karena lebih cocok dari SARIMA untuk data BPJS:<br>
                • Menangani <b>efek hari libur secara eksplisit</b> — semua nama dan tanggal dibaca langsung dari Google Calendar<br>
                • Tidak perlu data stasioner — cocok untuk klaim yang terus tumbuh<br>
                • Trend + Seasonality + Holiday dipisah secara interpretable<br><br>
                📅 <b>Sumber kalender:</b> Google Calendar API Indonesia (2019–2028, auto-refresh 24 jam).<br>
                {gcal_status}
                </div>""", unsafe_allow_html=True)

                pc1, pc2 = st.columns(2)
                with pc1:
                    target_prophet = st.selectbox("Target", targets, key='prophet_target')
                with pc2:
                    n_months_prophet = st.slider("Prediksi (bulan)", 6, 36, 12, 6)

                use_holidays = st.checkbox(
                    f"Gunakan kalender hari libur Indonesia dari Google Calendar ({n_holidays} hari libur, {n_htypes} jenis)",
                    value=(n_holidays > 0))

                if st.button("🔮 Jalankan Prophet (Semua Program)", type="primary", width='stretch'):
                    all_p_results = {}
                    prog_errors   = {}
                    with st.spinner(f"Melatih Prophet untuk semua program — {target_prophet}..."):
                        for cp in active_progs:
                            pr, pe = run_prophet(df_raw_m_p, target_prophet, cp,
                                                 n_months_prophet, use_holidays)
                            if pe:
                                prog_errors[cp] = pe
                            else:
                                all_p_results[cp] = pr
                    if prog_errors:
                        st.warning("Beberapa program gagal: " + ", ".join(
                            f"{k}: {v}" for k,v in prog_errors.items()))
                    if all_p_results:
                        st.session_state['prophet_all_results'] = all_p_results
                        st.session_state['prophet_meta'] = {
                            'target': target_prophet, 'use_holidays': use_holidays,
                            'n_months': n_months_prophet
                        }

                all_p_results = st.session_state.get('prophet_all_results', {})
                p_meta        = st.session_state.get('prophet_meta', {})

                if all_p_results and p_meta.get('target') == target_prophet:
                    tgt_label = target_prophet
                    st.markdown(
                        f'<div class="sec">Forecast Prophet — Semua Program ({tgt_label})</div>',
                        unsafe_allow_html=True)

                    def hex_rgba(hex_c, alpha):
                        hex_c = hex_c.lstrip('#')
                        r,g,b = int(hex_c[0:2],16), int(hex_c[2:4],16), int(hex_c[4:6],16)
                        return f'rgba({r},{g},{b},{alpha})'

                    all_y_vals = []
                    for cp, pr in all_p_results.items():
                        all_y_vals += list(pr['history']['y'].dropna())
                    global_ymin = max(0, min(all_y_vals) * 0.85) if all_y_vals else 0

                    fig_p_all    = go.Figure()
                    last_hist_ds = None

                    for i, (cp, pr) in enumerate(all_p_results.items()):
                        fc_df   = pr['forecast'].copy()
                        hist_df = pr['history'].copy()
                        col_c   = COLORS[i % len(COLORS)]
                        cutoff  = hist_df['ds'].max()
                        last_hist_ds = cutoff

                        last_actual   = float(hist_df['y'].iloc[-1])
                        floor_val     = max(0.0, last_actual * 0.0)
                        fc_future     = fc_df[fc_df['ds'] > cutoff].copy()
                        fc_future['yhat']       = fc_future['yhat'].clip(lower=floor_val)
                        fc_future['yhat_lower'] = fc_future['yhat_lower'].clip(lower=0)
                        fc_future['yhat_upper'] = fc_future['yhat_upper'].clip(lower=0)

                        fig_p_all.add_trace(go.Scatter(
                            x=hist_df['ds'], y=hist_df['y'],
                            name=f'{cp} Aktual',
                            mode='lines+markers',
                            legendgroup=cp,
                            line=dict(color=col_c, width=2.5),
                            marker=dict(size=5),
                            fill='tozeroy',
                            fillcolor=hex_rgba(col_c, 0.04),
                            hovertemplate=f'<b>{cp}</b><br>%{{x|%b %Y}}<br>{tgt_label}: %{{y:,.0f}}<extra></extra>'))

                        fig_p_all.add_trace(go.Scatter(
                            x=list(fc_future['ds']) + list(fc_future['ds'][::-1]),
                            y=list(fc_future['yhat_upper']) + list(fc_future['yhat_lower'][::-1]),
                            fill='toself',
                            fillcolor=hex_rgba(col_c, 0.08),
                            line=dict(color='rgba(0,0,0,0)'),
                            legendgroup=cp, showlegend=False, hoverinfo='skip'))

                        fig_p_all.add_trace(go.Scatter(
                            x=fc_future['ds'], y=fc_future['yhat'],
                            name=f'{cp} Prediksi',
                            mode='lines+markers',
                            legendgroup=cp,
                            line=dict(color=col_c, width=2, dash='dash'),
                            marker=dict(size=6, symbol='diamond'),
                            hovertemplate=f'<b>{cp} (Prediksi)</b><br>%{{x|%b %Y}}<br>{tgt_label}: %{{y:,.0f}}<extra></extra>'))

                    if last_hist_ds is not None:
                        fig_p_all.add_vline(
                            x=last_hist_ds.timestamp()*1000,
                            line_dash='dot', line_color='rgba(148,163,184,0.4)',
                            line_width=1.5,
                            annotation_text='← Aktual | Prediksi →',
                            annotation_font=dict(size=10, color='#94a3b8'),
                            annotation_position='top')

                    fig_p_all.update_layout(
                        **DARK,
                        title=dict(
                            text=f'Forecast Prophet — {tgt_label} per Program',
                            font=dict(size=14, color='#e2e8f0'), x=0),
                        height=520,
                        hovermode='x unified',
                        legend=dict(orientation='h', y=-0.22, font=dict(size=10),
                                    groupclick='toggleitem'),
                        margin=dict(t=50, b=130),
                        xaxis_title='Periode',
                        yaxis_title=tgt_label,
                        xaxis=dict(showgrid=True, gridcolor='#0f1923'),
                        yaxis=dict(
                            showgrid=True, gridcolor='#0f1923',
                            rangemode='nonnegative',
                            range=[global_ymin, None]))
                    st.plotly_chart(fig_p_all, width='stretch')

                    st.markdown('<div class="sec">Tabel Prediksi per Program</div>',
                                unsafe_allow_html=True)
                    prog_tabs = st.tabs(list(all_p_results.keys()))
                    for tab_i, (cp, pr) in zip(prog_tabs, all_p_results.items()):
                        with tab_i:
                            fc_df   = pr['forecast']
                            hist_df = pr['history']
                            fc_fut  = fc_df[fc_df['ds'] > hist_df['ds'].max()][
                                ['ds','yhat','yhat_lower','yhat_upper']].copy()
                            fc_fut.columns = ['Periode','Prediksi','Batas Bawah','Batas Atas']
                            fc_fut['Periode'] = fc_fut['Periode'].dt.strftime('%Y-%m')
                            # Tambah % ketidakpastian terhadap prediksi (poin 3)
                            fc_fut['% BB'] = fc_fut.apply(
                                lambda r: f"-{((r['Prediksi']-r['Batas Bawah'])/(r['Prediksi']+1e-9)*100):.1f}%"
                                if r['Prediksi']>0 else '—', axis=1)
                            fc_fut['% BA'] = fc_fut.apply(
                                lambda r: f"+{((r['Batas Atas']-r['Prediksi'])/(r['Prediksi']+1e-9)*100):.1f}%"
                                if r['Prediksi']>0 else '—', axis=1)
                            for col in ['Prediksi','Batas Bawah','Batas Atas']:
                                fc_fut[col] = fc_fut[col].apply(lambda x: f"{max(0,x):,.0f}")
                            st.dataframe(
                                fc_fut[['Periode','Prediksi','Batas Bawah','% BB','Batas Atas','% BA']],
                                width='stretch', height=320)
                            st.markdown(
                                '<div class="info-box" style="font-size:.79rem;margin-top:6px">'
                                '📊 <b>% BB / % BA</b> = persentase deviasi interval kepercayaan 80% '
                                'terhadap nilai prediksi. Semakin kecil % → prediksi lebih pasti.</div>',
                                unsafe_allow_html=True)

                    # ══════════════════════════════════════════════════════════
                    # HOLIDAY EFFECTS — ekstraksi BENAR via model.params['holidays']
                    # ══════════════════════════════════════════════════════════
                    st.markdown('<div class="sec">Efek Hari Libur per Program</div>',
                                unsafe_allow_html=True)

                    if n_holidays == 0:
                        st.markdown("""<div class="warn">
                        ⚠️ <b>Tidak ada data hari libur.</b> Pastikan GCAL_KEY sudah diset dengan benar
                        di Streamlit Secrets agar Prophet bisa mempelajari efek hari libur Indonesia.
                        </div>""", unsafe_allow_html=True)
                    else:
                        # ── Ekstraksi efek per holiday per program ───────────────────────
                        # CARA BENAR: baca dari model.params['holidays'] (posterior mean)
                        # bukan dari kolom forecast yang merupakan agregat semua holiday di
                        # tiap baris, bukan nilai per-holiday secara individual.
                        # model.params['holidays'] → DataFrame index=holiday_name, cols=iter
                        # kita ambil mean across iterations sebagai point estimate.

                        def _extract_holiday_effects(pr):
                            """
                            Ekstrak efek per holiday dari Prophet — menggunakan
                            model.train_holiday_names + kolom individual di forecast.

                            CARA KERJA PROPHET:
                            Saat predict(), Prophet membuat 1 kolom per holiday name
                            (setelah disanitize) di forecast DataFrame.
                            Kolom tsb berisi nilai efek NON-ZERO hanya pada baris yang
                            jatuh dalam window [lower_window, upper_window] holiday.
                            Baris di luar window = 0.0 persis.

                            Kita ambil mean(non-zero rows) per kolom = efek tipikal holiday.
                            model.train_holiday_names = list nama asli yang dipakai model.
                            Ini adalah ground truth — bukan dari GCAL/h_col_map.

                            KENAPA SEBELUMNYA SEMUA DIWALI:
                            h_col_map kosong (GCAL_KEY tidak diset) → semua lookup gagal →
                            hanya kolom 'Diwali' kebetulan match karena namanya simple
                            (tidak ada karakter non-word), holiday lain seperti
                            'New Year\'s Day' → 'New_Year_s_Day' tidak ketemu di map kosong.
                            """
                            model   = pr.get('model')
                            hist_df = pr.get('history')
                            fc_df   = pr.get('forecast', pd.DataFrame())

                            if model is None or hist_df is None or fc_df is None or len(fc_df) == 0:
                                return {}

                            avg_y  = float(hist_df['y'].mean()) if len(hist_df) > 0 else 1.0
                            s_mode = getattr(model, 'seasonality_mode', 'additive')

                            # ── Dapatkan daftar holiday yang benar-benar dipakai model ──
                            holiday_names = []
                            try:
                                holiday_names = list(model.train_holiday_names)
                            except Exception:
                                pass
                            if not holiday_names:
                                try:
                                    holiday_names = list(model.holidays['holiday'].unique())
                                except Exception:
                                    pass
                            if not holiday_names:
                                return {}

                            forecast_cols = set(fc_df.columns)
                            effects = {}

                            for orig_name in holiday_names:
                                # Sanitize: sama persis dengan yang dilakukan Prophet internal
                                san = _sanitize_prophet_name(orig_name)

                                # Cari kolom point estimate (bukan _lower/_upper)
                                col = san if (san in forecast_cols
                                              and not san.endswith('_lower')
                                              and not san.endswith('_upper')) else None

                                if col is None:
                                    continue

                                col_vals    = fc_df[col]
                                active_mask = col_vals.abs() > 1e-9

                                # Skip jika tidak ada baris dalam window holiday
                                if active_mask.sum() == 0:
                                    continue

                                # mean efek pada hari-hari yang kena window holiday
                                raw_eff = float(col_vals[active_mask].mean())

                                if s_mode == 'multiplicative':
                                    # multiplicative: efek sudah bentuk rasio * avg
                                    # kolom = nilai yhat_holiday / yhat_baseline - 1
                                    pct = raw_eff * 100.0
                                else:
                                    # additive: efek dalam unit y → konversi ke %
                                    pct = (raw_eff / (avg_y + 1e-9)) * 100.0

                                effects[orig_name] = pct

                            return effects

                        # ── Kumpulkan efek semua program ─────────────────────────────
                        heff_rows = []
                        for cp, pr in all_p_results.items():
                            hist_df = pr.get('history', pd.DataFrame())
                            avg_y_p = float(hist_df['y'].mean()) if len(hist_df) > 0 else 1.0
                            eff_map = _extract_holiday_effects(pr)
                            for hname, pct in eff_map.items():
                                heff_rows.append({
                                    'Program'    : cp,
                                    'Holiday'    : hname,
                                    'Efek_pct'   : pct,
                                    'avg_y'      : avg_y_p,
                                })

                        if not heff_rows:
                            st.info(
                                "Tidak ada efek hari libur yang terdeteksi. "
                                "Kemungkinan data bulanan belum cukup (minimal 12–24 bulan) "
                                "agar Prophet bisa mempelajari pola holiday secara signifikan."
                            )
                        else:
                            heff_df = pd.DataFrame(heff_rows)

                            # ── Kategorisasi semantik (grouping holiday yg sama) ──────
                            # Tujuan: 'Idul Fitri 2021', 'Idul Fitri 2022' → 'Idul Fitri'
                            # sehingga tiap program punya 1 angka per jenis hari libur.
                            def _cat_holiday(name):
                                nl = name.lower()
                                if any(k in nl for k in ['idul fitri','lebaran','eid al-fitr','eid ul-fitr','eid al fitr']): return 'Idul Fitri'
                                if any(k in nl for k in ['idul adha','eid al-adha','eid ul-adha','eid al adha']): return 'Idul Adha'
                                if any(k in nl for k in ['ramad','puasa']): return 'Ramadhan'
                                if any(k in nl for k in ['natal','christmas']): return 'Natal'
                                if any(k in nl for k in ["new year's",'tahun baru masehi','new year masehi']) and not any(x in nl for x in ['islam','imlek','chinese','hijri','lunar']): return 'Tahun Baru'
                                if any(k in nl for k in ['imlek','chinese new year','lunar new year']): return 'Imlek'
                                if any(k in nl for k in ['nyepi','day of silence','hindu new year']): return 'Nyepi'
                                if any(k in nl for k in ["isra","mi'raj",'miraj',"prophet's ascension"]): return "Isra Mi'raj"
                                if any(k in nl for k in ['waisak','vesak','buddha']): return 'Waisak'
                                if any(k in nl for k in ['good friday','wafat','easter','paskah','kenaikan yesus']): return 'Paskah/Wafat'
                                if any(k in nl for k in ['maulid','mawlid',"prophet's birthday"]): return 'Maulid Nabi'
                                if any(k in nl for k in ['muharram','islamic new year','hijri new year','tahun baru islam']): return 'Tahun Baru Islam'
                                if any(k in nl for k in ['buruh','labor day','labour day','may day']): return 'Hari Buruh'
                                if any(k in nl for k in ['pancasila']): return 'Hari Pancasila'
                                if any(k in nl for k in ['kemerdekaan','independence day','hut ri']): return 'HUT RI'
                                if any(k in nl for k in ['cuti bersama','joint holiday','collective leave']): return 'Cuti Bersama'
                                if any(k in nl for k in ['election','pemilu','pilpres']): return 'Pemilu'
                                if any(k in nl for k in ['kenaikan','ascension of jesus','corpus']): return 'Hari Kenaikan'
                                if any(k in nl for k in ['tahun baru','new year']) and not any(x in nl for x in ['islam','imlek','chinese','hijri']): return 'Tahun Baru'
                                # Nama asli jika tidak cocok dengan apapun (max 30 char)
                                return name[:30]

                            heff_df['Kategori'] = heff_df['Holiday'].apply(_cat_holiday)

                            # ── Agregasi: mean efek per Program × Kategori ────────────
                            heff_grp = (heff_df
                                        .groupby(['Program','Kategori'])
                                        .agg(
                                            Efek_pct = ('Efek_pct', 'mean'),
                                            avg_y    = ('avg_y',    'first'),
                                            n_events = ('Efek_pct', 'count'),
                                        )
                                        .reset_index())

                            # ── Filter: top-12 kategori by abs effect ACROSS semua program ─
                            top_cats = (heff_grp
                                        .groupby('Kategori')['Efek_pct']
                                        .apply(lambda x: x.abs().mean())
                                        .nlargest(12)
                                        .index.tolist())
                            heff_grp = heff_grp[heff_grp['Kategori'].isin(top_cats)].copy()

                            programs_list = sorted(heff_grp['Program'].unique())
                            max_abs_pct   = heff_grp['Efek_pct'].abs().max()

                            if len(heff_grp) == 0 or max_abs_pct < 0.01:
                                st.markdown("""<div class="warn">
                                ⚠️ <b>Efek hari libur sangat kecil.</b>
                                Tambah data bulanan (minimal 24 bulan) agar Prophet bisa belajar
                                pola holiday lebih baik.
                                </div>""", unsafe_allow_html=True)
                            else:
                                # ── Info box status ─────────────────────────────────────
                                n_cats_shown = len(top_cats)
                                n_total_hols = heff_df['Kategori'].nunique()
                                st.markdown(
                                    f'<div class="info-box">'
                                    f'📊 Menampilkan <b>{n_cats_shown} jenis hari libur</b> '
                                    f'(dari {n_total_hols} total yang dideteksi). '
                                    f'Efek diekstrak langsung dari parameter posterior Prophet per program — '
                                    f'tiap program belajar sendiri dari data historisnya.'
                                    f'</div>',
                                    unsafe_allow_html=True)

                                # ── Urutkan kategori by abs mean effect descending ───────
                                cat_order = (heff_grp
                                             .groupby('Kategori')['Efek_pct']
                                             .apply(lambda x: x.abs().mean())
                                             .sort_values(ascending=False)
                                             .index.tolist())

                                # ═══════════════════════════════════════════════════════
                                # CHART 1: Grouped Horizontal Bar — semua program, 1 chart
                                # Setiap kelompok = 1 jenis holiday, bar = per program
                                # ═══════════════════════════════════════════════════════
                                st.markdown('<div class="sec">Efek Hari Libur per Program (% dari rata-rata klaim bulanan)</div>', unsafe_allow_html=True)

                                fig_bar_h = go.Figure()
                                for pi, prog in enumerate(programs_list):
                                    pdata_prog = (heff_grp[heff_grp['Program'] == prog]
                                                  .set_index('Kategori')['Efek_pct'])
                                    y_vals  = [float(pdata_prog.get(c, np.nan)) for c in cat_order]
                                    colors_bar = ['#34d399' if (not np.isnan(v) and v >= 0) else '#f87171' for v in y_vals]

                                    fig_bar_h.add_trace(go.Bar(
                                        name=prog,
                                        y=cat_order,
                                        x=y_vals,
                                        orientation='h',
                                        marker_color=COLORS[pi % len(COLORS)],
                                        marker_line_width=0,
                                        text=[f'{v:+.1f}%' if not np.isnan(v) else '' for v in y_vals],
                                        textposition='outside',
                                        textfont=dict(size=9, color='#94a3b8'),
                                        hovertemplate=(
                                            f'<b>{prog}</b><br>'
                                            'Holiday: %{y}<br>'
                                            'Efek: <b>%{x:+.2f}%</b> dari rata-rata klaim'
                                            '<extra></extra>'
                                        ),
                                    ))

                                x_max = max(1.0, max_abs_pct * 1.45)
                                fig_bar_h.add_vline(
                                    x=0,
                                    line_color='rgba(255,255,255,0.25)',
                                    line_width=1.5)
                                fig_bar_h.update_layout(
                                    **DARK,
                                    barmode='group',
                                    height=max(420, len(cat_order) * 44 + 140),
                                    xaxis=dict(
                                        range=[-x_max, x_max],
                                        showgrid=True, gridcolor='#0f1923',
                                        zeroline=False,
                                        ticksuffix='%',
                                        tickfont=dict(size=10, color='#64748b'),
                                        title='Efek (%)',
                                    ),
                                    yaxis=dict(
                                        categoryorder='array',
                                        categoryarray=cat_order[::-1],
                                        tickfont=dict(size=11.5, color='#e2e8f0'),
                                        showgrid=True, gridcolor='#0f1923',
                                    ),
                                    legend=dict(
                                        orientation='h', y=-0.12,
                                        font=dict(size=11)),
                                    margin=dict(t=30, b=80, l=160, r=60),
                                    title=dict(
                                        text='Efek Hari Libur per Program — positif = klaim naik, negatif = klaim turun',
                                        font=dict(size=13, color='#e2e8f0'), x=0),
                                )
                                st.plotly_chart(fig_bar_h, width='stretch')

                                # ═══════════════════════════════════════════════════════
                                # CHART 2: Heatmap Program × Holiday
                                # ═══════════════════════════════════════════════════════
                                st.markdown('<div class="sec">Heatmap Intensitas Efek Holiday</div>', unsafe_allow_html=True)

                                hm_pivot = (heff_grp
                                            .pivot_table(index='Kategori', columns='Program',
                                                         values='Efek_pct', aggfunc='mean')
                                            .reindex(cat_order)
                                            .fillna(0))

                                # Custom diverging colorscale: merah=turun, putih=netral, hijau=naik
                                zmax = float(hm_pivot.abs().max().max())
                                fig_hm = go.Figure(go.Heatmap(
                                    z=hm_pivot.values,
                                    x=list(hm_pivot.columns),
                                    y=list(hm_pivot.index),
                                    colorscale=[
                                        [0.0,  '#7f1d1d'],
                                        [0.25, '#f87171'],
                                        [0.5,  '#0f172a'],
                                        [0.75, '#4ade80'],
                                        [1.0,  '#14532d'],
                                    ],
                                    zmid=0,
                                    zmin=-zmax,
                                    zmax=zmax,
                                    text=[[f'{v:+.1f}%' for v in row] for row in hm_pivot.values],
                                    texttemplate='%{text}',
                                    textfont=dict(size=11, color='white'),
                                    hovertemplate='<b>%{y}</b> × <b>%{x}</b><br>Efek: <b>%{z:+.2f}%</b><extra></extra>',
                                    colorbar=dict(
                                        title='Efek (%)',
                                        ticksuffix='%',
                                        tickfont=dict(color='#94a3b8', size=10),
                                        titlefont=dict(color='#94a3b8'),
                                    ),
                                ))
                                fig_hm.update_layout(
                                    **DARK,
                                    height=max(320, len(cat_order) * 38 + 100),
                                    margin=dict(t=30, b=40, l=160, r=60),
                                    xaxis=dict(tickfont=dict(size=11, color='#93c5fd'),
                                               side='top'),
                                    yaxis=dict(
                                        categoryorder='array',
                                        categoryarray=cat_order[::-1],
                                        tickfont=dict(size=11, color='#e2e8f0')),
                                    title=dict(text='Intensitas Efek: merah=klaim turun, hijau=klaim naik',
                                               font=dict(size=12, color='#94a3b8'), x=0),
                                )
                                st.plotly_chart(fig_hm, width='stretch')

                                # ═══════════════════════════════════════════════════════
                                # SCORECARD CARDS — per program, berbeda tiap program
                                # ═══════════════════════════════════════════════════════
                                st.markdown('<div class="sec">Ringkasan Efek per Program</div>',
                                            unsafe_allow_html=True)

                                card_cols = st.columns(len(programs_list))
                                for ci, prog in enumerate(programs_list):
                                    prog_data = heff_grp[heff_grp['Program'] == prog].copy()
                                    avg_y_p   = float(prog_data['avg_y'].iloc[0]) if len(prog_data) > 0 else 1.0
                                    col_c     = COLORS[ci % len(COLORS)]

                                    # Sort ascending/descending — unik per program
                                    pos_data = prog_data[prog_data['Efek_pct'] > 0.1].sort_values('Efek_pct', ascending=False)
                                    neg_data = prog_data[prog_data['Efek_pct'] < -0.1].sort_values('Efek_pct', ascending=True)
                                    net_prog  = float(prog_data['Efek_pct'].sum())

                                    def _pill_p(v):
                                        return (f'<span style="color:#34d399;font-weight:700">{v:+.1f}%</span>'
                                                if v > 0 else
                                                f'<span style="color:#f87171;font-weight:700">{v:+.1f}%</span>')

                                    def _delta_kasus(v):
                                        dk = abs(v / 100.0 * avg_y_p)
                                        if dk < 1: return ''
                                        return f'<span style="color:#475569;font-size:.7rem"> (~{dk:,.0f} kasus)</span>'

                                    # ── Paling Naik ──
                                    if len(pos_data) > 0:
                                        up_rows = pos_data.head(3)
                                        up_html = ''.join(
                                            f'<div style="display:flex;justify-content:space-between;'
                                            f'align-items:center;margin:5px 0;font-size:.8rem;gap:4px;">'
                                            f'<span><span style="color:#34d399;margin-right:4px">▲</span>'
                                            f'<span style="color:#e2e8f0">{row.Kategori}</span>'
                                            f'{_delta_kasus(row.Efek_pct)}</span>'
                                            f'{_pill_p(row.Efek_pct)}</div>'
                                            for row in up_rows.itertuples()
                                        )
                                    else:
                                        up_html = '<div style="color:#475569;font-size:.8rem;font-style:italic">Tidak ada efek positif</div>'

                                    # ── Paling Turun ──
                                    if len(neg_data) > 0:
                                        dn_rows = neg_data.head(3)
                                        dn_html = ''.join(
                                            f'<div style="display:flex;justify-content:space-between;'
                                            f'align-items:center;margin:5px 0;font-size:.8rem;gap:4px;">'
                                            f'<span><span style="color:#f87171;margin-right:4px">▼</span>'
                                            f'<span style="color:#e2e8f0">{row.Kategori}</span>'
                                            f'{_delta_kasus(row.Efek_pct)}</span>'
                                            f'{_pill_p(row.Efek_pct)}</div>'
                                            for row in dn_rows.itertuples()
                                        )
                                    else:
                                        dn_html = '<div style="color:#475569;font-size:.8rem;font-style:italic">Tidak ada efek negatif</div>'

                                    # ── Badge net ──
                                    if abs(net_prog) < 0.5:
                                        badge = '<span style="background:#1e2d45;color:#94a3b8;padding:2px 8px;border-radius:6px;font-size:.72rem">Netral</span>'
                                    elif net_prog > 0:
                                        badge = f'<span style="background:#052e16;color:#34d399;padding:2px 8px;border-radius:6px;font-size:.72rem">Net +{net_prog:.1f}%</span>'
                                    else:
                                        badge = f'<span style="background:#450a0a;color:#f87171;padding:2px 8px;border-radius:6px;font-size:.72rem">Net {net_prog:.1f}%</span>'

                                    with card_cols[ci]:
                                        st.markdown(f'''
                                        <div style="background:#0a1628;border:1px solid {col_c}40;
                                        border-top:3px solid {col_c};border-radius:12px;padding:16px 18px;">
                                          <div style="display:flex;justify-content:space-between;
                                          align-items:center;margin-bottom:12px;">
                                            <span style="font-size:.75rem;color:#94a3b8;font-weight:700;
                                            text-transform:uppercase;letter-spacing:1.5px;">{prog}</span>
                                            {badge}
                                          </div>
                                          <div style="font-size:.65rem;color:#475569;text-transform:uppercase;
                                          letter-spacing:1px;margin-bottom:6px;">📈 Klaim Naik Saat</div>
                                          {up_html}
                                          <div style="border-top:1px solid #1e2d45;margin:10px 0;"></div>
                                          <div style="font-size:.65rem;color:#475569;text-transform:uppercase;
                                          letter-spacing:1px;margin-bottom:6px;">📉 Klaim Turun Saat</div>
                                          {dn_html}
                                        </div>''', unsafe_allow_html=True)

                                # ═══════════════════════════════════════════════════════
                                # TABEL DETAIL — semua holiday semua program
                                # ── Korelasi efek holiday antar program (poin 1) ──────────────
                                st.markdown('<div class="sec">🔗 Korelasi Efek Holiday Antar Program</div>', unsafe_allow_html=True)
                                st.markdown(
                                    '<div class="info-box">💡 <b>Cara baca matriks korelasi:</b> '
                                    '+1 = kedua program <b>sama-sama naik</b> saat holiday tsb. '
                                    '-1 = <b>berlawanan arah</b> (satu naik, satu turun). '
                                    '0 = tidak ada pola bersama.</div>',
                                    unsafe_allow_html=True)
                                if len(programs_list) >= 2:
                                    corr_pivot = (heff_grp.pivot_table(
                                        index='Kategori', columns='Program',
                                        values='Efek_pct', aggfunc='mean').fillna(0))
                                    if corr_pivot.shape[1] >= 2:
                                        corr_matrix = corr_pivot.corr()
                                        fig_corr = go.Figure(go.Heatmap(
                                            z=corr_matrix.values,
                                            x=list(corr_matrix.columns),
                                            y=list(corr_matrix.index),
                                            colorscale=[[0,'#dc2626'],[0.25,'#fca5a5'],
                                                        [0.5,'#1e293b'],[0.75,'#86efac'],[1,'#16a34a']],
                                            zmid=0, zmin=-1, zmax=1,
                                            text=[[f'{v:.2f}' for v in row] for row in corr_matrix.values],
                                            texttemplate='%{text}',
                                            textfont=dict(size=14, color='white', family='JetBrains Mono'),
                                            hovertemplate='<b>%{y}</b> ↔ <b>%{x}</b><br>r = <b>%{z:.3f}</b><extra></extra>',
                                            colorbar=dict(title='r', tickfont=dict(color='#94a3b8', size=10),
                                                thickness=12, len=0.7),
                                        ))
                                        fig_corr.update_layout(
                                            **DARK, height=max(280, len(programs_list)*80),
                                            margin=dict(t=10, b=30, l=90, r=80),
                                            xaxis=dict(tickfont=dict(size=12, color='#7dd3fc')),
                                            yaxis=dict(tickfont=dict(size=12, color='#7dd3fc')),
                                        )
                                        st.plotly_chart(fig_corr, use_container_width=True)

                                        interp_lines = []
                                        progs_corr = list(corr_matrix.columns)
                                        for ia2, pa in enumerate(progs_corr):
                                            for pb in progs_corr[ia2+1:]:
                                                r = float(corr_matrix.loc[pa, pb])
                                                if abs(r) >= 0.5:
                                                    if r >= 0.7:
                                                        lbl = f'Korelasi kuat positif (r={r:+.2f}) — 📈 keduanya cenderung naik bersamaan saat hari libur'
                                                    elif r >= 0.5:
                                                        lbl = f'Korelasi sedang positif (r={r:+.2f}) — cenderung bergerak searah'
                                                    elif r <= -0.7:
                                                        lbl = f'Korelasi kuat negatif (r={r:+.2f}) — 🔄 berlawanan arah'
                                                    else:
                                                        lbl = f'Korelasi sedang negatif (r={r:+.2f}) — cenderung berlawanan'
                                                    interp_lines.append(f'<b>{pa}</b> ↔ <b>{pb}</b>: {lbl}')
                                        if interp_lines:
                                            st.markdown(
                                                '<div class="insight-note">📖 <b>Interpretasi (|r| ≥ 0.5):</b><br>'
                                                + '<br>'.join(interp_lines) + '</div>',
                                                unsafe_allow_html=True)
                                        else:
                                            st.markdown(
                                                '<div class="insight-note">Tidak ada korelasi signifikan antar program '
                                                '(semua |r| < 0.5) — tiap program merespons hari libur secara berbeda.</div>',
                                                unsafe_allow_html=True)
                                    else:
                                        st.info("Butuh ≥2 program untuk analisis korelasi.")
                                else:
                                    st.info("Aktifkan 2+ program di filter sidebar untuk melihat korelasi.")

                                # ═══════════════════════════════════════════════════════
                                with st.expander("📋 Tabel Detail Efek Semua Holiday × Semua Program"):
                                    detail_tbl = (heff_grp
                                                  .copy()
                                                  .sort_values(['Kategori','Efek_pct'], ascending=[True,False])
                                                  .rename(columns={
                                                      'Kategori':  'Hari Libur',
                                                      'Efek_pct':  'Efek (%)',
                                                      'n_events':  'Jumlah Event',
                                                  }))
                                    detail_tbl['Efek (%)'] = detail_tbl['Efek (%)'].round(2)
                                    detail_tbl['Arah'] = detail_tbl['Efek (%)'].apply(
                                        lambda v: '▲ Naik' if v > 0.1 else ('▼ Turun' if v < -0.1 else '– Netral'))
                                    st.dataframe(
                                        detail_tbl[['Program','Hari Libur','Efek (%)','Arah','Jumlah Event']]
                                        .style
                                        .applymap(lambda v: 'color:#34d399' if isinstance(v,str) and '▲' in v
                                                  else ('color:#f87171' if isinstance(v,str) and '▼' in v else ''))
                                        .format({'Efek (%)': '{:+.2f}%'}),
                                        width='stretch', height=360)

                                st.markdown("""<div class="info-box" style="margin-top:16px">
                                📊 <b>Cara baca:</b> Efek = % perubahan klaim dibanding rata-rata bulan normal.<br>
                                Contoh: <b>JKK +12% saat Idul Fitri</b> → klaim JKK rata-rata 12% lebih tinggi
                                di bulan yang mengandung Idul Fitri dibanding bulan biasa.<br>
                                Efek diekstrak dari <b>parameter posterior Prophet</b> (bukan kolom forecast agregat)
                                sehingga tiap program mendapat nilai yang benar-benar berbeda sesuai pola datanya.
                                Nama hari libur langsung dari <b>Google Calendar API Indonesia</b>.
                                </div>""", unsafe_allow_html=True)

    else:
        st.info("Klik **Jalankan Analisis ML** untuk memulai.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: PREDIKSI
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    p1, _ = st.columns([1, 3])
    with p1:
        target_pred = st.selectbox("Target", targets, key='pred_tgt')
        run_pred    = st.button("🔮 Hitung Prediksi", type="primary", width='stretch')

    ck_p    = f"{target_pred}_lags{n_lags}_test{test_pct}"
    ml_pred = results_cache.get(ck_p)

    if run_pred:
        if ml_pred is None:
            with st.spinner("Melatih model..."):
                ml_pred, err = run_ml(df, target_pred, n_lags, test_pct / 100)
            if err:
                st.error(err)
                ml_pred = None
            else:
                results_cache[ck_p] = ml_pred
                st.session_state.active_results = results_cache

    fut         = None
    fut_monthly = None

    if ml_pred:
        with st.spinner("Menghitung proyeksi..."):
            fut = forecast(df, ml_pred, n_future)

        df_raw_m = st.session_state.get('raw_monthly', None)
        if df_raw_m is not None and len(df_raw_m) > 0:
            if target_pred in df_raw_m.columns:
                try:
                    fut_monthly = compute_monthly_breakdown(df_raw_m, fut, target_pred)
                except Exception as e:
                    st.warning(f"Gagal hitung prediksi bulanan: {e}")
                    fut_monthly = None

        st.session_state['last_forecast']                          = fut
        st.session_state['last_forecast_monthly']                  = fut_monthly
        st.session_state[f'forecast_{target_pred}']                = fut
        st.session_state[f'forecast_monthly_{target_pred}']        = fut_monthly
        st.session_state[f'forecast_annual_{target_pred}']         = fut

        data_hash = hashlib.md5(df.to_csv().encode()).hexdigest()[:8]
        eid_pred  = f"{data_hash}_{target_pred}_L{n_lags}_T{test_pct}"
        label_pred = (f"📁 {datetime.now().strftime('%d/%m %H:%M')} | "
                      f"{target_pred} | {len(years)}yr | {len(active_progs)} prog")
        extra_snapshot = {
            k: st.session_state[k]
            for k in ['raw_monthly',
                      'forecast_Kasus', 'forecast_Nominal',
                      'forecast_monthly_Kasus', 'forecast_monthly_Nominal',
                      'forecast_annual_Kasus', 'forecast_annual_Nominal',
                      'last_forecast', 'last_forecast_monthly']
            if k in st.session_state and st.session_state[k] is not None
        }
        add_to_history(label_pred, eid_pred, df.copy(), dict(results_cache), extra_snapshot)

        future_yrs = sorted(fut['Tahun'].unique())
        yr_range   = f"{future_yrs[0]}-{future_yrs[-1]}"
        prog_list  = ", ".join(ml_pred['active_programs'])
        mode_note  = " | Mode 1 dataset (estimasi 5%/thn)" if single_yr else ""

        per_prog_info = ml_pred.get('per_prog', {})
        if per_prog_info:
            model_parts = " | ".join(
                f"<b>{cat}</b>→{info['best_name']}"
                for cat, info in per_prog_info.items()
            )
            badge_html = (
                '<div class="badge">'
                + "🎯 <b>Per-Program Model</b>: " + model_parts
                + " &nbsp;|&nbsp; Proyeksi <b>" + str(n_future) + " tahun</b> (" + yr_range + ")"
                + mode_note
                + "</div>"
            )
        else:
            best_used = ml_pred['best_name']
            badge_html = (
                '<div class="badge">'
                + "Model: <b>" + best_used + "</b>"
                + " &nbsp;|&nbsp; Proyeksi <b>" + str(n_future) + " tahun</b> (" + yr_range + ")"
                + " &nbsp;|&nbsp; Program: <b>" + prog_list + "</b>"
                + mode_note
                + "</div>"
            )
        st.markdown(badge_html, unsafe_allow_html=True)

        ptab_yr, ptab_mo = st.tabs(["📅 Prediksi Tahunan", "📆 Prediksi Bulanan"])

        with ptab_yr:
            hist = df_plot.groupby(['Tahun', 'Kategori'])[target_pred].sum().reset_index()
            hist['Jenis'] = 'Historis (Aktual)'
            fut_yr        = fut.copy()
            fut_yr['Jenis'] = 'Prediksi'

            st.markdown('<div class="sec">Tren Historis (Aktual) vs Prediksi</div>',
                        unsafe_allow_html=True)

            fig_main = go.Figure()
            cat_color = {c: COLORS[i % len(COLORS)]
                         for i, c in enumerate(sorted(hist['Kategori'].unique()))}

            for cat in sorted(hist['Kategori'].unique()):
                col = cat_color[cat]

                h = hist[hist['Kategori'] == cat].sort_values('Tahun')
                if len(h):
                    fig_main.add_trace(go.Scatter(
                        x=h['Tahun'], y=h[target_pred],
                        name=cat + " (Aktual)",
                        mode='lines+markers',
                        line=dict(color=col, width=2.5),
                        marker=dict(size=8, symbol='circle'),
                        legendgroup=cat,
                    ))

                p = fut_yr[fut_yr['Kategori'] == cat].sort_values('Tahun')
                if len(p):
                    x_p = list(p['Tahun'])
                    y_p = list(p[target_pred])
                    if len(h):
                        last_h = h.sort_values('Tahun').iloc[-1]
                        x_p = [int(last_h['Tahun'])] + x_p
                        y_p = [float(last_h[target_pred])] + y_p
                    fig_main.add_trace(go.Scatter(
                        x=x_p, y=y_p,
                        name=cat + " (Prediksi)",
                        mode='lines+markers',
                        line=dict(color=col, width=2.5, dash='dash'),
                        marker=dict(size=10, symbol='diamond'),
                        legendgroup=cat,
                    ))

            fig_main.add_vrect(
                x0=latest_year + 0.3, x1=future_yrs[-1] + 0.7,
                fillcolor=hex_to_rgba('#3b82f6', 0.05),
                line_width=0,
                annotation_text="▶ Zona Prediksi",
                annotation_position="top left",
                annotation_font=dict(color='#60a5fa', size=11, family='Inter'),
            )
            fig_main.add_vline(
                x=latest_year + 0.5,
                line_dash='dot', line_color=hex_to_rgba('#60a5fa', 0.6), line_width=1.5,
                annotation_text=f"← {latest_year} | {future_yrs[0]} →",
                annotation_font=dict(color='#475569', size=9),
                annotation_position="bottom right"
            )

            styled_chart(fig_main, height=540, legend_bottom=True, margin_b=140)
            fig_main.update_layout(
                xaxis=dict(dtick=1, showgrid=True, gridcolor='#0f1923'),
                yaxis_title=target_pred, xaxis_title='Tahun',
                hoverlabel=dict(bgcolor='#0d1f35', font_size=12))
            st.plotly_chart(fig_main, width='stretch')

            per_prog_inf = ml_pred.get('per_prog', {})
            if per_prog_inf:
                cards_html = '<div style="display:flex;flex-wrap:wrap;gap:8px;margin:12px 0;">'
                mape_grade = lambda m: (
                    'mpill-green' if m < 10 else 'mpill-blue' if m < 20
                    else 'mpill-yellow' if m < 50 else 'mpill-red')
                for cat, info in per_prog_inf.items():
                    mape_v = info.get('metrics', {}).get('MAPE (%)', None)
                    pill_cls = mape_grade(mape_v) if mape_v is not None else 'mpill-blue'
                    mape_txt = f"MAPE: {mape_v:.1f}%" if mape_v is not None else ""
                    cards_html += (
                        f'<div style="background:#0d1f35;border:1px solid #1e3a5f;'
                        f'border-radius:10px;padding:10px 14px;min-width:120px;">'
                        f'<div style="font-size:.65rem;color:#475569;font-weight:700;'
                        f'text-transform:uppercase;letter-spacing:1px;">{cat}</div>'
                        f'<div style="font-size:.9rem;font-weight:600;color:#e2e8f0;margin:4px 0;">'
                        f'{info["best_name"]}</div>'
                        f'<span class="mpill {pill_cls}">{mape_txt}</span>'
                        f'</div>'
                    )
                cards_html += '</div>'
                st.markdown(cards_html, unsafe_allow_html=True)

            st.markdown('<div class="sec">Nilai Prediksi per Tahun per Program</div>',
                        unsafe_allow_html=True)
            fig_bar = px.bar(
                fut, x='Tahun', y=target_pred, color='Kategori',
                barmode='group', color_discrete_sequence=COLORS,
                text=target_pred,
            )
            fig_bar.update_traces(texttemplate='%{text:,.0f}', textposition='outside',
                marker_line_width=0, textfont_size=9)
            styled_chart(fig_bar, height=400)
            fig_bar.update_layout(xaxis=dict(dtick=1))
            st.plotly_chart(fig_bar, width='stretch')

            if len(future_yrs) >= 2:
                st.markdown('<div class="sec">Total Klaim Prediksi — Waterfall per Tahun</div>',
                            unsafe_allow_html=True)
                tot_by_yr = fut.groupby('Tahun')[target_pred].sum().reset_index()
                tot_by_yr['delta'] = tot_by_yr[target_pred].diff().fillna(0)
                measures = ['absolute'] + ['relative'] * (len(tot_by_yr)-1)
                wf_y_vals = [tot_by_yr[target_pred].iloc[0]] + tot_by_yr['delta'].iloc[1:].tolist()
                fig_wf = go.Figure(go.Waterfall(
                    orientation='v',
                    measure=measures,
                    x=[str(y) for y in tot_by_yr['Tahun'].tolist()],
                    y=wf_y_vals,
                    textposition='outside',
                    text=[f"{v:+,.0f}" if i > 0 else f"{v:,.0f}" for i, v in enumerate(wf_y_vals)],
                    connector=dict(line=dict(color='#1e3a5f', width=1.5)),
                    increasing=dict(marker_color='#34d399'),
                    decreasing=dict(marker_color='#f87171'),
                    totals=dict(marker_color='#60a5fa')
                ))
                fig_wf.update_layout(**DARK, height=340, margin=dict(t=20,b=40,l=60,r=20),
                    showlegend=False, yaxis_title=target_pred)
                st.plotly_chart(fig_wf, width='stretch')

            st.markdown('<div class="sec">Distribusi per Tahun Prediksi</div>',
                        unsafe_allow_html=True)
            ncols = min(len(future_yrs), 3)
            pcols = st.columns(ncols)
            for i, fy in enumerate(future_yrs):
                with pcols[i % ncols]:
                    fy_d = fut[fut['Tahun'] == fy]
                    fp   = px.pie(fy_d, names='Kategori', values=target_pred,
                                  hole=0.4, title=str(fy),
                                  color_discrete_sequence=COLORS)
                    fp.update_traces(textinfo='label+percent', textposition='outside')
                    fp.update_layout(**DARK, showlegend=False, height=300,
                                     margin=dict(t=40, b=10, l=10, r=10))
                    st.plotly_chart(fp, width='stretch')

            if len(future_yrs) > 1:
                st.markdown('<div class="sec">Estimasi Pertumbuhan Total (%)</div>',
                            unsafe_allow_html=True)
                grow = []
                for cat in ml_pred['active_programs']:
                    cd = fut[fut['Kategori'] == cat][target_pred].values
                    if len(cd) >= 2:
                        g = (cd[-1] / (cd[0] + 1e-9) - 1) * 100
                        grow.append({'Kategori': cat, 'Pertumbuhan (%)': round(g, 2)})
                if grow:
                    gdf = pd.DataFrame(grow).sort_values('Pertumbuhan (%)', ascending=True)
                    fig_g = px.bar(gdf, x='Pertumbuhan (%)', y='Kategori',
                                   orientation='h', color='Pertumbuhan (%)',
                                   color_continuous_scale='RdYlGn',
                                   text='Pertumbuhan (%)')
                    fig_g.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    fig_g.update_layout(**DARK, coloraxis_showscale=False,
                                        height=max(260, len(active_progs) * 44 + 80),
                                        margin=dict(l=10, t=10, b=10, r=90))
                    st.plotly_chart(fig_g, width='stretch')

            st.markdown('<div class="sec">Tabel Prediksi Tahunan</div>',
                        unsafe_allow_html=True)
            tbl = fut[['Kategori', 'Tahun', target_pred]].copy().sort_values(['Kategori', 'Tahun'])
            fmt = 'Rp {:,.0f}' if target_pred == 'Nominal' else '{:,.0f}'
            tbl[target_pred] = tbl[target_pred].apply(lambda x: fmt.format(x))
            st.dataframe(tbl, width='stretch')

        with ptab_mo:
            if fut_monthly is not None and len(fut_monthly) > 0:
                st.markdown(
                    '<div class="info-box">'
                    'Prediksi bulanan dihitung dengan mendistribusikan total tahunan '
                    'menggunakan pola musiman (seasonal weights) dari data historis.'
                    '</div>',
                    unsafe_allow_html=True,
                )

                st.markdown('<div class="sec">Tren Bulanan: Aktual vs Prediksi</div>',
                            unsafe_allow_html=True)

                df_raw_m2 = st.session_state.get('raw_monthly', None)
                fig_mo    = go.Figure()

                for i, cat in enumerate(sorted(ml_pred['active_programs'])):
                    col = COLORS[i % len(COLORS)]

                    if df_raw_m2 is not None and target_pred in df_raw_m2.columns:
                        cat_raw = (df_raw_m2[df_raw_m2['Kategori'] == cat]
                                   .sort_values(['Tahun', 'Bulan']).copy())
                        if len(cat_raw):
                            cat_raw['Periode'] = (
                                cat_raw['Tahun'].astype(str) + '-'
                                + cat_raw['Bulan'].astype(str).str.zfill(2)
                            )
                            fig_mo.add_trace(go.Scatter(
                                x=cat_raw['Periode'],
                                y=cat_raw[target_pred],
                                name=cat + " (Aktual)",
                                mode='lines+markers',
                                line=dict(color=col, width=2),
                                marker=dict(size=5, symbol='circle'),
                                legendgroup=cat,
                            ))

                    cat_pred_mo = (fut_monthly[fut_monthly['Kategori'] == cat]
                                   .sort_values(['Tahun', 'Bulan']))
                    if len(cat_pred_mo):
                        fig_mo.add_trace(go.Scatter(
                            x=cat_pred_mo['Periode'],
                            y=cat_pred_mo[target_pred],
                            name=cat + " (Prediksi)",
                            mode='lines+markers',
                            line=dict(color=col, width=2, dash='dash'),
                            marker=dict(size=7, symbol='diamond'),
                            legendgroup=cat,
                        ))

                fig_mo.update_layout(
                    **DARK, height=520, hovermode='x unified',
                    xaxis_tickangle=-45,
                    legend=dict(
                        orientation='h', y=-0.35,
                        groupclick='toggleitem',
                        font=dict(size=11),
                    ),
                    margin=dict(b=150, t=20, l=70, r=20),
                    yaxis_title=target_pred,
                    xaxis_title='Periode (YYYY-MM)',
                )
                st.plotly_chart(fig_mo, width='stretch')

                st.markdown('<div class="sec">Detail Prediksi Bulanan per Program</div>',
                            unsafe_allow_html=True)
                sel_cat = st.selectbox(
                    "Pilih Program", sorted(ml_pred['active_programs']), key='mo_cat_sel'
                )
                cat_mo = (fut_monthly[fut_monthly['Kategori'] == sel_cat]
                          .sort_values(['Tahun', 'Bulan']))

                fig_cat = go.Figure()
                col_sel = COLORS[sorted(ml_pred['active_programs']).index(sel_cat) % len(COLORS)]

                if df_raw_m2 is not None and target_pred in df_raw_m2.columns:
                    h_mo = (df_raw_m2[df_raw_m2['Kategori'] == sel_cat]
                            .sort_values(['Tahun', 'Bulan']).copy())
                    if len(h_mo):
                        h_mo['Periode'] = (
                            h_mo['Tahun'].astype(str) + '-'
                            + h_mo['Bulan'].astype(str).str.zfill(2)
                        )
                        fig_cat.add_trace(go.Bar(
                            x=h_mo['Periode'], y=h_mo[target_pred],
                            name='Aktual', marker_color=col_sel, opacity=0.7,
                        ))

                fig_cat.add_trace(go.Bar(
                    x=cat_mo['Periode'], y=cat_mo[target_pred],
                    name='Prediksi',
                    marker_color=col_sel,
                    marker_pattern_shape='/',
                ))
                fig_cat.update_layout(
                    **DARK, height=400, barmode='group',
                    xaxis_tickangle=-45,
                    legend=dict(orientation='h'),
                    margin=dict(b=100, t=20),
                    title=f"{sel_cat} — Aktual Bulanan & Prediksi",
                )
                st.plotly_chart(fig_cat, width='stretch')

                st.markdown('<div class="sec">Tabel Prediksi Bulanan Lengkap</div>',
                            unsafe_allow_html=True)
                tbl_mo = (fut_monthly[['Kategori', 'Tahun', 'Bulan', 'Periode', target_pred]]
                          .copy().sort_values(['Kategori', 'Tahun', 'Bulan']))
                fmt = 'Rp {:,.0f}' if target_pred == 'Nominal' else '{:,.0f}'
                tbl_mo[target_pred] = tbl_mo[target_pred].apply(lambda x: fmt.format(x))
                st.dataframe(tbl_mo, width='stretch', height=420)

            else:
                df_raw_debug = st.session_state.get('raw_monthly', None)
                if df_raw_debug is None:
                    st.warning("**Data bulanan belum tersimpan.** Upload ulang file dataset Anda.")
                elif target_pred not in df_raw_debug.columns:
                    st.warning(f"Kolom **{target_pred}** tidak ditemukan di data bulanan.")
                else:
                    st.warning("Terjadi kesalahan saat menghitung prediksi bulanan. Coba klik **Hitung Prediksi** ulang.")

    else:
        st.info("Klik **Hitung Prediksi** — model ML akan dilatih otomatis jika belum ada.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: EXPORT
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="sec">Export Laporan</div>', unsafe_allow_html=True)
    ec1, ec2 = st.columns(2)

    best_ml             = next((v for v in results_cache.values() if v), None)
    last_fut            = st.session_state.get('last_forecast', None)
    fut_ann_kasus       = st.session_state.get('forecast_annual_Kasus',   None)
    fut_ann_nominal     = st.session_state.get('forecast_annual_Nominal', None)
    fut_mo_kasus        = st.session_state.get('forecast_monthly_Kasus',   None)
    fut_mo_nominal      = st.session_state.get('forecast_monthly_Nominal', None)

    if fut_ann_kasus is None and fut_ann_nominal is None and last_fut is not None:
        tc_lf = [c for c in last_fut.columns if c not in ['Kategori','Tahun','Type']]
        if tc_lf:
            if tc_lf[0] == 'Kasus':
                fut_ann_kasus = last_fut
            else:
                fut_ann_nominal = last_fut

    if fut_mo_kasus is None and fut_mo_nominal is None:
        lm = st.session_state.get('last_forecast_monthly', None)
        if lm is not None and len(lm) > 0:
            tc_lm = [c for c in lm.columns if c not in
                     ['Kategori','Tahun','Bulan','Periode','Type']]
            if tc_lm:
                if tc_lm[0] == 'Kasus':
                    fut_mo_kasus = lm
                else:
                    fut_mo_nominal = lm

    has_mo_kasus   = fut_mo_kasus   is not None and len(fut_mo_kasus)   > 0
    has_mo_nominal = fut_mo_nominal is not None and len(fut_mo_nominal) > 0

    with ec1:
        st.markdown("**📊 Excel dengan Chart Terintegrasi**")
        st.caption("Sheet: Data Gabungan · Pivot Kasus · Prediksi Tahunan · Prediksi Bulanan · Bulanan Detail · ML Results")

        has_ann_k = fut_ann_kasus   is not None and len(fut_ann_kasus)   > 0
        has_ann_n = fut_ann_nominal is not None and len(fut_ann_nominal) > 0
        has_mo_kasus   = fut_mo_kasus   is not None and len(fut_mo_kasus)   > 0
        has_mo_nominal = fut_mo_nominal is not None and len(fut_mo_nominal) > 0

        status_html = '<div class="info-box">'
        status_html += '<b>Prediksi Tahunan:</b><br>'
        status_html += (f'✅ Kasus tahunan siap ({len(fut_ann_kasus)} baris)<br>' if has_ann_k else '⚠️ Kasus tahunan belum ada<br>')
        status_html += (f'✅ Nominal tahunan siap ({len(fut_ann_nominal)} baris)<br>' if has_ann_n else '⚠️ Nominal tahunan belum ada<br>')
        status_html += '<br><b>Prediksi Bulanan:</b><br>'
        status_html += (f'✅ Kasus bulanan siap ({len(fut_mo_kasus)} baris)<br>' if has_mo_kasus else '⚠️ Kasus bulanan belum ada<br>')
        status_html += (f'✅ Nominal bulanan siap ({len(fut_mo_nominal)} baris)' if has_mo_nominal else '⚠️ Nominal bulanan belum ada')
        status_html += '</div>'
        st.markdown(status_html, unsafe_allow_html=True)

        if st.button("⚙️ Generate Excel", width='stretch'):
            with st.spinner("Membuat Excel..."):
                xlsx = export_excel(
                    df, best_ml, last_fut,
                    fut_kasus=fut_ann_kasus,
                    fut_nominal=fut_ann_nominal,
                    fut_monthly_kasus=fut_mo_kasus,
                    fut_monthly_nominal=fut_mo_nominal,
                )
            st.download_button(
                "⬇️ Download Excel", data=xlsx,
                file_name=f"BPJS_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width='stretch')

    with ec2:
        st.markdown("**📄 CSV Data Gabungan**")
        df_sorted = df.sort_values(['Tahun', 'Kategori']).reset_index(drop=True)
        st.download_button(
            "⬇️ Download CSV", data=df_sorted.to_csv(index=False),
            file_name=f"BPJS_Data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv", width='stretch')

        if has_mo_kasus or has_mo_nominal:
            st.markdown("**📄 CSV Prediksi Bulanan (Gabungan)**")
            frames_csv = []
            if has_mo_kasus:
                frames_csv.append(fut_mo_kasus.assign(Target='Kasus'))
            if has_mo_nominal:
                frames_csv.append(fut_mo_nominal.assign(Target='Nominal'))
            combined_csv = (pd.concat(frames_csv, ignore_index=True)
                            .sort_values(['Tahun', 'Bulan', 'Kategori', 'Target'])
                            .reset_index(drop=True))
            st.download_button(
                "⬇️ Download CSV Bulanan", data=combined_csv.to_csv(index=False),
                file_name=f"BPJS_Bulanan_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv", width='stretch')

    st.markdown('<div class="sec">Preview Data Aktif</div>', unsafe_allow_html=True)
    st.info(f"**{len(df)} baris** | **{len(active_progs)} program aktif** ({', '.join(active_progs)}) | **Tahun: {', '.join(map(str, years))}**")
    st.dataframe(df, width='stretch', height=360)