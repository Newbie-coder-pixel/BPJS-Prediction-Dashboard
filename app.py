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

# ── Keep-alive: cegah app hibernasi di Streamlit Cloud ────────────────────
# JS ping halaman sendiri setiap 10 menit — Streamlit hibernasi setelah ~15 menit idle
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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');
*{box-sizing:border-box;}
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:#05090f;color:#e2e8f0;}

/* ══════════════════════════════════════════════
   SEMBUNYIKAN UI BAWAAN STREAMLIT — AMAN
   ══════════════════════════════════════════════ */
[data-testid="stToolbar"]{display:none !important;}
[data-testid="stToolbarActions"]{display:none !important;}
#MainMenu{display:none !important;}
[data-testid="stHeader"]{display:none !important;}
header[data-testid="stHeader"]{display:none !important;}
footer{display:none !important;}
[data-testid="stDeployButton"]{display:none !important;}
[data-testid="stStatusWidget"]{display:none !important;}
.viewerBadge_container__1QSob{display:none !important;}
.viewerBadge_link__1S137{display:none !important;}
#stDecoration{display:none !important;}
[data-testid="stDecoration"]{display:none !important;}

/* ── KPI Cards ── */
.kpi{
  background:linear-gradient(145deg,#0d1424 0%,#0a1020 100%);
  border:1px solid #1a2840;border-radius:16px;padding:24px 20px;
  text-align:center;position:relative;overflow:hidden;
  transition:transform .2s,border-color .2s;
}
.kpi::before{
  content:'';position:absolute;top:0;left:0;right:0;height:3px;
  background:linear-gradient(90deg,#3b82f6,#8b5cf6);
}
.kpi .val{font-size:2rem;font-weight:800;
          background:linear-gradient(135deg,#60a5fa,#a78bfa);
          -webkit-background-clip:text;-webkit-text-fill-color:transparent;
          font-family:'JetBrains Mono',monospace;line-height:1.2;}
.kpi .lbl{font-size:.68rem;color:#475569;text-transform:uppercase;
          letter-spacing:1.5px;margin-top:8px;font-weight:600;}
.kpi .delta{font-size:.75rem;margin-top:6px;font-weight:500;}
.delta-pos{color:#34d399;} .delta-neg{color:#f87171;} .delta-neu{color:#94a3b8;}

/* ── Badge ── */
.badge{
  background:linear-gradient(135deg,#0f1f35 0%,#0a1a0a 100%);
  border:1px solid #1e3a5f;border-radius:12px;
  padding:14px 20px;margin:12px 0;font-size:.9rem;line-height:1.7;
}
.badge b{color:#93c5fd;}

/* ── Section header ── */
.sec{
  font-size:.68rem;font-weight:700;color:#334155;
  text-transform:uppercase;letter-spacing:2.5px;
  margin:28px 0 12px;padding-bottom:8px;
  border-bottom:1px solid #0f1f35;
}

/* ── Alerts ── */
.warn{background:#1a1200;border:1px solid #92400e50;border-radius:12px;
      padding:14px 18px;color:#fbbf24;font-size:.88rem;margin:10px 0;
      border-left:3px solid #f59e0b;}
.info-box{background:#0a1628;border:1px solid #1e3a5f;border-radius:12px;
          padding:16px 18px;font-size:.84rem;color:#7dd3fc;
          margin:10px 0;line-height:1.85;border-left:3px solid #3b82f6;}
.success-box{background:#052e16;border:1px solid #14532d50;border-radius:12px;
          padding:14px 18px;font-size:.84rem;color:#86efac;
          margin:10px 0;border-left:3px solid #22c55e;}

/* ── Program tags ── */
.tag-add{display:inline-block;background:#14532d20;color:#86efac;border:1px solid #14532d;
         border-radius:6px;padding:3px 10px;font-size:.75rem;margin:2px;font-weight:500;}
.tag-rem{display:inline-block;background:#450a0a20;color:#fca5a5;border:1px solid #7f1d1d50;
         border-radius:6px;padding:3px 10px;font-size:.75rem;margin:2px;font-weight:500;}
.tag-stable{display:inline-block;background:#1e3a5f20;color:#93c5fd;border:1px solid #1e3a5f;
            border-radius:6px;padding:3px 10px;font-size:.75rem;margin:2px;font-weight:500;}

/* ── Metric pill ── */
.mpill{display:inline-block;padding:4px 12px;border-radius:20px;
       font-size:.78rem;font-weight:600;margin:2px;}
.mpill-green{background:#052e1640;color:#4ade80;border:1px solid #14532d;}
.mpill-blue{background:#1e3a5f40;color:#60a5fa;border:1px solid #1e3a5f;}
.mpill-yellow{background:#451a0340;color:#fbbf24;border:1px solid #78350f;}
.mpill-red{background:#450a0a40;color:#f87171;border:1px solid #7f1d1d;}

/* ── Insight card ── */
.insight-card{
  background:linear-gradient(135deg,#0a1628,#0d1f0d);
  border:1px solid #1e3a5f;border-radius:14px;
  padding:20px;margin:8px 0;
}
.insight-card .ic-title{font-size:.7rem;color:#475569;text-transform:uppercase;
  letter-spacing:1.5px;font-weight:700;margin-bottom:8px;}
.insight-card .ic-val{font-size:1.4rem;font-weight:800;color:#e2e8f0;font-family:'JetBrains Mono',monospace;}
.insight-card .ic-sub{font-size:.8rem;color:#64748b;margin-top:4px;}

/* ── Progress bar ── */
.prog-bar-wrap{background:#0f1923;border-radius:6px;height:8px;margin:4px 0;}
.prog-bar-fill{height:8px;border-radius:6px;
  background:linear-gradient(90deg,#3b82f6,#8b5cf6);}

/* ── Sidebar ── */
section[data-testid="stSidebar"]{background:#030712 !important;}
section[data-testid="stSidebar"] .stMarkdown{color:#94a3b8;}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"]{gap:4px;background:#0a0f1a;
  border-radius:12px;padding:4px;}
.stTabs [data-baseweb="tab"]{border-radius:9px;padding:8px 18px;
  font-size:.85rem;font-weight:500;color:#475569;}
.stTabs [aria-selected="true"]{background:#1e2d45 !important;color:#e2e8f0 !important;}

/* ── Dataframe ── */
.stDataFrame{border-radius:10px;overflow:hidden;}
</style>
""", unsafe_allow_html=True)

# Badge disembunyikan via CSS data-testid selector di atas

# ══════════════════════════════════════════════════════════════════════════════
# PERSISTENT HISTORY — saved to disk, survives Streamlit restart
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

# ── Session state (in-memory, loaded from disk on first run) ──────────────────
for k, v in [('active_data', None), ('active_results', {}), ('active_entry_id', None), ('history_loaded', False)]:
    if k not in st.session_state:
        st.session_state[k] = v

DARK = dict(template='plotly_dark', paper_bgcolor='#05090f', plot_bgcolor='#05090f',
            font_color='#e2e8f0', font_family='Inter',
            title_font_family='Inter', title_font_color='#e2e8f0')
COLORS = ['#60a5fa','#34d399','#fb923c','#a78bfa','#f87171',
          '#fbbf24','#38bdf8','#f472b6','#4ade80','#e879f9']
COLORS_ALPHA = {c: c for c in COLORS}  # placeholder for rgba conversion

def hex_to_rgba(hex_c, alpha=1.0):
    h = hex_c.lstrip('#')
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f'rgba({r},{g},{b},{alpha})'

def styled_chart(fig, height=400, legend_bottom=True, margin_b=80):
    """Apply consistent styling to all plotly charts."""
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
# ML CORE — Per-Program Best Model (fast, accurate, per-program prediction)
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# FORECASTING METHODS — Smart selection based on data size
# ══════════════════════════════════════════════════════════════════════════════

def score_model(yt, yp):
    """Return metrics for a prediction pair."""
    yt, yp = np.array(yt, dtype=float), np.array(yp, dtype=float)
    mae  = float(mean_absolute_error(yt, yp))
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    # R2 is only meaningful with 3+ points; with LOO on small data use None
    r2   = float(r2_score(yt, yp)) if len(yt) >= 3 else None
    mape = float(np.mean(np.abs((yt - yp) / (np.abs(yt) + 1e-9))) * 100)
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE (%)': mape}


# ── Statistical methods (best for ≤10 years data) ────────────────────────────

def forecast_holt(history, n_steps, alpha=None, beta=None):
    """Holt's Double Exponential Smoothing — trend-aware, ideal for small data."""
    y = np.array(history, dtype=float)
    n = len(y)
    if n < 2:
        return np.array([y[-1]] * n_steps)

    # Auto-optimize alpha & beta via grid search on in-sample LOO
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

    # Final fit with best params
    lvl = np.zeros(n); trnd = np.zeros(n)
    lvl[0] = y[0]; trnd[0] = y[1] - y[0] if n > 1 else 0
    for i in range(1, n):
        lvl[i]  = best_a * y[i] + (1-best_a) * (lvl[i-1] + trnd[i-1])
        trnd[i] = best_b * (lvl[i] - lvl[i-1]) + (1-best_b) * trnd[i-1]
    preds = np.array([lvl[-1] + (s+1)*trnd[-1] for s in range(n_steps)])
    return preds, best_a, best_b, lvl, trnd


def forecast_ses(history, n_steps):
    """Simple Exponential Smoothing — best for flat/no-trend series."""
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
    """Weighted Moving Average — recent values weighted more."""
    y = np.array(history, dtype=float)
    w = min(window, len(y))
    weights = np.arange(1, w+1, dtype=float)
    weights /= weights.sum()
    base = float(np.dot(y[-w:], weights))
    # Estimate trend from last 2 points
    if len(y) >= 2:
        trend = float(y[-1] - y[-2])
    else:
        trend = 0.0
    trend = np.clip(trend, -abs(base)*0.3, abs(base)*0.3)
    return np.array([base + trend*(s+1)*0.5 for s in range(n_steps)])


def loo_cv_stat(history, method_fn, n_steps=1):
    """LOO cross-validation for a statistical forecasting function."""
    y = np.array(history, dtype=float)
    n = len(y)
    yt_all, yp_all = [], []
    for leave in range(1, n):  # predict each point from previous
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


# ── ML methods (for ≥8 years data) ──────────────────────────────────────────

def build_features(series, n_lags=1, cat_id=0.0):
    """Build lag features. Keep simple for small data."""
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
    """LOO cross-validation for an ML model."""
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


# ══════════════════════════════════════════════════════════════════════════════
# MAIN: TRAIN BEST PER PROGRAM
# ══════════════════════════════════════════════════════════════════════════════

def train_best_per_program(df, target, n_lags, test_ratio):
    """
    Smart model selection per program:
    - ≤7 years: statistical methods (Holt, SES, WMA) — proven for small data
    - ≥8 years: ML models via LOO-CV
    Always retrain winner on ALL data for forecasting.
    """
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

        # ── STATISTICAL METHODS (always evaluated) ────────────────────────
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

        # ── ML METHODS (only if ≥8 years) ────────────────────────────────
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
                # Retrain on full data
                try:
                    mdl_full = copy.deepcopy(mdl_obj)
                    Xc_s = sc_full.transform(Xc)
                    mdl_full.fit(Xc_s if mname in SCALED_MODELS else Xc, yc)
                    ml_models_fitted[mname] = {'model': mdl_full, 'scaler': sc_full, 'Xc': Xc, 'yc': yc}
                except:
                    pass

        # ── PICK BEST across stat + ML ────────────────────────────────────
        all_scores = {**stat_scores, **ml_scores}
        # For small data: MAPE is the only reliable metric
        # Filter out clearly broken models (MAPE > 200%)
        valid = {m: s for m, s in all_scores.items() if s['MAPE (%)'] < 200}
        pool  = valid if valid else all_scores
        if not pool:
            pool = stat_scores  # fallback to stat

        # Best = lowest MAPE (most reliable for small N)
        best_name = min(pool, key=lambda m: pool[m]['MAPE (%)'])
        best_sc   = all_scores[best_name]

        # Store per-program info
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
            # Store stat method params for reproducibility
            entry['stat_fn_name'] = best_name
        else:
            # Store fitted ML model
            if best_name in ml_models_fitted:
                info_ml = ml_models_fitted[best_name]
                entry['best_model'] = info_ml['model']
                entry['scaler']     = info_ml['scaler']
                entry['n_lags_used'] = min(n_lags, 2)

        per_prog[cat] = entry

    # ── Build summary tables ──────────────────────────────────────────────
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

    # Avg metrics across programs — use MAPE as primary (R2 unreliable for small N)
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


# Aliases
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

    # Use per_prog data directly — more accurate since each program has its own model
    bpp = ml_result.get('best_per_program', pd.DataFrame())
    per_prog = ml_result.get('per_prog', {})

    # Aggregate metrics across all programs
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
# INDONESIAN HOLIDAYS — 100% dari Google Calendar API (no hardcoded)
# ══════════════════════════════════════════════════════════════════════════════

GCAL_ID  = "en.indonesian%23holiday%40group.v.calendar.google.com"
GCAL_KEY = "AIzaSyD1gERwlm6R_mnm6Jd1sap3dFuPz0hpBfQ"

# Window overrides: hari libur tertentu punya dampak lebih luas pada klaim BPJS
_WINDOW_RULES = {
    # keyword (lowercase) → (lower_window, upper_window)
    'idul fitri'          : (-7,  7),   # Lebaran — dampak 2 minggu
    'lebaran'             : (-7,  7),
    'ramadan'             : ( 0, 29),   # Ramadhan — 1 bulan penuh
    'ramadhan'            : ( 0, 29),
    'puasa'               : ( 0, 29),
    'idul adha'           : (-3,  3),   # Kurban — 1 minggu
    'natal'               : (-2,  2),   # Natal — H-2 s/d H+2
    'christmas'           : (-2,  2),
    'tahun baru'          : (-1,  2),   # New Year
    'new year'            : (-1,  2),
    'cuti bersama'        : (-1,  1),
    'default'             : (-1,  1),   # semua holiday lainnya
}

def _get_window(name: str):
    """Tentukan lower/upper window berdasarkan nama hari libur."""
    nl = name.lower()
    for keyword, (lo, hi) in _WINDOW_RULES.items():
        if keyword != 'default' and keyword in nl:
            return lo, hi
    return _WINDOW_RULES['default']


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_google_holidays(year_start: int = 2019, year_end: int = 2028) -> list:
    """
    Ambil semua hari libur Indonesia dari Google Calendar API.
    - Mengambil data per tahun (year_start s/d year_end)
    - Pagination otomatis (nextPageToken) agar tidak ada yang terlewat
    - Window dampak disesuaikan otomatis per jenis hari libur
    - Return [] jika API error (graceful fallback)
    """
    import urllib.request, urllib.parse, json

    base_url = f"https://www.googleapis.com/calendar/v3/calendars/{GCAL_ID}/events"
    all_rows  = []
    seen_keys = set()   # deduplikasi ds+holiday

    for year in range(year_start, year_end + 1):
        page_token = None
        while True:
            params = {
                'key'          : GCAL_KEY,
                'timeMin'      : f'{year}-01-01T00:00:00Z',
                'timeMax'      : f'{year}-12-31T23:59:59Z',
                'maxResults'   : '2500',          # max allowed by API
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
                break   # network error — skip year, try next

            # Cek error dari API
            if 'error' in data:
                return []   # key invalid / quota habis — return kosong

            for item in data.get('items', []):
                # Ambil tanggal (all-day event: pakai 'date', bukan 'dateTime')
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

            # Pagination
            page_token = data.get('nextPageToken')
            if not page_token:
                break   # semua halaman sudah diambil

    return all_rows


def build_holiday_df() -> pd.DataFrame:
    """
    Build DataFrame hari libur dari Google Calendar API murni.
    Jika API gagal, kembalikan DataFrame kosong (Prophet tetap jalan tanpa holiday).
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


# Build at startup — pure Google Calendar, cached 24 jam
INDONESIAN_HOLIDAYS = build_holiday_df()



# ══════════════════════════════════════════════════════════════════════════════
# PROPHET — time-series forecasting with Indonesian calendar
# Why Prophet over SARIMA:
#   - Handles holiday effects explicitly (Ramadhan, Lebaran, etc.)
#   - Works well with yearly data aggregated to monthly
#   - More interpretable trend + seasonality components
#   - No stationarity requirement
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

    # Floor value — Prophet tidak akan pernah prediksi di bawah ini
    y_floor = 0.0
    y_cap   = float(cat_df['y'].max()) * 3.0  # cap atas wajar: 3x nilai tertinggi historis

    try:
        # Pilih mode berdasarkan panjang data
        n_data = len(cat_df)
        if n_data >= 24:
            # Data cukup → multiplicative (lebih akurat untuk klaim yang tumbuh)
            s_mode = 'multiplicative'
            cp_scale = 0.05   # konservatif agar tidak overfit
        else:
            # Data sedikit → additive lebih stabil, hindari prediksi liar
            s_mode = 'additive'
            cp_scale = 0.03

        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            holidays=holidays_df,
            seasonality_mode=s_mode,
            interval_width=0.80,          # 80% CI (lebih konservatif dari 95%)
            changepoint_prior_scale=cp_scale,
            seasonality_prior_scale=5.0,  # batasi efek musiman agar tidak liar
            holidays_prior_scale=5.0,
            growth='flat' if n_data < 12 else 'linear',  # flat untuk data sangat sedikit
        )
        m.fit(cat_df)
        future = m.make_future_dataframe(periods=n_months, freq='MS')
        fc = m.predict(future)

        # ── Hard clamp: prediksi tidak pernah negatif ──────────────────
        fc['yhat']       = fc['yhat'].clip(lower=y_floor)
        fc['yhat_lower'] = fc['yhat_lower'].clip(lower=y_floor)
        fc['yhat_upper'] = fc['yhat_upper'].clip(lower=y_floor)
        # Juga clamp atas — hindari angka astronomis
        fc['yhat']       = fc['yhat'].clip(upper=y_cap)
        fc['yhat_upper'] = fc['yhat_upper'].clip(upper=y_cap * 1.2)

        # In-sample metrics
        hist_pred = fc[fc['ds'].isin(cat_df['ds'])]
        if len(hist_pred) > 0:
            yt = cat_df.set_index('ds').loc[hist_pred['ds'], 'y'].values
            yp = hist_pred['yhat'].values
            r2_is   = float(r2_score(yt, yp)) if len(yt) > 1 else 0.0
            mape_is = float(np.mean(np.abs((yt - yp)/(np.abs(yt)+1e-9)))*100)
        else:
            r2_is = mape_is = 0.0

        n_holidays = len(holidays_df) if holidays_df is not None else 0
        return {'model': m, 'forecast': fc, 'history': cat_df,
                'r2_insample': r2_is, 'mape_insample': mape_is,
                'n_holidays': n_holidays, 'gcal_used': n_holidays > 0}, None
    except Exception as e:
        return None, str(e)



def forecast(df, ml, n_years):
    """Forecast using each program's own best model (stat or ML)."""
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
            # Use statistical method for multi-step: iteratively forecast
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

        # ML model path
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
                    # Sanity: no more than 50% drop or 100% spike
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
# EXPORT EXCEL — Kasus & Nominal bulanan digabung dalam 1 sheet + masing-masing chart
# ══════════════════════════════════════════════════════════════════════════════

def _write_monthly_block(ws, wb, df_mo, target_col, hdr_fmt, num_fmt,
                         start_row, sheet_name, chart_title):
    """
    Tulis satu blok tabel pivot bulanan ke sheet ws mulai dari start_row.
    Kembalikan (next_free_row, chart_object).
    start_row = baris pertama header (0-indexed untuk xlsxwriter).
    """
    if df_mo is None or len(df_mo) == 0:
        return start_row, None

    # Pivot: baris=Periode, kolom=Kategori
    piv = (df_mo.pivot_table(index='Periode', columns='Kategori',
                             values=target_col, aggfunc='sum')
           .reset_index()
           .sort_values('Periode')
           .reset_index(drop=True))

    n_rows = len(piv)
    n_cats = len(piv.columns) - 1   # exclude Periode

    # ── tulis header ──────────────────────────────────────────────────────
    for col_idx, col_name in enumerate(piv.columns):
        ws.write(start_row, col_idx, str(col_name), hdr_fmt)
        ws.set_column(col_idx, col_idx, 16)

    # ── tulis data ────────────────────────────────────────────────────────
    for row_idx in range(n_rows):
        ws.write(start_row + 1 + row_idx, 0, piv.iloc[row_idx, 0])   # Periode (teks)
        for col_idx in range(1, n_cats + 1):
            ws.write(start_row + 1 + row_idx, col_idx,
                     piv.iloc[row_idx, col_idx], num_fmt)

    # ── buat chart ────────────────────────────────────────────────────────
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

    # baris berikutnya setelah tabel (beri jarak 2 baris)
    next_row = start_row + n_rows + 3
    return next_row, ch, start_row + n_rows   # (next_free, chart, last_data_row)


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

        # ── Sheet 1: Data Gabungan ────────────────────────────────────────
        df_sorted = df.sort_values(['Tahun', 'Kategori']).reset_index(drop=True)
        df_sorted.to_excel(writer, sheet_name='Data Gabungan', index=False)
        ws1 = writer.sheets['Data Gabungan']
        for i, c in enumerate(df_sorted.columns):
            ws1.write(0, i, c, hdr)
            ws1.set_column(i, i, 22)

        # ── Sheet 2: Pivot Kasus ──────────────────────────────────────────
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

        # ── Sheet 3: Prediksi Tahunan — Kasus & Nominal DIGABUNG ────────────
        # Layout per blok:
        #   Row 0     : judul seksi
        #   Row 1     : header (Tahun | Program1 | Program2 | ...)
        #   Row 2..N+1: data (historis aktual abu-abu, prediksi biru)
        #   Chart     : di sebelah kanan tabel
        # ─────────────────────────────────────────────────────────────────────

        has_fut_k = fut_kasus   is not None and len(fut_kasus)   > 0
        has_fut_n = fut_nominal is not None and len(fut_nominal) > 0

        # Jika tidak ada keduanya, fallback ke fut_df lama
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

            # Format tambahan
            hdr_sec  = wb.add_format({'bold': True, 'bg_color': '#0f2744',
                                       'font_color': '#93c5fd', 'font_size': 12})
            hdr_yr   = wb.add_format({'bold': True, 'bg_color': '#1e3a5f',
                                       'font_color': 'white', 'border': 1,
                                       'align': 'center'})
            num_k    = wb.add_format({'num_format': '#,##0',   'border': 1})
            num_n    = wb.add_format({'num_format': '#,##0',   'border': 1})
            num_hist = wb.add_format({'num_format': '#,##0',   'border': 1,
                                       'bg_color': '#1a1a2e'})   # baris historis sedikit lebih gelap

            cursor = 0

            def _write_annual_block(ws, wb, df_hist, df_pred, value_col,
                                    title_txt, cursor, ws_name):
                """
                Tulis blok tahunan: gabungkan historis aktual + prediksi.
                df_hist: DataFrame dengan kolom Tahun, Kategori, value_col (aktual)
                df_pred: DataFrame dengan kolom Kategori, Tahun, value_col (prediksi)
                Kembalikan cursor setelah blok.
                """
                # Pivot historis
                ph = (df_hist.groupby(['Tahun','Kategori'])[value_col].sum()
                      .reset_index()
                      .pivot(index='Tahun', columns='Kategori', values=value_col)
                      .reset_index())
                ph['_type'] = 'Aktual'

                # Pivot prediksi
                pp = (df_pred.groupby(['Tahun','Kategori'])[value_col].sum()
                      .reset_index()
                      .pivot(index='Tahun', columns='Kategori', values=value_col)
                      .reset_index())
                pp['_type'] = 'Prediksi'

                # Gabungkan
                combined = pd.concat([ph, pp], ignore_index=True)\
                             .sort_values('Tahun').reset_index(drop=True)

                cats = [c for c in combined.columns if c not in ['Tahun','_type']]
                n_rows = len(combined)
                n_cats = len(cats)

                # Judul seksi
                ws.merge_range(cursor, 0, cursor, n_cats + 1, title_txt, hdr_sec)
                hdr_row = cursor + 1

                # Header: Tahun | Tipe | Cat1 | Cat2 | ...
                ws.write(hdr_row, 0, 'Tahun',   hdr_yr); ws.set_column(0, 0, 10)
                ws.write(hdr_row, 1, 'Tipe',    hdr_yr); ws.set_column(1, 1, 12)
                for ci, cat in enumerate(cats):
                    ws.write(hdr_row, ci + 2, cat, hdr_yr)
                    ws.set_column(ci + 2, ci + 2, 20)

                # Data rows
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

                # ── Chart: 1 series per program, tiap series = historis + prediksi ──
                # Karena xlsxwriter tidak bisa beda warna per baris di 1 series,
                # kita buat 2 series per program: Aktual (solid) & Prediksi (dashed)
                ch = wb.add_chart({'type': 'line'})

                # Index baris aktual dan prediksi
                aktual_rows  = [data_start + i for i, r in combined.iterrows()
                                if r['_type'] == 'Aktual']
                pred_rows    = [data_start + i for i, r in combined.iterrows()
                                if r['_type'] == 'Prediksi']

                CHART_COLORS = ['#4472C4','#ED7D31','#A9D18E','#FF0000',
                                '#7030A0','#00B0F0','#92D050','#FFC000']

                for ci, cat in enumerate(cats):
                    col_excel = ci + 2
                    color     = CHART_COLORS[ci % len(CHART_COLORS)]

                    # Series Aktual — solid line, circle marker
                    if aktual_rows:
                        # Untuk line chart yang rapi, kita harus pass semua row sebagai 1 series
                        # Tapi row aktual & prediksi bergantian → pisahkan berdasarkan _type
                        # Solusi: buat 2 chart series terpisah, sambung di titik terakhir aktual
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

                    # Series Prediksi — dashed line, diamond marker
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
                ch.set_style(10)   # style bawaan Excel yang lebih rapi

                # Taruh chart di kolom setelah tabel (col = n_cats+2+1, row = hdr_row)
                chart_col = xl_col_to_name(n_cats + 3)
                ws.insert_chart(f'{chart_col}{hdr_row + 1}', ch)

                # Kembalikan cursor setelah blok (jarak 3 baris)
                return last_data_row + 4

            # Ambil historis aktual dari df (sudah tahunan)
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

        # ── Sheet 4: Prediksi Bulanan — Kasus & Nominal DIGABUNG ──────────
        # Layout:
        #   Baris 0      : judul seksi "PREDIKSI KASUS (BULANAN)"
        #   Baris 1      : header pivot Kasus
        #   Baris 2..N+1 : data Kasus
        #   gap 2 baris
        #   Baris N+4    : judul seksi "PREDIKSI NOMINAL (BULANAN)"
        #   Baris N+5    : header pivot Nominal
        #   dst.
        #   Chart Kasus  : di kolom H, baris 1
        #   Chart Nominal: di kolom H, baris N+5
        # ─────────────────────────────────────────────────────────────────

        has_kasus   = fut_monthly_kasus   is not None and len(fut_monthly_kasus)   > 0
        has_nominal = fut_monthly_nominal is not None and len(fut_monthly_nominal) > 0

        if has_kasus or has_nominal:
            ws4_name = 'Prediksi Bulanan'
            # Buat sheet kosong dulu
            writer.book.add_worksheet(ws4_name)
            ws4 = writer.sheets[ws4_name]

            cursor = 0   # baris saat ini (0-indexed)

            # ── Blok Kasus ────────────────────────────────────────────────
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

                # Chart Kasus
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
                # Taruh chart di kanan tabel (kolom = ncat_k + 2)
                chart_col_k = xl_col_to_name(ncat_k + 2)
                ws4.insert_chart(f'{chart_col_k}{header_row_k + 1}', ch_k)

                cursor = last_data_row_k + 3   # gap 2 baris

            # ── Blok Nominal ──────────────────────────────────────────────
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

                # Chart Nominal
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

        # ── Sheet 5: Bulanan Detail flat (semua kolom, mudah di-filter) ───
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
                # Merge on Periode+Tahun+Bulan+Kategori
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
            # Number fmt untuk kolom angka
            for ci, col_name in enumerate(detail_all.columns):
                if col_name in ('Kasus', 'Nominal'):
                    for ri in range(len(detail_all)):
                        ws5.write(ri + 1, ci, detail_all.iloc[ri][col_name], num_fmt)

        # ── Sheet 6: ML Results ───────────────────────────────────────────
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
    """Convert 0-based column index to Excel column letter(s). e.g. 0→A, 25→Z, 26→AA"""
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
                        # Restore semua data bulanan & forecast
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

if df is None and not uploaded:
    pass  # Tidak auto-load riwayat — user harus upload atau klik riwayat manual

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

st.markdown("""
<div style="margin:0 0 16px;padding:24px 28px;
  background:linear-gradient(135deg,#0a1628 0%,#0d1a0d 100%);
  border:1px solid #1e2d45;border-radius:16px;
  border-left:4px solid #3b82f6;">
  <div style="font-size:1.6rem;font-weight:800;
    background:linear-gradient(135deg,#60a5fa,#a78bfa);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    margin-bottom:4px;">
    📊 Dashboard Prediksi Klaim BPJS Ketenagakerjaan
  </div>
  <div style="font-size:.82rem;color:#475569;font-weight:500;">
    Analisis tren historis & proyeksi menggunakan metode statistik &amp; ML adaptif
  </div>
</div>""", unsafe_allow_html=True)

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

df_active_only = df[df['Kategori'].isin(active_progs)]

# Compute deltas if multi-year
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
    # Avg yearly growth
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

df_plot = df[df['Kategori'].isin(active_progs)].copy()

tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "🤖 ML Analysis", "🔮 Prediksi", "📥 Export"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    df_lat = df_plot[df_plot['Tahun'] == latest_year]

    # ── Insight Summary Row ───────────────────────────────────────────────
    if not single_yr:
        top_prog = df_lat.groupby('Kategori')['Kasus'].sum().idxmax()
        top_val  = int(df_lat.groupby('Kategori')['Kasus'].sum().max())
        # Fastest growing program
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
        # Center annotation
        total_kasus = int(pie_d['Kasus'].sum())
        fig.add_annotation(text=f"<b>{total_kasus:,}</b><br><span style='font-size:10px'>Total</span>",
            showarrow=False, font=dict(size=13, color='#e2e8f0'), align='center')
        st.plotly_chart(fig, width='stretch')

    with r2:
        st.markdown(f'<div class="sec">Market Share per Program — {latest_year}</div>',
                    unsafe_allow_html=True)
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

        st.markdown('<div class="sec">Tren Kasus per Tahun</div>', unsafe_allow_html=True)
        fig3 = go.Figure()
        for i, cat in enumerate(sorted(df_plot['Kategori'].unique())):
            cd = trend[trend['Kategori']==cat].sort_values('Tahun')
            col_c = COLORS[i % len(COLORS)]
            fig3.add_trace(go.Scatter(
                x=cd['Tahun'], y=cd['Kasus'],
                name=cat, mode='lines+markers',
                line=dict(color=col_c, width=2.5),
                marker=dict(size=9, color=col_c,
                    line=dict(color='#05090f', width=2)),
                fill='tozeroy', fillcolor=hex_to_rgba(col_c, 0.05),
                hovertemplate=f"<b>{cat}</b><br>Tahun: %{{x}}<br>Kasus: %{{y:,}}<extra></extra>"
            ))
        styled_chart(fig3, height=400)
        fig3.update_layout(xaxis=dict(dtick=1, showgrid=True, gridcolor='#0f1923'))
        st.plotly_chart(fig3, width='stretch')

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
            hm_p = (df_plot.groupby(['Kategori', 'Tahun'])['Kasus'].sum()
                    .reset_index()
                    .pivot(index='Kategori', columns='Tahun', values='Kasus')
                    .fillna(0))
            fig5 = px.imshow(hm_p, color_continuous_scale='Blues',
                             aspect='auto', text_auto=',')
            fig5.update_layout(**DARK, height=360, margin=dict(t=10,b=10,l=10,r=10))
            fig5.update_traces(textfont_size=11)
            st.plotly_chart(fig5, width='stretch')

        # YoY growth + CAGR
        st.markdown('<div class="sec">Year-over-Year Growth & CAGR per Program</div>',
                    unsafe_allow_html=True)
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

        # Box plot per program — spread/variability
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

        # Kasus vs Nominal correlation scatter
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
        with st.spinner(f"Melatih model untuk {len(active_progs)} program (Holt, SES, WMA + ML jika ≥8 tahun)..."):
            ml_res, err = run_ml(df, target_ml, n_lags, test_pct / 100)
        if err:
            st.error(f"Error: {err}"); ml_res = None
        else:
            results_cache[ck] = ml_res
            st.session_state.active_results = results_cache
            # Per-program analysis
            with st.spinner("Menganalisis model per program..."):
                # Per-program analysis is already in ml_res — no separate run needed
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

        # ── Sub-tabs ──────────────────────────────────────────────────────
        mtab1, mtab2, mtab3, mtab4 = st.tabs([
            "📊 Perbandingan Model", "🎯 Model per Program",
            "📝 Conclusion & Metrics", "🔮 Prophet + Kalender"
        ])

        # ── Sub-tab 1: Model per Program Summary ─────────────────────────
        with mtab1:
            st.markdown('<div class="sec">Model Terbaik per Program (Ringkasan)</div>',
                        unsafe_allow_html=True)
            if not bpp.empty:
                # Color scorecard per program
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
                    # MAPE per program — primary metric
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
                    # Model type distribution pie
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

            # Use bpp (best per program) for charts since results_df is now empty
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

            # Trend historis per program
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

                # Feature importance untuk tree/boosting models
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

        # ── Sub-tab 2: Per-Program Model Comparison ───────────────────────
        with mtab2:
            det = ml_res.get('detail', pd.DataFrame())
            if bpp.empty:
                st.info("Klik **Jalankan Analisis ML** untuk melihat model terbaik per program.")
            else:

                st.markdown('<div class="sec">Model Terbaik per Program</div>',
                            unsafe_allow_html=True)
                # Color-coded table
                st.dataframe(
                    bpp.style
                       .highlight_max(subset=['R2'], color='#14532d')
                       .highlight_min(subset=['MAPE (%)'], color='#14532d')
                       .format({'R2': '{:.4f}', 'MAPE (%)': '{:.2f}',
                                'MAE': '{:,.0f}', 'RMSE': '{:,.0f}'}),
                    width='stretch', height=260)

                # Heatmap: R² tiap model × program
                st.markdown('<div class="sec">Heatmap R² — Semua Model × Semua Program</div>',
                            unsafe_allow_html=True)
                heat = det.pivot_table(index='Model', columns='Program',
                                       values='R2', aggfunc='mean').fillna(0)
                fig_heat = px.imshow(heat, color_continuous_scale='Blues',
                                     aspect='auto', text_auto='.3f',
                                     title='R² per Model per Program (lebih biru = lebih baik)')
                fig_heat.update_layout(**DARK, height=400, margin=dict(t=50, b=20))
                st.plotly_chart(fig_heat, width='stretch')

                # Bar: best model per program
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

                # MAPE comparison
                st.markdown('<div class="sec">MAPE % Model Terbaik per Program</div>',
                            unsafe_allow_html=True)
                fig_mape = px.bar(bpp, x='Program', y='MAPE (%)', color='Model',
                                  color_discrete_sequence=COLORS,
                                  title='MAPE % per Program (lebih rendah = lebih baik)')
                fig_mape.add_hline(y=20, line_dash='dash', line_color='#fbbf24',
                                   annotation_text='Threshold 20%')
                fig_mape.update_layout(**DARK, height=380, margin=dict(t=50, b=40))
                st.plotly_chart(fig_mape, width='stretch')

                # Detailed table all models all programs
                with st.expander("📋 Tabel Detail Semua Model × Semua Program"):
                    st.dataframe(
                        det.sort_values(['Program','R2'], ascending=[True,False])
                           .style.format({'R2': '{:.4f}', 'MAPE (%)': '{:.2f}',
                                          'MAE': '{:,.0f}', 'RMSE': '{:,.0f}'}),
                        width='stretch', height=400)

        # ── Sub-tab 3: Conclusion & Metrics ───────────────────────────────
        with mtab3:
            conclusions = build_conclusion(ml_res, ml_res, df, target_ml, n_future)
            if conclusions:
                st.markdown('<div class="sec">Kesimpulan & Rekomendasi Otomatis</div>',
                            unsafe_allow_html=True)

                # Auto-insight banner
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
                    Gunakan <b>Tab Prediksi</b> untuk proyeksi tahunan dan <b>Prophet + Kalender</b> untuk analisis bulanan musiman.
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

                # Radar chart — MAPE-focused, reliable for small data
                st.markdown('<div class="sec">Radar Chart — Profil Kualitas per Program</div>',
                            unsafe_allow_html=True)
                if not bpp.empty and 'MAPE (%)' in bpp.columns:
                    rdf_r = bpp.dropna(subset=['MAPE (%)','MAE','RMSE']).copy()
                    # Normalize: 0=worst, 1=best
                    rdf_r['MAPE_n']  = (1 - (rdf_r['MAPE (%)'] / 100).clip(0, 1))
                    rdf_r['MAE_n']   = 1 - (rdf_r['MAE'] / (rdf_r['MAE'].max() + 1e-9))
                    rdf_r['RMSE_n']  = 1 - (rdf_r['RMSE'] / (rdf_r['RMSE'].max() + 1e-9))
                    # Stability: how close MAPE is to median (0=outlier, 1=on-par)
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
                    if n_yrs < 8:
                        st.markdown('<div class="info-box">ℹ️ R² dihilangkan dari radar chart karena tidak bermakna secara statistik dengan data < 8 tahun. Axis yang ditampilkan berbasis MAPE, MAE, RMSE, dan Stabilitas — semua lebih reliable untuk data kecil.</div>', unsafe_allow_html=True)
                else:
                    st.info("Data tidak cukup untuk radar chart.")

                # Summary table with grades
                st.markdown('<div class="sec">Scorecard per Program</div>',
                            unsafe_allow_html=True)

                if n_yrs < 8:
                    st.info(f"""ℹ️ **Catatan data kecil ({n_yrs} tahun):** R² tidak bermakna secara statistik
                    dengan data < 8 tahun — gunakan **MAPE** sebagai acuan utama akurasi prediksi.
                    MAPE mengukur rata-rata % error prediksi vs aktual. MAPE < 20% = layak pakai.""")

                def grade_mape(v):
                    if v is None or np.isnan(v): return "⚪ N/A"
                    return "🟢 Sangat Akurat (<10%)" if v<10 else "🔵 Akurat (10-20%)" if v<20 else "🟡 Cukup (20-50%)" if v<50 else "🔴 Tidak Akurat (>50%)"

                sc_df = bpp.copy() if not bpp.empty else pd.DataFrame()
                if not sc_df.empty:
                    sc_df['Grade MAPE'] = sc_df['MAPE (%)'].apply(grade_mape)
                    # Show/hide R2 based on data size
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
                Lalu push ke GitHub dan tunggu Streamlit Cloud redeploy.
                """)
            elif df_raw_m_p is None or len(df_raw_m_p) == 0:
                st.warning("Upload dataset dengan data bulanan terlebih dahulu untuk menggunakan Prophet.")
            else:
                n_holidays  = len(INDONESIAN_HOLIDAYS)
                n_htypes    = INDONESIAN_HOLIDAYS['holiday'].nunique() if n_holidays > 0 else 0
                gcal_status = (f"✅ <b>{n_holidays} hari libur</b>, <b>{n_htypes} jenis</b> berhasil dimuat dari Google Calendar API."
                               if n_holidays > 0 else
                               "⚠️ Google Calendar API belum merespons. Prophet tetap jalan tanpa holiday effect.")
                st.markdown(f"""<div class="info-box">
                🔮 <b>Prophet</b> dipilih karena lebih cocok dari SARIMA untuk data BPJS:<br>
                • Menangani <b>efek hari libur secara eksplisit</b> — Ramadhan (window 30 hari),
                  Idul Fitri (window 14 hari), Idul Adha (window 6 hari), Nyepi, Paskah, Natal, dll<br>
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
                    value=True)

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
                    # ── Combined forecast chart ────────────────────────────
                    tgt_label = target_prophet  # e.g. "Kasus" or "Nominal"
                    st.markdown(
                        f'<div class="sec">Forecast Prophet — Semua Program ({tgt_label})</div>',
                        unsafe_allow_html=True)

                    def hex_rgba(hex_c, alpha):
                        hex_c = hex_c.lstrip('#')
                        r,g,b = int(hex_c[0:2],16), int(hex_c[2:4],16), int(hex_c[4:6],16)
                        return f'rgba({r},{g},{b},{alpha})'

                    # Compute global y-min from actuals only (to clamp axis)
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

                        # Clamp forecast: never below 0, never below 20% of last actual
                        last_actual   = float(hist_df['y'].iloc[-1])
                        floor_val     = max(0.0, last_actual * 0.0)  # allow 0 floor
                        fc_future     = fc_df[fc_df['ds'] > cutoff].copy()
                        fc_future['yhat']       = fc_future['yhat'].clip(lower=floor_val)
                        fc_future['yhat_lower'] = fc_future['yhat_lower'].clip(lower=0)
                        fc_future['yhat_upper'] = fc_future['yhat_upper'].clip(lower=0)

                        # Actuals — area + line for weight
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

                        # CI band
                        fig_p_all.add_trace(go.Scatter(
                            x=list(fc_future['ds']) + list(fc_future['ds'][::-1]),
                            y=list(fc_future['yhat_upper']) + list(fc_future['yhat_lower'][::-1]),
                            fill='toself',
                            fillcolor=hex_rgba(col_c, 0.08),
                            line=dict(color='rgba(0,0,0,0)'),
                            legendgroup=cp, showlegend=False, hoverinfo='skip'))

                        # Prediction line — dashed
                        fig_p_all.add_trace(go.Scatter(
                            x=fc_future['ds'], y=fc_future['yhat'],
                            name=f'{cp} Prediksi',
                            mode='lines+markers',
                            legendgroup=cp,
                            line=dict(color=col_c, width=2, dash='dash'),
                            marker=dict(size=6, symbol='diamond'),
                            hovertemplate=f'<b>{cp} (Prediksi)</b><br>%{{x|%b %Y}}<br>{tgt_label}: %{{y:,.0f}}<extra></extra>'))

                    # Separator line
                    if last_hist_ds is not None:
                        fig_p_all.add_vline(
                            x=last_hist_ds.timestamp()*1000,
                            line_dash='dot', line_color='rgba(148,163,184,0.4)',
                            line_width=1.5,
                            annotation_text='← Aktual | Prediksi →',
                            annotation_font=dict(size=10, color='#94a3b8'),
                            annotation_position='top')

                    # Y-axis floor — never show negative
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
                            rangemode='nonnegative',   # ← kunci: tidak pernah negatif
                            range=[global_ymin, None]))
                    st.plotly_chart(fig_p_all, width='stretch')

                    # ── Per-program forecast table (tabs) ─────────────────
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
                            for col in ['Prediksi','Batas Bawah','Batas Atas']:
                                fc_fut[col] = fc_fut[col].apply(lambda x: f"{max(0,x):,.0f}")
                            st.dataframe(fc_fut, width='stretch', height=320)

                    # ── Holiday effects — dalam % perubahan klaim ─────────
                    st.markdown('<div class="sec">Efek Hari Libur per Program</div>',
                                unsafe_allow_html=True)

                    # ── Konversi efek Prophet ke % perubahan dari rata-rata ─
                    # Prophet additive: efek dalam satuan y (jumlah kasus)
                    # Prophet multiplicative: efek adalah proporsi (0.1 = +10%)
                    # Kita konversi keduanya ke % agar mudah dibaca
                    heff_all = []
                    holiday_names = list(INDONESIAN_HOLIDAYS['holiday'].unique()) if len(INDONESIAN_HOLIDAYS) > 0 else []
                    for cp, pr in all_p_results.items():
                        fc_rep   = pr['forecast']
                        hist_df  = pr['history']
                        avg_y    = float(hist_df['y'].mean()) if len(hist_df) > 0 else 1.0
                        s_mode   = pr.get('model').seasonality_mode if pr.get('model') else 'additive'

                        for hname in holiday_names:
                            if hname in fc_rep.columns:
                                raw_eff = float(fc_rep[hname].mean())
                                # Konversi ke % perubahan
                                if s_mode == 'multiplicative':
                                    pct = raw_eff * 100.0          # sudah proporsional
                                else:
                                    pct = (raw_eff / (avg_y + 1e-9)) * 100.0   # additive ÷ mean
                                heff_all.append({
                                    'Program'  : cp,
                                    'Hari Libur': hname,
                                    'Efek_raw' : raw_eff,
                                    'Efek_pct' : pct,           # dalam %
                                    'avg_y'    : avg_y,
                                })

                    if heff_all:
                        heff_df = pd.DataFrame(heff_all)

                        # Grouping semantik
                        def _cat_holiday(name):
                            nl = name.lower()
                            if 'idul fitri' in nl or 'lebaran' in nl or 'eid al-fitr' in nl: return 'Idul Fitri'
                            if 'idul adha' in nl or 'eid al-adha' in nl: return 'Idul Adha'
                            if 'ramad' in nl or 'puasa' in nl: return 'Ramadhan'
                            if 'natal' in nl or 'christmas' in nl: return 'Natal'
                            if 'tahun baru' in nl and 'islam' not in nl and 'imlek' not in nl and 'chinese' not in nl: return 'Tahun Baru'
                            if 'imlek' in nl or 'chinese new year' in nl or 'lunar' in nl: return 'Imlek'
                            if 'nyepi' in nl or 'silence' in nl or "bali's day" in nl: return 'Nyepi'
                            if 'isra' in nl or "mi'raj" in nl or 'miraj' in nl or 'ascension of the prophet' in nl: return "Isra Mi'raj"
                            if 'waisak' in nl or 'vesak' in nl or 'buddha' in nl: return 'Waisak'
                            if 'kenaikan' in nl or 'good friday' in nl or 'wafat' in nl or 'easter' in nl or 'paskah' in nl: return 'Paskah / Wafat'
                            if 'maulid' in nl or 'muhammad' in nl or 'nabi' in nl: return 'Maulid Nabi'
                            if 'muharram' in nl or 'tahun baru islam' in nl or 'islamic new year' in nl: return 'Tahun Baru Islam'
                            if 'buruh' in nl or 'labor' in nl or 'labour' in nl: return 'Hari Buruh'
                            if 'pancasila' in nl: return 'Hari Pancasila'
                            if 'kemerdekaan' in nl or 'independence' in nl or 'hut ri' in nl: return 'HUT RI'
                            if 'cuti bersama' in nl or 'joint holiday' in nl: return 'Cuti Bersama'
                            if 'election' in nl or 'pemilu' in nl: return 'Pemilu'
                            return None

                        heff_df['Kategori'] = heff_df['Hari Libur'].apply(_cat_holiday)
                        heff_df = heff_df[heff_df['Kategori'].notna()]

                        # Rata-rata % per program × kategori
                        heff_grp = (heff_df.groupby(['Program','Kategori'])
                                    .agg(Efek_pct=('Efek_pct','mean'),
                                         Efek_raw=('Efek_raw','mean'),
                                         avg_y=('avg_y','first'))
                                    .reset_index())

                        # Top-10 kategori paling berpengaruh
                        top_cats = (heff_grp.groupby('Kategori')['Efek_pct']
                                    .apply(lambda x: x.abs().mean())
                                    .nlargest(10).index.tolist())
                        heff_grp = heff_grp[heff_grp['Kategori'].isin(top_cats)].copy()

                        if len(heff_grp) == 0:
                            st.info("Tidak ada efek hari libur yang terdeteksi.")
                        else:
                            programs_list  = sorted(heff_grp['Program'].unique())
                            max_abs_pct    = heff_grp['Efek_pct'].abs().max()
                            has_meaningful = max_abs_pct > 0.1  # > 0.1% baru bermakna

                            if not has_meaningful:
                                st.markdown("""<div class="warn">
                                ⚠️ <b>Efek hari libur sangat kecil (&lt;0.1%) untuk semua program.</b>
                                Kemungkinan karena data bulanan belum cukup banyak (&lt;24 bulan).
                                Nilai tetap ditampilkan sebagai referensi.
                                </div>""", unsafe_allow_html=True)

                            cat_order = (heff_grp.groupby('Kategori')['Efek_pct']
                                         .apply(lambda x: x.abs().mean())
                                         .sort_values(ascending=False)
                                         .index.tolist())

                            # ── LOLLIPOP CHART (dalam %) ──────────────────
                            from plotly.subplots import make_subplots
                            n_prog = len(programs_list)
                            fig_lol = make_subplots(
                                rows=1, cols=n_prog,
                                subplot_titles=programs_list,
                                shared_yaxes=True,
                                horizontal_spacing=0.03,
                            )
                            x_max = max(1.0, max_abs_pct * 1.35)

                            for pi, prog in enumerate(programs_list):
                                pdata = (heff_grp[heff_grp['Program'] == prog]
                                         .set_index('Kategori')['Efek_pct'])

                                for ci, cat in enumerate(cat_order):
                                    v = float(pdata.get(cat, 0.0))
                                    bar_col = '#34d399' if v >= 0 else '#f87171'
                                    # Stem
                                    fig_lol.add_trace(go.Scatter(
                                        x=[0, v], y=[cat, cat],
                                        mode='lines',
                                        line=dict(color=bar_col, width=2.5),
                                        showlegend=False, hoverinfo='skip',
                                    ), row=1, col=pi+1)
                                    # Dot + label
                                    fig_lol.add_trace(go.Scatter(
                                        x=[v], y=[cat],
                                        mode='markers+text',
                                        marker=dict(color=bar_col, size=11,
                                                    line=dict(color='#05090f', width=1.5)),
                                        text=[f'{v:+.1f}%'],
                                        textposition='middle right' if v >= 0 else 'middle left',
                                        textfont=dict(size=9.5, color='#e2e8f0'),
                                        showlegend=False,
                                        hovertemplate=f'<b>{prog} — {cat}</b><br>Efek: <b>{v:+.2f}%</b> dari rata-rata klaim<extra></extra>',
                                    ), row=1, col=pi+1)

                                fig_lol.add_vline(x=0, line_color='rgba(255,255,255,0.2)',
                                                  line_width=1, row=1, col=pi+1)
                                fig_lol.update_xaxes(
                                    range=[-x_max, x_max],
                                    showgrid=True, gridcolor='#0f1923',
                                    zeroline=False,
                                    ticksuffix='%',
                                    tickfont=dict(size=9, color='#64748b'),
                                    row=1, col=pi+1)

                            fig_lol.update_yaxes(
                                categoryorder='array',
                                categoryarray=cat_order[::-1],
                                tickfont=dict(size=11, color='#e2e8f0'),
                                showgrid=True, gridcolor='#0f1923', col=1)

                            for ann in fig_lol.layout.annotations:
                                ann.font = dict(size=12, color='#93c5fd')

                            fig_lol.update_layout(
                                **DARK,
                                height=max(400, len(cat_order) * 46 + 120),
                                showlegend=False,
                                margin=dict(t=60, b=50, l=150, r=40),
                                title=dict(
                                    text='Efek Hari Libur terhadap Klaim (% dari rata-rata bulanan)',
                                    font=dict(size=14, color='#e2e8f0'), x=0),
                            )
                            st.plotly_chart(fig_lol, width='stretch')

                            # ── SCORECARD CARDS ────────────────────────────
                            st.markdown('<div class="sec">Ringkasan Efek per Program</div>',
                                        unsafe_allow_html=True)
                            card_cols = st.columns(n_prog)
                            for ci, prog in enumerate(programs_list):
                                pdata    = (heff_grp[heff_grp['Program'] == prog]
                                            .set_index('Kategori')['Efek_pct'])
                                avg_y_p  = float(heff_grp[heff_grp['Program']==prog]['avg_y'].iloc[0])
                                col_c    = COLORS[ci % len(COLORS)]
                                top_up   = pdata.sort_values(ascending=False).head(3)
                                top_down = pdata.sort_values(ascending=True).head(3)

                                def _pill(v):
                                    if abs(v) < 0.1:
                                        return f'<span style="color:#475569;font-size:.8rem">±0% <span style="font-size:.7rem">(tidak signifikan)</span></span>'
                                    elif v > 0:
                                        return f'<span style="color:#34d399;font-weight:700">{v:+.1f}%</span>'
                                    else:
                                        return f'<span style="color:#f87171;font-weight:700">{v:+.1f}%</span>'

                                # Contoh angka konkret: efek dalam jumlah kasus
                                def _konkret(v):
                                    delta_kasus = abs(v / 100.0 * avg_y_p)
                                    if delta_kasus < 0.5: return ''
                                    return f'<span style="color:#475569;font-size:.72rem"> (~{delta_kasus:,.0f} kasus)</span>'

                                up_html = ''.join(
                                    f'<div style="display:flex;justify-content:space-between;'
                                    f'align-items:center;margin:5px 0;font-size:.82rem;gap:4px;">'
                                    f'<span><span style="color:#34d399;margin-right:4px">▲</span>'
                                    f'<span style="color:#e2e8f0">{k}</span>{_konkret(v)}</span>'
                                    f'{_pill(v)}</div>'
                                    for k, v in top_up.items())

                                down_html = ''.join(
                                    f'<div style="display:flex;justify-content:space-between;'
                                    f'align-items:center;margin:5px 0;font-size:.82rem;gap:4px;">'
                                    f'<span><span style="color:#f87171;margin-right:4px">▼</span>'
                                    f'<span style="color:#e2e8f0">{k}</span>{_konkret(v)}</span>'
                                    f'{_pill(v)}</div>'
                                    for k, v in top_down.items())

                                net = float(pdata.sum())
                                if abs(net) < 0.5:
                                    badge = '<span style="background:#1e2d45;color:#94a3b8;padding:2px 8px;border-radius:6px;font-size:.72rem">Netral</span>'
                                elif net > 0:
                                    badge = f'<span style="background:#052e16;color:#34d399;padding:2px 8px;border-radius:6px;font-size:.72rem">Net +{net:.1f}%</span>'
                                else:
                                    badge = f'<span style="background:#450a0a;color:#f87171;padding:2px 8px;border-radius:6px;font-size:.72rem">Net {net:.1f}%</span>'

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
                                      <div style="font-size:.68rem;color:#475569;text-transform:uppercase;
                                      letter-spacing:1px;margin-bottom:6px;">📈 Paling Naik</div>
                                      {up_html}
                                      <div style="border-top:1px solid #1e2d45;margin:10px 0;"></div>
                                      <div style="font-size:.68rem;color:#475569;text-transform:uppercase;
                                      letter-spacing:1px;margin-bottom:6px;">📉 Paling Turun</div>
                                      {down_html}
                                    </div>''', unsafe_allow_html=True)

                            st.markdown("""<div class="info-box" style="margin-top:16px">
                            📊 <b>Satuan: % perubahan dari rata-rata klaim bulanan program tersebut.</b><br>
                            Contoh: <b>Ramadhan +15%</b> artinya klaim program itu rata-rata <b>15% lebih tinggi</b> dari biasanya saat bulan Ramadhan.
                            <b>Hari Buruh −8%</b> artinya klaim <b>8% lebih rendah</b> dari rata-rata saat Hari Buruh.
                            Angka dalam kurung (~X kasus) adalah estimasi selisih jumlah klaim dari rata-rata.
                            </div>""", unsafe_allow_html=True)
                    else:
                        st.info("Efek holiday akan muncul jika Prophet berhasil mendeteksi pola pada data bulanan.")





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

        # Simpan per target agar Kasus & Nominal bisa digabung di export
        st.session_state['last_forecast']                          = fut
        st.session_state['last_forecast_monthly']                  = fut_monthly
        st.session_state[f'forecast_{target_pred}']                = fut
        st.session_state[f'forecast_monthly_{target_pred}']        = fut_monthly
        st.session_state[f'forecast_annual_{target_pred}']         = fut   # tahunan per target

        # Update history dengan forecast terbaru
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

        # Show per-program model used
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

        # ════════════════════════════════════════════════════════════════════
        # PREDIKSI TAHUNAN
        # ════════════════════════════════════════════════════════════════════
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

            # Per-program model info cards
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

            # Waterfall chart: total per year
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

            st.markdown('<div class="sec">Tabel Prediksi Tahunan (Aktual per Tahun)</div>',
                        unsafe_allow_html=True)
            tbl = fut[['Kategori', 'Tahun', target_pred]].copy().sort_values(['Kategori', 'Tahun'])
            fmt = 'Rp {:,.0f}' if target_pred == 'Nominal' else '{:,.0f}'
            tbl[target_pred] = tbl[target_pred].apply(lambda x: fmt.format(x))
            st.dataframe(tbl, width='stretch')

        # ════════════════════════════════════════════════════════════════════
        # PREDIKSI BULANAN
        # ════════════════════════════════════════════════════════════════════
        with ptab_mo:
            if fut_monthly is not None and len(fut_monthly) > 0:
                st.markdown(
                    '<div class="info-box">'
                    'Prediksi bulanan dihitung dengan mendistribusikan total tahunan '
                    'menggunakan pola musiman (seasonal weights) dari data historis. '
                    'Nilai per bulan adalah <b>aktual per bulan</b> (bukan kumulatif).'
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

                st.markdown('<div class="sec">Tabel Prediksi Bulanan Lengkap (Aktual per Bulan)</div>',
                            unsafe_allow_html=True)
                tbl_mo = (fut_monthly[['Kategori', 'Tahun', 'Bulan', 'Periode', target_pred]]
                          .copy().sort_values(['Kategori', 'Tahun', 'Bulan']))
                fmt = 'Rp {:,.0f}' if target_pred == 'Nominal' else '{:,.0f}'
                tbl_mo[target_pred] = tbl_mo[target_pred].apply(lambda x: fmt.format(x))
                st.dataframe(tbl_mo, width='stretch', height=420)

            else:
                df_raw_debug = st.session_state.get('raw_monthly', None)
                if df_raw_debug is None:
                    st.warning(
                        "**Data bulanan belum tersimpan.** "
                        "Upload ulang file dataset Anda (hapus file lama di sidebar lalu upload lagi). "
                        "Pastikan nama file mengandung tahun, contoh: `data_2021.xlsx`."
                    )
                elif target_pred not in df_raw_debug.columns:
                    st.warning(
                        f"Kolom **{target_pred}** tidak ditemukan di data bulanan. "
                        f"Kolom tersedia: {list(df_raw_debug.columns)}"
                    )
                else:
                    st.warning(
                        "Terjadi kesalahan saat menghitung prediksi bulanan. "
                        "Coba klik **Hitung Prediksi** ulang."
                    )

    else:
        st.info("Klik **Hitung Prediksi** — model ML akan dilatih otomatis jika belum ada.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: EXPORT — Kasus & Nominal bulanan digabung dalam 1 sheet
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

    # fallback annual: jika belum ada per-target, pakai last_forecast
    if fut_ann_kasus is None and fut_ann_nominal is None and last_fut is not None:
        tc_lf = [c for c in last_fut.columns if c not in ['Kategori','Tahun','Type']]
        if tc_lf:
            if tc_lf[0] == 'Kasus':
                fut_ann_kasus = last_fut
            else:
                fut_ann_nominal = last_fut

    # fallback monthly: jika belum ada per-target, pakai last_forecast_monthly
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
        st.caption(
            "Sheet: Data Gabungan · Pivot Kasus · Prediksi Tahunan · "
            "**Prediksi Bulanan** (Kasus + Nominal + 2 chart) · Bulanan Detail · ML Results"
        )

        # Status data
        has_ann_k = fut_ann_kasus   is not None and len(fut_ann_kasus)   > 0
        has_ann_n = fut_ann_nominal is not None and len(fut_ann_nominal) > 0
        has_mo_kasus   = fut_mo_kasus   is not None and len(fut_mo_kasus)   > 0
        has_mo_nominal = fut_mo_nominal is not None and len(fut_mo_nominal) > 0

        status_html = '<div class="info-box">'
        status_html += '<b>Prediksi Tahunan:</b><br>'
        status_html += (f'✅ Kasus tahunan siap ({len(fut_ann_kasus)} baris)<br>'
                        if has_ann_k else '⚠️ Kasus tahunan belum ada<br>')
        status_html += (f'✅ Nominal tahunan siap ({len(fut_ann_nominal)} baris)<br>'
                        if has_ann_n else '⚠️ Nominal tahunan belum ada<br>')
        status_html += '<br><b>Prediksi Bulanan:</b><br>'
        status_html += (f'✅ Kasus bulanan siap ({len(fut_mo_kasus)} baris)<br>'
                        if has_mo_kasus else '⚠️ Kasus bulanan belum ada<br>')
        status_html += (f'✅ Nominal bulanan siap ({len(fut_mo_nominal)} baris)'
                        if has_mo_nominal else '⚠️ Nominal bulanan belum ada')
        status_html += '<br><br><i style="color:#64748b">Jalankan Prediksi untuk Kasus dan Nominal agar semua sheet terisi lengkap.</i>'
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
        st.caption("Data tahunan yang sudah diagregasi dari semua file yang diupload")
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
    st.info(f"**{len(df)} baris** | "
            f"**{len(active_progs)} program aktif** ({', '.join(active_progs)}) | "
            f"**Tahun: {', '.join(map(str, years))}**")
    st.dataframe(df, width='stretch', height=360)