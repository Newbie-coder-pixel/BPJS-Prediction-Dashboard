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
            if password == "bpjs2026":   # ← ganti password di sini
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Password salah!")
        st.stop()

check_password()

# === sisa kode app.py di bawah sini ===

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

# SARIMA
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.stattools import adfuller
    SARIMA_OK = True
except ImportError:
    SARIMA_OK = False

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

st.set_page_config(page_title="BPJS ML Dashboard", layout="wide", page_icon="📊")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
html,body,[class*="css"]{font-family:'Plus Jakarta Sans',sans-serif;}
.stApp{background:#080c14;color:#dde3f0;}
.kpi{background:linear-gradient(135deg,#111827,#0d1525);border:1px solid #1e2d45;
     border-radius:14px;padding:22px 18px;text-align:center;}
.kpi .val{font-size:1.9rem;font-weight:800;color:#60a5fa;
          font-family:'JetBrains Mono',monospace;line-height:1.1;}
.kpi .lbl{font-size:.72rem;color:#64748b;text-transform:uppercase;
          letter-spacing:1.2px;margin-top:6px;}
.badge{background:linear-gradient(90deg,#1e3a5f,#14532d);border:1px solid #3b82f640;
       border-radius:10px;padding:14px 20px;margin:10px 0;font-size:.95rem;}
.badge b{color:#93c5fd;}
.sec{font-size:.75rem;font-weight:700;color:#475569;text-transform:uppercase;
     letter-spacing:2px;margin:24px 0 10px;padding-bottom:8px;
     border-bottom:1px solid #1e2d45;}
.warn{background:#1c1500;border:1px solid #d9770640;border-radius:10px;
      padding:14px;color:#fbbf24;font-size:.88rem;margin:8px 0;}
.info-box{background:#0f1f35;border:1px solid #1e3a5f;border-radius:10px;
          padding:14px;font-size:.85rem;color:#93c5fd;margin:8px 0;line-height:1.8;}
.tag-add{display:inline-block;background:#14532d;color:#86efac;
         border-radius:5px;padding:2px 8px;font-size:.78rem;margin:2px;}
.tag-rem{display:inline-block;background:#450a0a;color:#fca5a5;
         border-radius:5px;padding:2px 8px;font-size:.78rem;margin:2px;}
.tag-stable{display:inline-block;background:#1e3a5f;color:#93c5fd;
            border-radius:5px;padding:2px 8px;font-size:.78rem;margin:2px;}
</style>
""", unsafe_allow_html=True)

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

DARK = dict(template='plotly_dark', paper_bgcolor='#080c14', plot_bgcolor='#080c14',
            font_color='#dde3f0', font_family='Plus Jakarta Sans')
COLORS = ['#60a5fa','#34d399','#f87171','#fbbf24','#a78bfa',
          '#f472b6','#38bdf8','#fb923c','#4ade80','#e879f9']

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
# ML CORE — UPGRADED with XGBoost, LightGBM, ElasticNet, ExtraTrees, TimeSeriesCV
# ══════════════════════════════════════════════════════════════════════════════

def build_features(series, n_lags, cat_id=0.0):
    """Build rich lag + rolling features from a 1D time series."""
    pad = list(series)
    while len(pad) <= n_lags:
        pad.insert(0, pad[0])
    X_all, y_all = [], []
    for i in range(n_lags, len(pad)):
        win3 = pad[max(0, i-3):i]
        win6 = pad[max(0, i-6):i]
        lags = [pad[i - l] for l in range(1, n_lags + 1)]
        trend = (pad[i-1] - pad[i-2]) if i >= 2 else 0.0
        trend2= (pad[i-1] - pad[i-3]) if i >= 3 else 0.0
        feat = lags + [
            np.mean(win3), np.std(win3) if len(win3)>1 else 0.0,
            np.mean(win6), np.std(win6) if len(win6)>1 else 0.0,
            trend, trend2,
            np.min(win6) if win6 else 0.0,
            np.max(win6) if win6 else 0.0,
            cat_id
        ]
        X_all.append(feat)
        y_all.append(pad[i])
    return np.array(X_all), np.array(y_all)


def get_models(n_train):
    """Return all available models dict."""
    k = min(3, max(1, n_train - 1))
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge':             Ridge(alpha=1.0),
        'Lasso':             Lasso(alpha=0.1, max_iter=10000),
        'ElasticNet':        ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000),
        'Huber':             HuberRegressor(max_iter=500),
        'Decision Tree':     DecisionTreeRegressor(max_depth=4, random_state=42),
        'Random Forest':     RandomForestRegressor(n_estimators=200, max_depth=6,
                                                    min_samples_leaf=1, random_state=42),
        'Extra Trees':       ExtraTreesRegressor(n_estimators=200, max_depth=6, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                                        learning_rate=0.05, random_state=42),
        'SVR':               SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1),
        'KNN':               KNeighborsRegressor(n_neighbors=k, weights='distance'),
    }
    if XGBOOST_OK:
        models['XGBoost'] = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                          subsample=0.8, colsample_bytree=0.8,
                                          random_state=42, verbosity=0)
    if LGBM_OK:
        models['LightGBM'] = LGBMRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                             subsample=0.8, random_state=42, verbose=-1)
    return models


SCALED_MODELS = {'SVR', 'KNN', 'Ridge', 'Lasso', 'ElasticNet', 'Huber', 'Linear Regression'}


def score_model(yt, yp):
    """Return metrics dict for a prediction."""
    mae  = float(mean_absolute_error(yt, yp))
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    r2   = float(r2_score(yt, yp)) if len(yt) > 1 else 0.0
    mape = float(np.mean(np.abs((yt - yp) / (np.abs(yt) + 1e-9))) * 100)
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE (%)': mape}


def run_ml(df, target, n_lags, test_ratio):
    active    = get_active_programs(df)
    df_active = df[df['Kategori'].isin(active)].copy()
    years     = sorted(df_active['Tahun'].unique())
    cat_enc   = {c: float(i) for i, c in enumerate(active)}
    single    = len(years) == 1

    if single:
        rows_s = []
        for cat in active:
            v = df_active[df_active['Kategori']==cat][target].dropna().values
            if not len(v): continue
            val = float(v[0])
            rows_s.append([float(cat_enc[cat]), val, np.log1p(val)])
        if not rows_s: return None, "Tidak ada data valid."
        X = np.array(rows_s); y = X[:, 1].copy()
        X_train = X_test = X; y_train = y_test = y
    else:
        all_X, all_y = [], []
        for cat in active:
            sub = (df_active[df_active['Kategori']==cat]
                   .sort_values('Tahun')[target].dropna().values.astype(float))
            if not len(sub): continue
            Xc, yc = build_features(sub, n_lags, cat_enc[cat])
            all_X.append(Xc); all_y.append(yc)
        if not all_X: return None, "Tidak cukup data."
        X = np.vstack(all_X); y = np.concatenate(all_y)
        split = max(1, int(len(X) * test_ratio)) if len(X) >= 4 else 0
        if split == 0:
            X_train=X_test=X; y_train=y_test=y
        else:
            X_train,X_test = X[:-split], X[-split:]
            y_train,y_test = y[:-split], y[-split:]

    scaler = RobustScaler()
    Xtr_s  = scaler.fit_transform(X_train)
    Xte_s  = scaler.transform(X_test)

    models   = get_models(len(y_train))
    results, preds, fitted = [], {}, {}
    for name, mdl in models.items():
        try:
            Xtr = Xtr_s if name in SCALED_MODELS else X_train
            Xte = Xte_s if name in SCALED_MODELS else X_test
            mdl.fit(Xtr, y_train)
            yp = mdl.predict(Xte)
            sc = score_model(y_test, yp)
            results.append({'Model': name, **sc})
            preds[name]  = yp
            fitted[name] = mdl
        except:
            pass

    if not results: return None, "Semua model gagal."
    rdf = pd.DataFrame(results)
    # Composite score: R2 higher better, MAPE lower better
    rdf['_score'] = rdf['R2'].clip(-1,1)*0.5 - (rdf['MAPE (%)']/100).clip(0,2)*0.3 - (rdf['RMSE']/(rdf['RMSE'].max()+1e-9))*0.2
    rdf = rdf.sort_values('_score', ascending=False).reset_index(drop=True)
    best = rdf.iloc[0]['Model']
    rdf  = rdf.drop(columns=['_score'])

    return {
        'results_df': rdf, 'best_name': best,
        'fitted': fitted, 'scaler': scaler,
        'preds': preds, 'y_test': y_test,
        'cat_enc': cat_enc, 'single': single,
        'n_lags': n_lags, 'target': target,
        'active_programs': active,
        'X_train': X_train, 'X_test': X_test,
    }, None


# ══════════════════════════════════════════════════════════════════════════════
# PER-PROGRAM MODEL COMPARISON — with proper leave-one-out CV for small data
# ══════════════════════════════════════════════════════════════════════════════

def run_ml_per_program(df, target, n_lags, test_ratio):
    active = get_active_programs(df)
    rows   = []

    for cat in active:
        sub = (df[df['Kategori']==cat]
               .sort_values('Tahun')[target].dropna().values.astype(float))
        if len(sub) < 2:
            continue

        Xc, yc = build_features(sub, n_lags, 0.0)
        if len(Xc) < 2:
            continue

        # Use leave-one-out if too few samples, else time-split
        n = len(Xc)
        if n <= 4:
            # LOO cross-validation
            all_yt, all_yp = [], []
            for leave in range(n):
                idx_tr = [i for i in range(n) if i != leave]
                if not idx_tr: continue
                Xtr, ytr = Xc[idx_tr], yc[idx_tr]
                Xte, yte = Xc[[leave]], yc[[leave]]
                sc_pp = RobustScaler().fit(Xtr)
                Xtr_s, Xte_s = sc_pp.transform(Xtr), sc_pp.transform(Xte)
                models = get_models(len(ytr))
                for mname, mdl in models.items():
                    try:
                        mdl.fit(Xtr_s if mname in SCALED_MODELS else Xtr, ytr)
                        yp = mdl.predict(Xte_s if mname in SCALED_MODELS else Xte)
                        rows.append({'Program': cat, 'Model': mname,
                                     'R2': float(r2_score([yte[0]],[yp[0]])) if False else None,
                                     '_yt': float(yte[0]), '_yp': float(yp[0])})
                    except:
                        pass
        else:
            split = max(1, int(n * test_ratio))
            Xtr, Xte = Xc[:-split], Xc[-split:]
            ytr, yte = yc[:-split], yc[-split:]
            sc_pp = RobustScaler().fit(Xtr)
            Xtr_s, Xte_s = sc_pp.transform(Xtr), sc_pp.transform(Xte)
            models = get_models(len(ytr))
            for mname, mdl in models.items():
                try:
                    mdl.fit(Xtr_s if mname in SCALED_MODELS else Xtr, ytr)
                    yp  = mdl.predict(Xte_s if mname in SCALED_MODELS else Xte)
                    sc  = score_model(yte, yp)
                    rows.append({'Program': cat, 'Model': mname, **sc, '_yt': None, '_yp': None})
                except:
                    pass

    if not rows:
        return None

    # For LOO rows, aggregate predictions per program+model then score
    res_rows = []
    raw_df   = pd.DataFrame(rows)

    for (cat, mname), grp in raw_df.groupby(['Program','Model']):
        if grp['_yt'].notna().all() and grp['_yp'].notna().all():
            yt_arr = grp['_yt'].values
            yp_arr = grp['_yp'].values
            sc = score_model(yt_arr, yp_arr)
        else:
            sc = {'R2': grp.get('R2', pd.Series([np.nan])).mean(),
                  'MAPE (%)': grp.get('MAPE (%)', pd.Series([np.nan])).mean(),
                  'MAE': grp.get('MAE', pd.Series([np.nan])).mean(),
                  'RMSE': grp.get('RMSE', pd.Series([np.nan])).mean()}
        res_rows.append({'Program': cat, 'Model': mname, **sc})

    res_df = pd.DataFrame(res_rows)
    # Fill NaN R2 with -999 for sorting, then restore
    res_df['_sort_r2'] = res_df['R2'].fillna(-999)
    best_per_prog = (res_df.sort_values('_sort_r2', ascending=False)
                     .groupby('Program').first().reset_index()
                     [['Program','Model','R2','MAPE (%)','MAE','RMSE']])
    res_df = res_df.drop(columns=['_sort_r2'])

    return {'detail': res_df, 'best_per_program': best_per_prog}


# ══════════════════════════════════════════════════════════════════════════════
# CONCLUSION METRICS
# ══════════════════════════════════════════════════════════════════════════════

def build_conclusion(ml_result, per_prog_result, df, target, n_future):
    lines = []
    if ml_result is None:
        return lines
    rdf  = ml_result['results_df']
    best = ml_result['best_name']
    br   = rdf[rdf['Model'] == best].iloc[0]
    r2_val, mape_val = br['R2'], br['MAPE (%)']
    r2_grade   = ("Sangat Baik (>0.9)" if r2_val > 0.9 else
                  "Baik (0.8–0.9)"     if r2_val > 0.8 else
                  "Cukup (0.6–0.8)"    if r2_val > 0.6 else "Lemah (<0.6)")
    mape_grade = ("Sangat Akurat (<10%)" if mape_val < 10 else
                  "Akurat (10–20%)"      if mape_val < 20 else
                  "Cukup (20–50%)"       if mape_val < 50 else "Tidak Akurat (>50%)")
    lines.append(('🏆', 'Model Terbaik Global',
        f"**{best}** dipilih sebagai model terbaik dengan R² = **{r2_val:.4f}** ({r2_grade}) "
        f"dan MAPE = **{mape_val:.2f}%** ({mape_grade})."))
    if per_prog_result is not None:
        bpp = per_prog_result['best_per_program']
        prog_str = ', '.join(f"{r['Program']} → **{r['Model']}**" for _, r in bpp.iterrows())
        lines.append(('📊', 'Model Terbaik per Program', prog_str))
        worst  = bpp.sort_values('R2').iloc[0]
        best_p = bpp.sort_values('R2', ascending=False).iloc[0]
        lines.append(('🔍', 'Akurasi per Program',
            f"Program **{best_p['Program']}** paling mudah diprediksi (R²={best_p['R2']:.3f}). "
            f"Program **{worst['Program']}** paling sulit (R²={worst['R2']:.3f}) — "
            "perlu data lebih banyak atau fitur eksternal."))
    base_yr = int(df['Tahun'].max())
    lines.append(('📅', 'Horizon Prediksi',
        f"Model dilatih pada data s/d **{base_yr}** dan mampu memproyeksikan hingga "
        f"**{base_yr + n_future}** ({n_future} tahun ke depan). "
        "Akurasi menurun semakin jauh horizon waktu."))
    yrs = sorted(df['Tahun'].unique())
    lines.append(('📁', 'Kualitas Data',
        f"Dataset mencakup **{len(yrs)} tahun** ({yrs[0]}–{yrs[-1]}) "
        f"dengan **{len(get_active_programs(df))} program aktif**. "
        + ("Jumlah tahun cukup untuk ML multi-lag." if len(yrs) >= 4
           else "Tambah data historis untuk meningkatkan akurasi.")))
    if r2_val >= 0.8 and mape_val <= 20:
        rec = "✅ Model layak digunakan untuk perencanaan anggaran dan proyeksi klaim."
    elif r2_val >= 0.6:
        rec = "⚠️ Model cukup untuk proyeksi kasar, namun perlu validasi tambahan."
    else:
        rec = "❌ Model belum cukup akurat. Tambah data historis atau fitur tambahan."
    lines.append(('💡', 'Rekomendasi', rec))
    return lines


# ══════════════════════════════════════════════════════════════════════════════
# PROPHET — with Indonesian Islamic calendar
# ══════════════════════════════════════════════════════════════════════════════

INDONESIAN_HOLIDAYS = pd.DataFrame([
    {'holiday':'Idul Fitri', 'ds':'2021-05-13','lower_window':-7,'upper_window':7},
    {'holiday':'Idul Fitri', 'ds':'2022-05-02','lower_window':-7,'upper_window':7},
    {'holiday':'Idul Fitri', 'ds':'2023-04-22','lower_window':-7,'upper_window':7},
    {'holiday':'Idul Fitri', 'ds':'2024-04-10','lower_window':-7,'upper_window':7},
    {'holiday':'Idul Fitri', 'ds':'2025-03-31','lower_window':-7,'upper_window':7},
    {'holiday':'Idul Fitri', 'ds':'2026-03-20','lower_window':-7,'upper_window':7},
    {'holiday':'Idul Adha',  'ds':'2021-07-20','lower_window':-3,'upper_window':3},
    {'holiday':'Idul Adha',  'ds':'2022-07-09','lower_window':-3,'upper_window':3},
    {'holiday':'Idul Adha',  'ds':'2023-06-29','lower_window':-3,'upper_window':3},
    {'holiday':'Idul Adha',  'ds':'2024-06-17','lower_window':-3,'upper_window':3},
    {'holiday':'Idul Adha',  'ds':'2025-06-07','lower_window':-3,'upper_window':3},
    {'holiday':'Idul Adha',  'ds':'2026-05-27','lower_window':-3,'upper_window':3},
    {'holiday':'Ramadhan',   'ds':'2021-04-13','lower_window':0, 'upper_window':30},
    {'holiday':'Ramadhan',   'ds':'2022-04-02','lower_window':0, 'upper_window':30},
    {'holiday':'Ramadhan',   'ds':'2023-03-23','lower_window':0, 'upper_window':30},
    {'holiday':'Ramadhan',   'ds':'2024-03-11','lower_window':0, 'upper_window':30},
    {'holiday':'Ramadhan',   'ds':'2025-03-01','lower_window':0, 'upper_window':30},
    {'holiday':'Natal',      'ds':'2021-12-25','lower_window':-1,'upper_window':1},
    {'holiday':'Natal',      'ds':'2022-12-25','lower_window':-1,'upper_window':1},
    {'holiday':'Natal',      'ds':'2023-12-25','lower_window':-1,'upper_window':1},
    {'holiday':'Natal',      'ds':'2024-12-25','lower_window':-1,'upper_window':1},
    {'holiday':'Tahun Baru', 'ds':'2021-01-01','lower_window':-1,'upper_window':1},
    {'holiday':'Tahun Baru', 'ds':'2022-01-01','lower_window':-1,'upper_window':1},
    {'holiday':'Tahun Baru', 'ds':'2023-01-01','lower_window':-1,'upper_window':1},
    {'holiday':'Tahun Baru', 'ds':'2024-01-01','lower_window':-1,'upper_window':1},
    {'holiday':'Tahun Baru', 'ds':'2025-01-01','lower_window':-1,'upper_window':1},
    {'holiday':'Tahun Baru', 'ds':'2026-01-01','lower_window':-1,'upper_window':1},
])
INDONESIAN_HOLIDAYS['ds'] = pd.to_datetime(INDONESIAN_HOLIDAYS['ds'])




def run_sarima(df_monthly_raw, target, cat, n_months):
    """Run SARIMA(p,d,q)(P,D,Q,12) on monthly data for a single program."""
    if not SARIMA_OK:
        return None, "statsmodels tidak terinstall."
    cat_df = df_monthly_raw[df_monthly_raw['Kategori'] == cat].copy()
    if len(cat_df) < 12:
        return None, f"Data {cat} kurang dari 12 bulan untuk SARIMA."
    cat_df = cat_df.sort_values(['Tahun','Bulan'])
    cat_df['ds'] = pd.to_datetime(
        cat_df['Tahun'].astype(str) + '-' + cat_df['Bulan'].astype(str).str.zfill(2) + '-01')
    ts = cat_df.groupby('ds')[target].sum().sort_index()
    ts = ts[ts > 0]
    if len(ts) < 12:
        return None, "Data terlalu sedikit setelah filtering."

    # Auto-select order: try common configurations
    best_aic = np.inf
    best_cfg = (1,1,1,0,1,1)
    configs = [
        (1,1,1,0,1,1), (1,1,0,0,1,1), (0,1,1,0,1,1),
        (2,1,1,0,1,1), (1,1,2,0,1,1), (1,0,1,0,1,1),
    ]
    best_model = None
    for p,d,q,P,D,Q in configs:
        try:
            m = SARIMAX(ts, order=(p,d,q), seasonal_order=(P,D,Q,12),
                        enforce_stationarity=False, enforce_invertibility=False)
            res = m.fit(disp=False, maxiter=200)
            if res.aic < best_aic:
                best_aic = res.aic
                best_cfg = (p,d,q,P,D,Q)
                best_model = res
        except:
            pass

    if best_model is None:
        return None, "Semua konfigurasi SARIMA gagal."

    try:
        forecast_obj = best_model.get_forecast(steps=n_months)
        fc_mean = forecast_obj.predicted_mean
        fc_ci   = forecast_obj.conf_int(alpha=0.05)
        last_date = ts.index[-1]
        future_idx = pd.date_range(
            start=last_date + pd.DateOffset(months=1), periods=n_months, freq='MS')
        fc_df = pd.DataFrame({
            'ds': future_idx,
            'yhat': fc_mean.values,
            'yhat_lower': fc_ci.iloc[:,0].values,
            'yhat_upper': fc_ci.iloc[:,1].values,
        })
        fc_df['yhat']       = fc_df['yhat'].clip(0)
        fc_df['yhat_lower'] = fc_df['yhat_lower'].clip(0)
        fc_df['yhat_upper'] = fc_df['yhat_upper'].clip(0)

        # In-sample fit metrics
        fitted_vals = best_model.fittedvalues
        common_idx  = ts.index.intersection(fitted_vals.index)
        yt_is = ts[common_idx].values
        yp_is = fitted_vals[common_idx].values
        mape_is = float(np.mean(np.abs((yt_is - yp_is)/(np.abs(yt_is)+1e-9)))*100)
        r2_is   = float(r2_score(yt_is, yp_is)) if len(yt_is)>1 else 0.0

        return {
            'forecast': fc_df,
            'history':  ts.reset_index().rename(columns={target:'y','ds':'ds'}) if target in ts.name else
                        ts.reset_index().rename(columns={ts.name:'y'}),
            'config':   best_cfg, 'aic': best_aic,
            'mape_insample': mape_is, 'r2_insample': r2_is,
            'fitted_values': fitted_vals, 'ts': ts,
        }, None
    except Exception as e:
        return None, str(e)


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
    try:
        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            holidays=INDONESIAN_HOLIDAYS if use_holidays else None,
            seasonality_mode='multiplicative',
            interval_width=0.95,
        )
        m.fit(cat_df)
        future = m.make_future_dataframe(periods=n_months, freq='MS')
        fc = m.predict(future)
        return {'model': m, 'forecast': fc, 'history': cat_df}, None
    except Exception as e:
        return None, str(e)


def forecast(df, ml, n_years):
    target  = ml['target']
    best    = ml['best_name']
    mdl     = ml['fitted'][best]
    sc      = ml['scaler']
    enc     = ml['cat_enc']
    nlags   = ml['n_lags']
    single  = ml['single']
    active  = ml['active_programs']
    scaled  = {'SVR', 'KNN', 'Ridge', 'Lasso'}
    base_yr = int(df['Tahun'].max())
    rows    = []

    for cat in active:
        cat_id  = float(enc.get(cat, 0))
        history = list(df[df['Kategori'] == cat]
                       .sort_values('Tahun')[target]
                       .dropna().values.astype(float))
        if not history: continue

        for fy in range(1, n_years + 1):
            if single:
                pred = history[0] * (1.05 ** fy)
            else:
                pad  = history[:]
                while len(pad) <= nlags:
                    pad.insert(0, pad[0])
                win  = pad[-3:]
                lags = [pad[-l] for l in range(1, nlags + 1)]
                feat = np.array(lags + [
                    np.mean(win),
                    np.std(win) if len(win) > 1 else 0.0,
                    pad[-1] - pad[-2] if len(pad) >= 2 else 0.0,
                    cat_id
                ]).reshape(1, -1)
                if best in scaled: feat = sc.transform(feat)
                pred = float(mdl.predict(feat)[0])

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
    st.markdown("## 📊 BPJS ML")
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
    <div style="text-align:center;padding:100px 0">
      <div style="font-size:5rem">📂</div>
      <h2 style="color:#60a5fa">Upload Dataset BPJS Ketenagakerjaan</h2>
      <p style="color:#64748b;max-width:560px;margin:auto">
        Upload 1 atau lebih file CSV/Excel.<br>
        Nama file <b>tidak harus mengandung tahun</b> (contoh: <code>BPJS_jaminan.xlsx</code>).<br><br>
        Struktur kolom yang didukung:<br>
        <code>KODE_DETIL | DATE | PROGRAM | NOMINAL | KASUS</code>
      </p>
    </div>""", unsafe_allow_html=True)
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

st.markdown("## 📊 Dashboard Prediksi Klaim BPJS Ketenagakerjaan")

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

c1, c2, c3, c4 = st.columns(4)
df_active_only = df[df['Kategori'].isin(active_progs)]

with c1:
    st.markdown(f'<div class="kpi"><div class="val">{len(years)}</div>'
                f'<div class="lbl">Tahun Data</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="kpi"><div class="val">{len(active_progs)}</div>'
                f'<div class="lbl">Program Aktif</div></div>', unsafe_allow_html=True)
with c3:
    tk = int(df_active_only['Kasus'].sum())
    st.markdown(f'<div class="kpi"><div class="val">{tk:,}</div>'
                f'<div class="lbl">Total Kasus</div></div>', unsafe_allow_html=True)
with c4:
    if has_nom:
        tn = df_active_only['Nominal'].sum() / 1e9
        st.markdown(f'<div class="kpi"><div class="val">{tn:,.1f}T</div>'
                    f'<div class="lbl">Total Nominal (Rp)</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="kpi"><div class="val">{latest_year}</div>'
                    f'<div class="lbl">Tahun Terbaru</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

df_plot = df[df['Kategori'].isin(active_progs)].copy()

tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "🤖 ML Analysis", "🔮 Prediksi", "📥 Export"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    df_lat = df_plot[df_plot['Tahun'] == latest_year]

    r1, r2 = st.columns(2)
    with r1:
        st.markdown('<div class="sec">Distribusi Kasus — Semua Tahun (Program Aktif)</div>',
                    unsafe_allow_html=True)
        pie_d = df_plot.groupby('Kategori')['Kasus'].sum().reset_index()
        fig = px.pie(pie_d, names='Kategori', values='Kasus', hole=0.42,
                     color_discrete_sequence=COLORS)
        fig.update_traces(textinfo='label+percent', textposition='outside')
        fig.update_layout(**DARK, showlegend=False, height=400,
                          margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, width='stretch')

    with r2:
        st.markdown(f'<div class="sec">Kasus per Program — Tahun {latest_year}</div>',
                    unsafe_allow_html=True)
        bar_d = (df_lat.groupby('Kategori')['Kasus'].sum()
                 .sort_values(ascending=True).reset_index())
        fig2 = px.bar(bar_d, x='Kasus', y='Kategori', orientation='h',
                      color='Kasus', color_continuous_scale='Blues', text='Kasus')
        fig2.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
        fig2.update_layout(**DARK, height=400, showlegend=False,
                           coloraxis_showscale=False,
                           margin=dict(t=10, b=10, l=10, r=90))
        st.plotly_chart(fig2, width='stretch')

    if not single_yr:
        st.markdown('<div class="sec">Tren Kasus per Tahun</div>', unsafe_allow_html=True)
        trend = df_plot.groupby(['Tahun', 'Kategori'])['Kasus'].sum().reset_index()
        fig3 = px.line(trend, x='Tahun', y='Kasus', color='Kategori',
                       markers=True, color_discrete_sequence=COLORS)
        fig3.update_layout(**DARK, height=380,
                           legend=dict(orientation='h', y=-0.25),
                           margin=dict(b=90, t=10))
        st.plotly_chart(fig3, width='stretch')

        st.markdown('<div class="sec">Komposisi Stacked per Tahun</div>', unsafe_allow_html=True)
        fig4 = px.bar(trend, x='Tahun', y='Kasus', color='Kategori',
                      barmode='stack', color_discrete_sequence=COLORS)
        fig4.update_layout(**DARK, height=370,
                           legend=dict(orientation='h', y=-0.25),
                           margin=dict(b=90, t=10))
        st.plotly_chart(fig4, width='stretch')

        st.markdown('<div class="sec">Heatmap Kasus (Program × Tahun)</div>',
                    unsafe_allow_html=True)
        hm_p = (df_plot.groupby(['Kategori', 'Tahun'])['Kasus'].sum()
                .reset_index()
                .pivot(index='Kategori', columns='Tahun', values='Kasus')
                .fillna(0))
        fig5 = px.imshow(hm_p, color_continuous_scale='Blues',
                         aspect='auto', text_auto=True)
        fig5.update_layout(**DARK,
                           height=max(280, len(active_progs) * 48 + 80),
                           margin=dict(t=10, b=10))
        st.plotly_chart(fig5, width='stretch')

        st.markdown('<div class="sec">Year-over-Year Growth Kasus (%)</div>',
                    unsafe_allow_html=True)
        yoy = []
        for cat in active_progs:
            cd = df_plot[df_plot['Kategori'] == cat].sort_values('Tahun')
            for i in range(1, len(cd)):
                prev = cd.iloc[i-1]['Kasus']
                curr = cd.iloc[i]['Kasus']
                yoy.append({
                    'Kategori': cat,
                    'Tahun': int(cd.iloc[i]['Tahun']),
                    'Growth (%)': round((curr / (prev + 1e-9) - 1) * 100, 2)
                })
        if yoy:
            ydf = pd.DataFrame(yoy)
            fig_y = px.bar(ydf, x='Tahun', y='Growth (%)', color='Kategori',
                           barmode='group', color_discrete_sequence=COLORS)
            fig_y.add_hline(y=0, line_color='#475569')
            fig_y.update_layout(**DARK, height=360,
                                legend=dict(orientation='h', y=-0.25),
                                margin=dict(b=80, t=10))
            st.plotly_chart(fig_y, width='stretch')

    if has_nom:
        nc1, nc2 = st.columns(2)
        with nc1:
            st.markdown('<div class="sec">Distribusi Nominal</div>', unsafe_allow_html=True)
            np_d = df_plot.groupby('Kategori')['Nominal'].sum().reset_index()
            fp = px.pie(np_d, names='Kategori', values='Nominal', hole=0.42,
                        color_discrete_sequence=COLORS)
            fp.update_traces(textinfo='label+percent', textposition='outside')
            fp.update_layout(**DARK, showlegend=False, height=340,
                             margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fp, width='stretch')
        with nc2:
            if not single_yr:
                st.markdown('<div class="sec">Tren Nominal (Rp Miliar)</div>',
                            unsafe_allow_html=True)
                nt = df_plot.groupby(['Tahun', 'Kategori'])['Nominal'].sum().reset_index()
                nt['Nominal_B'] = nt['Nominal'] / 1e9
                fn = px.area(nt, x='Tahun', y='Nominal_B', color='Kategori',
                             color_discrete_sequence=COLORS)
                fn.update_layout(**DARK, height=340,
                                 legend=dict(orientation='h', y=-0.3),
                                 margin=dict(b=80, t=10))
                st.plotly_chart(fn, width='stretch')
            else:
                st.markdown('<div class="sec">Nominal per Program (Rp Miliar)</div>',
                            unsafe_allow_html=True)
                nb = df_plot.groupby('Kategori')['Nominal'].sum().reset_index()
                nb['Nominal_B'] = nb['Nominal'] / 1e9
                fn = px.bar(nb.sort_values('Nominal_B', ascending=True),
                            x='Nominal_B', y='Kategori', orientation='h',
                            color='Nominal_B', color_continuous_scale='Greens',
                            text='Nominal_B')
                fn.update_traces(texttemplate='%{text:,.1f}B', textposition='outside')
                fn.update_layout(**DARK, height=340, coloraxis_showscale=False,
                                 margin=dict(t=10, b=10, l=10, r=90))
                st.plotly_chart(fn, width='stretch')

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
        with st.spinner(f"Melatih 8 model ML untuk {len(active_progs)} program aktif..."):
            ml_res, err = run_ml(df, target_ml, n_lags, test_pct / 100)
        if err:
            st.error(f"Error: {err}"); ml_res = None
        else:
            results_cache[ck] = ml_res
            st.session_state.active_results = results_cache
            # Per-program analysis
            with st.spinner("Menganalisis model per program..."):
                pp_res = run_ml_per_program(df, target_ml, n_lags, test_pct / 100)
                st.session_state[f'per_prog_{target_ml}'] = pp_res
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
        rdf      = ml_res['results_df']
        best     = ml_res['best_name']
        best_row = rdf[rdf['Model'] == best].iloc[0]
        pp_res   = st.session_state.get(f'per_prog_{target_ml}', None)

        mode_note = '&nbsp;|&nbsp; ⚠️ Mode 1 Tahun' if single_yr else ''
        st.markdown(
            f'<div class="badge">'
            f'🏆 <b>Model Terbaik (Auto-Selected):</b> {best}'
            f'&nbsp;|&nbsp; R² = <b>{best_row["R2"]:.4f}</b>'
            f'&nbsp;|&nbsp; MAPE = <b>{best_row["MAPE (%)"]:.2f}%</b>'
            f'&nbsp;|&nbsp; MAE = <b>{best_row["MAE"]:,.0f}</b>'
            f'&nbsp;|&nbsp; Dilatih pada <b>{len(active_progs)} program aktif</b>'
            f'{mode_note}</div>',
            unsafe_allow_html=True)

        # ── Sub-tabs ──────────────────────────────────────────────────────
        mtab1, mtab2, mtab3, mtab4, mtab5 = st.tabs([
            "📊 Perbandingan Model", "🎯 Model per Program",
            "📝 Conclusion & Metrics", "🔮 Prophet + Kalender", "📈 SARIMA"
        ])

        # ── Sub-tab 1: Model Comparison ───────────────────────────────────
        with mtab1:
            st.markdown('<div class="sec">Perbandingan Semua Model</div>',
                        unsafe_allow_html=True)
            st.dataframe(
                rdf.style
                   .highlight_max(subset=['R2'], color='#1e3a5f')
                   .highlight_min(subset=['MAE', 'RMSE', 'MAPE (%)'], color='#14532d')
                   .format({'MAE': '{:,.0f}', 'RMSE': '{:,.0f}',
                            'R2': '{:.4f}', 'MAPE (%)': '{:.2f}'}),
                width='stretch', height=320)

            mc1, mc2 = st.columns(2)
            with mc1:
                fig_r2 = px.bar(rdf, x='Model', y='R2', color='R2',
                                color_continuous_scale='Blues',
                                title='R² Score (lebih tinggi = lebih baik)')
                fig_r2.add_hline(y=0.8, line_dash='dash', line_color='#34d399',
                                 annotation_text='Target 0.8')
                fig_r2.update_layout(**DARK, height=360, xaxis_tickangle=-30,
                                     coloraxis_showscale=False, margin=dict(b=90, t=40))
                st.plotly_chart(fig_r2, width='stretch')
            with mc2:
                fig_mp = px.bar(rdf, x='Model', y='MAPE (%)', color='MAPE (%)',
                                color_continuous_scale='Reds_r',
                                title='MAPE % (lebih rendah = lebih baik)')
                fig_mp.add_hline(y=20, line_dash='dash', line_color='#34d399',
                                 annotation_text='Threshold 20%')
                fig_mp.update_layout(**DARK, height=360, xaxis_tickangle=-30,
                                     coloraxis_showscale=False, margin=dict(b=90, t=40))
                st.plotly_chart(fig_mp, width='stretch')

            if not single_yr and len(ml_res['y_test']) > 0:
                st.markdown(f'<div class="sec">Aktual vs Prediksi — {best}</div>',
                            unsafe_allow_html=True)
                yt = ml_res['y_test']
                yp = ml_res['preds'].get(best, yt)
                n  = min(len(yt), len(yp))
                fav = go.Figure()
                fav.add_trace(go.Scatter(y=yt[:n], name='Aktual',
                                         line=dict(color='#60a5fa', width=2.5),
                                         mode='lines+markers'))
                fav.add_trace(go.Scatter(y=yp[:n], name='Prediksi',
                                         line=dict(color='#34d399', width=2.5, dash='dash'),
                                         mode='lines+markers'))
                fav.update_layout(**DARK, height=360, hovermode='x unified',
                                  legend=dict(orientation='h'), margin=dict(t=20, b=40))
                st.plotly_chart(fav, width='stretch')

                res_plot = yt[:n] - yp[:n]
                fig_res = px.histogram(x=res_plot, nbins=min(20, n),
                                       title='Distribusi Residual',
                                       color_discrete_sequence=['#60a5fa'])
                fig_res.update_layout(**DARK, height=280, margin=dict(t=40, b=20))
                st.plotly_chart(fig_res, width='stretch')

            if best in ('Random Forest', 'Gradient Boosting', 'Decision Tree'):
                m_obj  = ml_res['fitted'][best]
                nf     = ml_res['X_train'].shape[1]
                lnames = [f'Lag_{i}' for i in range(1, n_lags + 1)]
                extras = ['MA3', 'Std3', 'Trend', 'cat_id']
                fnames = (lnames + extras)[:nf]
                while len(fnames) < nf:
                    fnames.append(f'feat_{len(fnames)}')
                fi = pd.DataFrame({'Feature': fnames,
                                   'Importance': m_obj.feature_importances_})\
                       .sort_values('Importance', ascending=False)
                fig_fi = px.bar(fi, x='Importance', y='Feature', orientation='h',
                                color='Importance', color_continuous_scale='Viridis',
                                title='Feature Importance')
                fig_fi.update_layout(**DARK, height=max(300, nf * 32 + 80),
                                     coloraxis_showscale=False, margin=dict(l=100, t=40))
                st.plotly_chart(fig_fi, width='stretch')

        # ── Sub-tab 2: Per-Program Model Comparison ───────────────────────
        with mtab2:
            if pp_res is None:
                st.info("Klik **Jalankan Analisis ML** untuk melihat model terbaik per program.")
            else:
                bpp = pp_res['best_per_program']
                det = pp_res['detail']

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
            conclusions = build_conclusion(ml_res, pp_res, df, target_ml, n_future)
            if conclusions:
                st.markdown('<div class="sec">Kesimpulan Otomatis Analisis ML</div>',
                            unsafe_allow_html=True)
                for icon, title, text in conclusions:
                    st.markdown(f"""
                    <div style="background:#0f1f35;border:1px solid #1e3a5f;border-radius:12px;
                                padding:16px 20px;margin:10px 0;">
                        <div style="font-size:.7rem;font-weight:700;color:#475569;
                                    text-transform:uppercase;letter-spacing:1.5px;
                                    margin-bottom:6px;">{icon} {title}</div>
                        <div style="color:#dde3f0;font-size:.92rem;line-height:1.7;">{text}</div>
                    </div>""", unsafe_allow_html=True)

                # Radar chart all models
                st.markdown('<div class="sec">Radar Chart — Profil Kualitas Model</div>',
                            unsafe_allow_html=True)
                # Normalize metrics for radar
                rdf_r = ml_res['results_df'].copy()
                rdf_r['R2_n']      = rdf_r['R2'].clip(0, 1)
                rdf_r['MAPE_n']    = (1 - (rdf_r['MAPE (%)'] / 100).clip(0, 1))
                rdf_r['MAE_n']     = 1 - (rdf_r['MAE'] / (rdf_r['MAE'].max() + 1e-9))
                rdf_r['RMSE_n']    = 1 - (rdf_r['RMSE'] / (rdf_r['RMSE'].max() + 1e-9))
                cats_radar = ['R² Score', 'Akurasi (1-MAPE)', 'Presisi (1-MAE)', 'Konsistensi (1-RMSE)']
                fig_radar = go.Figure()
                for i, row in rdf_r.iterrows():
                    vals = [row['R2_n'], row['MAPE_n'], row['MAE_n'], row['RMSE_n']]
                    vals += [vals[0]]
                    cats_r = cats_radar + [cats_radar[0]]
                    fig_radar.add_trace(go.Scatterpolar(
                        r=vals, theta=cats_r, fill='toself',
                        name=row['Model'], opacity=0.7,
                        line=dict(color=COLORS[i % len(COLORS)])
                    ))
                fig_radar.update_layout(
                    **DARK, height=500,
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 1],
                                        gridcolor='#1e2d45', tickfont=dict(color='#64748b')),
                        angularaxis=dict(gridcolor='#1e2d45'),
                        bgcolor='#080c14',
                    ),
                    legend=dict(orientation='h', y=-0.15),
                    margin=dict(t=30, b=60)
                )
                st.plotly_chart(fig_radar, width='stretch')

                # Summary table with grades
                st.markdown('<div class="sec">Scorecard Semua Model</div>',
                            unsafe_allow_html=True)
                def grade_r2(v):
                    return "🟢 Sangat Baik" if v>0.9 else "🔵 Baik" if v>0.8 else "🟡 Cukup" if v>0.6 else "🔴 Lemah"
                def grade_mape(v):
                    return "🟢 <10%" if v<10 else "🔵 10-20%" if v<20 else "🟡 20-50%" if v<50 else "🔴 >50%"
                sc_df = ml_res['results_df'].copy()
                sc_df['Grade R²']   = sc_df['R2'].apply(grade_r2)
                sc_df['Grade MAPE'] = sc_df['MAPE (%)'].apply(grade_mape)
                sc_df['Best?']      = sc_df['Model'].apply(lambda m: '🏆' if m == best else '')
                st.dataframe(
                    sc_df[['Model','R2','MAPE (%)','MAE','RMSE','Grade R²','Grade MAPE','Best?']]
                    .style.format({'R2': '{:.4f}', 'MAPE (%)': '{:.2f}',
                                   'MAE': '{:,.0f}', 'RMSE': '{:,.0f}'}),
                    width='stretch', height=320)

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
                st.markdown("""<div class="info-box">
                🔮 <b>Prophet</b> adalah model time-series dari Meta yang mampu mendeteksi
                <b>pola musiman</b> (tahunan, bulanan) dan <b>efek hari libur</b> seperti
                Ramadhan, Idul Fitri, dan Idul Adha pada pola klaim BPJS.
                </div>""", unsafe_allow_html=True)

                pc1, pc2, pc3 = st.columns(3)
                with pc1:
                    target_prophet = st.selectbox("Target", targets, key='prophet_target')
                with pc2:
                    cat_prophet = st.selectbox("Program", active_progs, key='prophet_cat')
                with pc3:
                    n_months_prophet = st.slider("Prediksi (bulan)", 6, 36, 12, 6)

                use_holidays = st.checkbox("Gunakan Kalender Islam Indonesia (Ramadhan, Idul Fitri, dll)", value=True)

                if st.button("🔮 Jalankan Prophet", type="primary", width='stretch'):
                    with st.spinner(f"Melatih Prophet untuk {cat_prophet} — {target_prophet}..."):
                        p_result, p_err = run_prophet(
                            df_raw_m_p, target_prophet, cat_prophet,
                            n_months_prophet, use_holidays
                        )
                    if p_err:
                        st.error(f"Prophet Error: {p_err}")
                    else:
                        st.session_state['prophet_result'] = p_result
                        st.session_state['prophet_meta'] = {
                            'target': target_prophet, 'cat': cat_prophet,
                            'use_holidays': use_holidays
                        }

                p_result = st.session_state.get('prophet_result', None)
                p_meta   = st.session_state.get('prophet_meta', {})

                if p_result and p_meta.get('cat') == cat_prophet and p_meta.get('target') == target_prophet:
                    fc_df   = p_result['forecast']
                    hist_df = p_result['history']
                    m_obj   = p_result['model']

                    # ── Main forecast plot ────────────────────────────────
                    st.markdown(f'<div class="sec">Forecast Prophet — {cat_prophet} ({target_prophet})</div>',
                                unsafe_allow_html=True)
                    fig_p = go.Figure()
                    # History
                    fig_p.add_trace(go.Scatter(
                        x=hist_df['ds'], y=hist_df['y'],
                        name='Aktual', mode='lines+markers',
                        line=dict(color='#60a5fa', width=2.5),
                        marker=dict(size=6)
                    ))
                    # Forecast
                    fc_future = fc_df[fc_df['ds'] > hist_df['ds'].max()]
                    fig_p.add_trace(go.Scatter(
                        x=fc_future['ds'], y=fc_future['yhat'],
                        name='Prediksi Prophet', mode='lines+markers',
                        line=dict(color='#34d399', width=2.5, dash='dash'),
                        marker=dict(size=7, symbol='diamond')
                    ))
                    # Confidence interval
                    fig_p.add_trace(go.Scatter(
                        x=pd.concat([fc_future['ds'], fc_future['ds'][::-1]]),
                        y=pd.concat([fc_future['yhat_upper'], fc_future['yhat_lower'][::-1]]),
                        fill='toself', fillcolor='rgba(52,211,153,0.1)',
                        line=dict(color='rgba(0,0,0,0)'),
                        name='Interval 95%', showlegend=True
                    ))
                    # Holiday markers
                    if use_holidays:
                        for _, hrow in INDONESIAN_HOLIDAYS.iterrows():
                            if hist_df['ds'].min() <= hrow['ds'] <= fc_future['ds'].max():
                                fig_p.add_vline(
                                    x=hrow['ds'].timestamp() * 1000,
                                    line_dash='dot', line_color='#fbbf24',
                                    line_width=1, opacity=0.6,
                                    annotation_text=hrow['holiday'],
                                    annotation_font=dict(size=9, color='#fbbf24'),
                                    annotation_position='top left'
                                )
                    fig_p.update_layout(
                        **DARK, height=500, hovermode='x unified',
                        legend=dict(orientation='h', y=-0.2),
                        margin=dict(b=80, t=20, l=70, r=20),
                        xaxis_title='Periode', yaxis_title=target_prophet
                    )
                    st.plotly_chart(fig_p, width='stretch')

                    # ── Components plot ───────────────────────────────────
                    st.markdown('<div class="sec">Komponen Pola (Trend + Seasonality + Holiday)</div>',
                                unsafe_allow_html=True)
                    comp_cols = st.columns(2)
                    # Trend
                    with comp_cols[0]:
                        fig_tr = go.Figure()
                        fig_tr.add_trace(go.Scatter(
                            x=fc_df['ds'], y=fc_df['trend'],
                            mode='lines', line=dict(color='#60a5fa', width=2),
                            name='Trend'
                        ))
                        fig_tr.update_layout(**DARK, height=280, title='Trend',
                                             margin=dict(t=40, b=20))
                        st.plotly_chart(fig_tr, width='stretch')
                    # Yearly seasonality
                    with comp_cols[1]:
                        if 'yearly' in fc_df.columns:
                            fig_yr = go.Figure()
                            yr_data = fc_df[['ds','yearly']].copy()
                            yr_data['month'] = yr_data['ds'].dt.month
                            yr_avg = yr_data.groupby('month')['yearly'].mean().reset_index()
                            yr_avg['Bulan'] = yr_avg['month'].apply(
                                lambda m: ['Jan','Feb','Mar','Apr','Mei','Jun',
                                           'Jul','Agt','Sep','Okt','Nov','Des'][m-1])
                            fig_yr.add_trace(go.Bar(
                                x=yr_avg['Bulan'], y=yr_avg['yearly'],
                                marker_color=COLORS[1], name='Efek Musiman'
                            ))
                            fig_yr.update_layout(**DARK, height=280,
                                                 title='Pola Musiman Tahunan',
                                                 margin=dict(t=40, b=20))
                            st.plotly_chart(fig_yr, width='stretch')

                    # Holiday effects
                    if use_holidays:
                        st.markdown('<div class="sec">Efek Hari Libur pada Klaim</div>',
                                    unsafe_allow_html=True)
                        holiday_cols = [c for c in fc_df.columns
                                        if c in ['Idul Fitri','Idul Adha','Ramadhan',
                                                 'Natal','Tahun Baru']]
                        if holiday_cols:
                            heff_rows = []
                            for hc in holiday_cols:
                                heff_rows.append({'Hari Libur': hc,
                                                  'Efek Rata-rata': fc_df[hc].mean(),
                                                  'Efek Max': fc_df[hc].max(),
                                                  'Efek Min': fc_df[hc].min()})
                            heff_df = pd.DataFrame(heff_rows).sort_values('Efek Rata-rata', ascending=False)
                            fig_heff = px.bar(heff_df, x='Hari Libur', y='Efek Rata-rata',
                                              color='Efek Rata-rata',
                                              color_continuous_scale='RdYlGn',
                                              title='Dampak Hari Libur terhadap Volume Klaim',
                                              text='Efek Rata-rata')
                            fig_heff.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
                            fig_heff.update_layout(**DARK, height=360, coloraxis_showscale=False,
                                                   margin=dict(t=50, b=40))
                            st.plotly_chart(fig_heff, width='stretch')
                            st.markdown("""<div class="info-box">
                            <b>Interpretasi Efek Holiday:</b><br>
                            • <b>Positif</b> = klaim cenderung <b>naik</b> saat periode tersebut<br>
                            • <b>Negatif</b> = klaim cenderung <b>turun</b> (misal: libur panjang, kantor tutup)<br>
                            • Ramadhan sering menunjukkan penurunan JKK (kecelakaan kerja) karena aktivitas lebih sedikit
                            </div>""", unsafe_allow_html=True)
                        else:
                            st.info("Efek holiday akan muncul setelah Prophet selesai dilatih dengan data yang cukup.")

                    # Prophet forecast table
                    st.markdown('<div class="sec">Tabel Prediksi Prophet</div>',
                                unsafe_allow_html=True)
                    fc_show = fc_future[['ds','yhat','yhat_lower','yhat_upper']].copy()
                    fc_show.columns = ['Periode','Prediksi','Batas Bawah (95%)','Batas Atas (95%)']
                    fc_show['Periode'] = fc_show['Periode'].dt.strftime('%Y-%m')
                    for col in ['Prediksi','Batas Bawah (95%)','Batas Atas (95%)']:
                        fc_show[col] = fc_show[col].apply(lambda x: f"{max(0,x):,.0f}")
                    st.dataframe(fc_show, width='stretch', height=380)


        # ── Sub-tab 5: SARIMA ─────────────────────────────────────────────
        with mtab5:
            df_raw_m_s = st.session_state.get('raw_monthly', None)
            if not SARIMA_OK:
                st.warning("statsmodels tidak terinstall. Jalankan: `pip install statsmodels`")
            elif df_raw_m_s is None or len(df_raw_m_s) == 0:
                st.warning("Upload dataset bulanan terlebih dahulu untuk menggunakan SARIMA.")
            else:
                st.markdown("""<div class="info-box">
                📈 <b>SARIMA</b> (Seasonal AutoRegressive Integrated Moving Average) adalah model
                statistik klasik yang sangat cocok untuk <b>data time-series bulanan</b>.
                SARIMA dapat menangkap pola musiman (seasonal) seperti lonjakan klaim di awal/akhir tahun,
                dan lebih <b>interpretable</b> dibanding ML black-box.
                Order otomatis dipilih berdasarkan AIC terkecil dari beberapa konfigurasi.
                </div>""", unsafe_allow_html=True)

                sc1, sc2, sc3 = st.columns(3)
                with sc1:
                    target_sarima = st.selectbox("Target", targets, key='sarima_target')
                with sc2:
                    cat_sarima = st.selectbox("Program", active_progs, key='sarima_cat')
                with sc3:
                    n_months_sarima = st.slider("Prediksi (bulan)", 6, 36, 12, 6, key='sarima_months')

                if st.button("📈 Jalankan SARIMA", type="primary", width='stretch'):
                    with st.spinner(f"Melatih SARIMA untuk {cat_sarima} — {target_sarima}..."):
                        s_result, s_err = run_sarima(
                            df_raw_m_s, target_sarima, cat_sarima, n_months_sarima)
                    if s_err:
                        st.error(f"SARIMA Error: {s_err}")
                    else:
                        st.session_state['sarima_result'] = s_result
                        st.session_state['sarima_meta'] = {
                            'target': target_sarima, 'cat': cat_sarima}

                s_result = st.session_state.get('sarima_result', None)
                s_meta   = st.session_state.get('sarima_meta', {})

                if s_result and s_meta.get('cat') == cat_sarima and s_meta.get('target') == target_sarima:
                    ts       = s_result['ts']
                    fc_df_s  = s_result['forecast']
                    cfg      = s_result['config']
                    p,d,q,P,D,Q = cfg

                    # KPI row
                    ki1, ki2, ki3, ki4 = st.columns(4)
                    with ki1:
                        st.markdown(f'''<div class="kpi"><div class="val">SARIMA</div>
                        <div class="lbl">({p},{d},{q})({P},{D},{Q},12)</div></div>''', unsafe_allow_html=True)
                    with ki2:
                        st.markdown(f'''<div class="kpi"><div class="val">{s_result["aic"]:.1f}</div>
                        <div class="lbl">AIC Score</div></div>''', unsafe_allow_html=True)
                    with ki3:
                        st.markdown(f'''<div class="kpi"><div class="val">{s_result["r2_insample"]:.4f}</div>
                        <div class="lbl">R² In-Sample</div></div>''', unsafe_allow_html=True)
                    with ki4:
                        st.markdown(f'''<div class="kpi"><div class="val">{s_result["mape_insample"]:.2f}%</div>
                        <div class="lbl">MAPE In-Sample</div></div>''', unsafe_allow_html=True)

                    # ── Forecast chart ─────────────────────────────────────
                    st.markdown(f'<div class="sec">Forecast SARIMA — {cat_sarima} ({target_sarima})</div>',
                                unsafe_allow_html=True)
                    fig_s = go.Figure()
                    # History
                    fig_s.add_trace(go.Scatter(
                        x=ts.index, y=ts.values,
                        name='Aktual', mode='lines+markers',
                        line=dict(color='#60a5fa', width=2.5), marker=dict(size=6)
                    ))
                    # In-sample fitted
                    fv = s_result['fitted_values']
                    fig_s.add_trace(go.Scatter(
                        x=fv.index, y=fv.values.clip(0),
                        name='Fitted (In-sample)', mode='lines',
                        line=dict(color='#a78bfa', width=1.5, dash='dot'), opacity=0.8
                    ))
                    # Forecast
                    fig_s.add_trace(go.Scatter(
                        x=fc_df_s['ds'], y=fc_df_s['yhat'],
                        name='Prediksi SARIMA', mode='lines+markers',
                        line=dict(color='#34d399', width=2.5, dash='dash'),
                        marker=dict(size=7, symbol='diamond')
                    ))
                    # CI band
                    fig_s.add_trace(go.Scatter(
                        x=pd.concat([fc_df_s['ds'], fc_df_s['ds'][::-1]]),
                        y=pd.concat([fc_df_s['yhat_upper'], fc_df_s['yhat_lower'][::-1]]),
                        fill='toself', fillcolor='rgba(52,211,153,0.12)',
                        line=dict(color='rgba(0,0,0,0)'), name='Interval 95%'
                    ))
                    # Holiday markers
                    for _, hrow in INDONESIAN_HOLIDAYS.iterrows():
                        if ts.index.min() <= hrow['ds'] <= fc_df_s['ds'].max():
                            fig_s.add_vline(
                                x=hrow['ds'].timestamp()*1000,
                                line_dash='dot', line_color='#fbbf24',
                                line_width=1, opacity=0.5,
                                annotation_text=hrow['holiday'],
                                annotation_font=dict(size=8, color='#fbbf24'),
                                annotation_position='top left'
                            )
                    fig_s.update_layout(
                        **DARK, height=500, hovermode='x unified',
                        legend=dict(orientation='h', y=-0.2),
                        margin=dict(b=80, t=20, l=70, r=20),
                        xaxis_title='Periode', yaxis_title=target_sarima
                    )
                    st.plotly_chart(fig_s, width='stretch')

                    # ── Residual analysis ──────────────────────────────────
                    st.markdown('<div class="sec">Analisis Residual SARIMA</div>',
                                unsafe_allow_html=True)
                    resid = ts.values - s_result['fitted_values'][ts.index.intersection(s_result['fitted_values'].index)].values
                    rc1, rc2 = st.columns(2)
                    with rc1:
                        fig_res_s = px.histogram(x=resid, nbins=20,
                                                  title='Distribusi Residual',
                                                  color_discrete_sequence=['#60a5fa'])
                        fig_res_s.update_layout(**DARK, height=280, margin=dict(t=40,b=20))
                        st.plotly_chart(fig_res_s, width='stretch')
                    with rc2:
                        fig_res_line = go.Figure()
                        fig_res_line.add_trace(go.Scatter(
                            x=list(range(len(resid))), y=resid,
                            mode='lines+markers', name='Residual',
                            line=dict(color='#f87171', width=1.5)
                        ))
                        fig_res_line.add_hline(y=0, line_dash='dash', line_color='#64748b')
                        fig_res_line.update_layout(**DARK, height=280, title='Residual vs Time',
                                                   margin=dict(t=40,b=20))
                        st.plotly_chart(fig_res_line, width='stretch')

                    # ── Forecast table ─────────────────────────────────────
                    st.markdown('<div class="sec">Tabel Prediksi SARIMA</div>',
                                unsafe_allow_html=True)
                    fc_show_s = fc_df_s[['ds','yhat','yhat_lower','yhat_upper']].copy()
                    fc_show_s.columns = ['Periode','Prediksi','Batas Bawah (95%)','Batas Atas (95%)']
                    fc_show_s['Periode'] = fc_show_s['Periode'].dt.strftime('%Y-%m')
                    for col in ['Prediksi','Batas Bawah (95%)','Batas Atas (95%)']:
                        fc_show_s[col] = fc_show_s[col].apply(lambda x: f"{max(0,x):,.0f}")
                    st.dataframe(fc_show_s, width='stretch', height=380)

                    # ── SARIMA vs Prophet comparison (if both available) ───
                    if st.session_state.get('prophet_result') and st.session_state.get('prophet_meta',{}).get('cat') == cat_sarima:
                        p_res_cmp = st.session_state['prophet_result']
                        st.markdown('<div class="sec">Perbandingan SARIMA vs Prophet</div>',
                                    unsafe_allow_html=True)
                        fig_cmp = go.Figure()
                        fig_cmp.add_trace(go.Scatter(x=ts.index, y=ts.values,
                            name='Aktual', mode='lines+markers',
                            line=dict(color='#60a5fa', width=2)))
                        fig_cmp.add_trace(go.Scatter(x=fc_df_s['ds'], y=fc_df_s['yhat'],
                            name='SARIMA', mode='lines',
                            line=dict(color='#34d399', width=2, dash='dash')))
                        p_fc_cmp = p_res_cmp['forecast']
                        p_future_cmp = p_fc_cmp[p_fc_cmp['ds'] > ts.index[-1]]
                        fig_cmp.add_trace(go.Scatter(x=p_future_cmp['ds'], y=p_future_cmp['yhat'],
                            name='Prophet', mode='lines',
                            line=dict(color='#f59e0b', width=2, dash='dot')))
                        fig_cmp.update_layout(**DARK, height=420, hovermode='x unified',
                                             legend=dict(orientation='h',y=-0.2),
                                             margin=dict(b=80,t=20))
                        st.plotly_chart(fig_cmp, width='stretch')


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
        best_used  = ml_pred['best_name']
        prog_list  = ", ".join(ml_pred['active_programs'])
        mode_note  = " | Mode 1 dataset (estimasi 5%/thn)" if single_yr else ""

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
                fillcolor='rgba(96,165,250,0.07)',
                line_width=0,
                annotation_text="Zona Prediksi",
                annotation_position="top left",
                annotation_font=dict(color='#60a5fa', size=11),
            )
            fig_main.add_vline(
                x=latest_year + 0.5,
                line_dash='dot', line_color='#60a5fa', line_width=1.5,
            )

            fig_main.update_layout(
                **DARK, height=520,
                hovermode='x unified',
                legend=dict(
                    orientation='h', y=-0.28,
                    groupclick='toggleitem',
                    font=dict(size=11),
                ),
                margin=dict(b=130, t=20, l=70, r=20),
                yaxis_title=target_pred,
                xaxis_title='Tahun',
                xaxis=dict(dtick=1),
            )
            st.plotly_chart(fig_main, width='stretch')

            st.markdown('<div class="sec">Nilai Prediksi per Tahun per Program</div>',
                        unsafe_allow_html=True)
            fig_bar = px.bar(
                fut, x='Tahun', y=target_pred, color='Kategori',
                barmode='group', color_discrete_sequence=COLORS,
                text=target_pred,
            )
            fig_bar.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
            fig_bar.update_layout(**DARK, height=400,
                                  legend=dict(orientation='h', y=-0.25),
                                  margin=dict(b=80, t=10),
                                  xaxis=dict(dtick=1))
            st.plotly_chart(fig_bar, width='stretch')

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