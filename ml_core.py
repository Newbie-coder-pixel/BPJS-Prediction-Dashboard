"""
ml_core.py — Model ML, statistik, Prophet, dan fungsi forecast.
"""
import re
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from auth import _secret
from data_utils import get_active_programs

# ── Optional heavy deps ───────────────────────────────────────────────────────
try:
    from xgboost import XGBRegressor
    XGBOOST_OK = True
except ImportError:
    XGBOOST_OK = False

try:
    from lightgbm import LGBMRegressor
    LGBM_OK = True
except ImportError:
    LGBM_OK = False

try:
    from prophet import Prophet
    PROPHET_OK = True
except ImportError:
    try:
        from fbprophet import Prophet  # type: ignore[import]
        PROPHET_OK = True
    except ImportError:
        PROPHET_OK = False

# ── Holidays ──────────────────────────────────────────────────────────────────
GCAL_ID  = "en.indonesian%23holiday%40group.v.calendar.google.com"
GCAL_KEY = _secret("GCAL_KEY", "")

_WINDOW_RULES = {
    'idul fitri': (-7, 7), 'lebaran': (-7, 7),
    'ramadan': (0, 29), 'ramadhan': (0, 29), 'puasa': (0, 29),
    'idul adha': (-3, 3), 'natal': (-2, 2), 'christmas': (-2, 2),
    'tahun baru': (-1, 2), 'new year': (-1, 2),
    'cuti bersama': (-1, 1), 'default': (-1, 1),
}

def _get_window(name: str):
    nl = name.lower()
    for keyword, (lo, hi) in _WINDOW_RULES.items():
        if keyword != 'default' and keyword in nl:
            return lo, hi
    return _WINDOW_RULES['default']


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_google_holidays(year_start: int = 2019, year_end: int = 2028) -> list:
    import urllib.request, urllib.parse, json
    if not GCAL_KEY:
        return []
    base_url  = f"https://www.googleapis.com/calendar/v3/calendars/{GCAL_ID}/events"
    all_rows  = []
    seen_keys = set()
    for year in range(year_start, year_end + 1):
        page_token = None
        while True:
            params = {
                'key': GCAL_KEY,
                'timeMin': f'{year}-01-01T00:00:00Z',
                'timeMax': f'{year}-12-31T23:59:59Z',
                'maxResults': '2500',
                'singleEvents': 'true',
                'orderBy': 'startTime',
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
                all_rows.append({'ds': ds, 'holiday': name, 'lower_window': lo, 'upper_window': hi})
            page_token = data.get('nextPageToken')
            if not page_token:
                break
    return all_rows


@st.cache_data(ttl=86400, show_spinner=False)
def build_holiday_df() -> pd.DataFrame:
    rows = fetch_google_holidays()
    if not rows:
        return pd.DataFrame(columns=['ds', 'holiday', 'lower_window', 'upper_window'])
    df_h = pd.DataFrame(rows)
    df_h['ds'] = pd.to_datetime(df_h['ds'])
    df_h = df_h.drop_duplicates(subset=['ds','holiday']).sort_values('ds').reset_index(drop=True)
    return df_h


INDONESIAN_HOLIDAYS = build_holiday_df()


def _sanitize_prophet_name(name: str) -> str:
    return re.sub(r'[^\w]', '_', str(name))


def _build_holiday_col_map(holidays_df: pd.DataFrame) -> dict:
    if holidays_df is None or len(holidays_df) == 0:
        return {}
    col_map = {}
    for orig_name in holidays_df['holiday'].unique():
        sanitized = _sanitize_prophet_name(orig_name)
        if sanitized not in col_map:
            col_map[sanitized] = orig_name
    return col_map


# ── Scoring ───────────────────────────────────────────────────────────────────
def score_model(yt, yp) -> dict:
    yt, yp = np.array(yt, dtype=float), np.array(yp, dtype=float)
    mae  = float(mean_absolute_error(yt, yp))
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    r2   = float(r2_score(yt, yp)) if len(yt) >= 3 else None
    mape = float(np.mean(np.abs((yt - yp) / (np.abs(yt) + 1e-9))) * 100)
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE (%)': mape}


# ── Statistical forecasters ───────────────────────────────────────────────────
def forecast_holt(history, n_steps, alpha=None, beta=None):
    y = np.array(history, dtype=float)
    n = len(y)
    if n < 2:
        return np.array([y[-1]] * n_steps)
    best_mape, best_a, best_b = np.inf, 0.3, 0.1
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
            except Exception:
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
    best_mape, best_a = np.inf, 0.3
    for a in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        lvl = y[0]; fitted = []
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
    weights = np.arange(1, w+1, dtype=float); weights /= weights.sum()
    base = float(np.dot(y[-w:], weights))
    trend = float(y[-1] - y[-2]) if len(y) >= 2 else 0.0
    trend = np.clip(trend, -abs(base)*0.3, abs(base)*0.3)
    return np.array([base + trend*(s+1)*0.5 for s in range(n_steps)])


def loo_cv_stat(history, method_fn, n_steps=1) -> dict:
    y = np.array(history, dtype=float)
    yt_all, yp_all = [], []
    for leave in range(1, len(y)):
        train = y[:leave]
        actual = y[leave]
        try:
            preds = method_fn(train, n_steps)
            if isinstance(preds, tuple): preds = preds[0]
            yp_all.append(float(preds[0])); yt_all.append(actual)
        except Exception:
            pass
    if len(yt_all) < 1:
        return {'MAE': np.inf, 'RMSE': np.inf, 'R2': -999, 'MAPE (%)': np.inf}
    return score_model(np.array(yt_all), np.array(yp_all))


# ── ML helpers ────────────────────────────────────────────────────────────────
SCALED_MODELS = {'SVR', 'KNN', 'Ridge', 'Lasso', 'ElasticNet', 'Linear Regression', 'Huber'}


def build_features(series, n_lags=1, cat_id=0.0):
    pad = list(series)
    while len(pad) <= n_lags:
        pad.insert(0, pad[0])
    X_all, y_all = [], []
    for i in range(n_lags, len(pad)):
        lags = [pad[i - l] for l in range(1, n_lags + 1)]
        win  = pad[max(0, i-3):i]
        feat = lags + [
            np.mean(win), np.std(win) if len(win) > 1 else 0.0,
            pad[i-1] - pad[i-2] if i >= 2 else 0.0, cat_id
        ]
        X_all.append(feat); y_all.append(pad[i])
    return np.array(X_all), np.array(y_all)


def get_ml_models(n_train: int) -> dict:
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


def loo_cv_ml(Xc, yc, model_name: str, model_obj) -> dict:
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
        except Exception:
            pass
    if len(yt_all) < 2:
        return {'MAE': np.inf, 'RMSE': np.inf, 'R2': -999, 'MAPE (%)': np.inf}
    return score_model(np.array(yt_all), np.array(yp_all))


# ── Main training ─────────────────────────────────────────────────────────────
def train_best_per_program(df, target: str, n_lags: int, test_ratio: float):
    import copy
    from collections import Counter

    active  = get_active_programs(df)
    cat_enc = {c: float(i) for i, c in enumerate(active)}
    years   = sorted(df['Tahun'].unique())
    single  = len(years) == 1

    per_prog, detail_rows = {}, []

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

        stat_candidates = {
            'Holt Smoothing': lambda h, s: forecast_holt(h, s),
            'Exp Smoothing':  lambda h, s: (forecast_ses(h, s)[0],),
            'Weighted MA':    lambda h, s: (forecast_moving_avg(h, s),),
        }
        stat_scores = {}
        for mname, fn in stat_candidates.items():
            sc = loo_cv_stat(list(sub), fn)
            stat_scores[mname] = sc
            detail_rows.append({'Program': cat, 'Model': mname, **sc})

        ml_scores, ml_models_fitted = {}, {}
        use_ml = (len(sub) >= 8)
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
                    ml_models_fitted[mname] = {'model': mdl_full, 'scaler': sc_full}
                except Exception:
                    pass

        all_scores = {**stat_scores, **ml_scores}
        valid = {m: s for m, s in all_scores.items() if s['MAPE (%)'] < 200}
        pool  = valid if valid else (all_scores or stat_scores)
        best_name = min(pool, key=lambda m: pool[m]['MAPE (%)'])
        best_sc   = all_scores[best_name]
        is_stat   = best_name in stat_candidates

        entry = {
            'best_name': best_name, 'method_type': 'stat' if is_stat else 'ml',
            'history': list(sub), 'single': False, 'metrics': best_sc,
            'all_scores': all_scores, 'cat_id': cat_enc.get(cat, 0.0),
        }
        if is_stat:
            entry['stat_fn_name'] = best_name
        elif best_name in ml_models_fitted:
            info_ml = ml_models_fitted[best_name]
            entry['best_model']   = info_ml['model']
            entry['scaler']       = info_ml['scaler']
            entry['n_lags_used']  = min(n_lags, 2)
        per_prog[cat] = entry

    bpp_rows = []
    for cat, info in per_prog.items():
        m = info.get('metrics', {})
        bpp_rows.append({
            'Program': cat, 'Model': info['best_name'],
            'Tipe': '📊 Statistik' if info.get('method_type') == 'stat' else '🤖 ML',
            'R2': m.get('R2'), 'MAPE (%)': m.get('MAPE (%)'),
            'MAE': m.get('MAE'), 'RMSE': m.get('RMSE'),
        })

    best_per_prog = pd.DataFrame(bpp_rows)
    detail_df     = pd.DataFrame(detail_rows) if detail_rows else pd.DataFrame()
    valid_bpp = [r for r in bpp_rows if r['MAPE (%)'] is not None and r['MAPE (%)'] < 200]
    avg_r2   = float(np.mean([r['R2'] for r in valid_bpp if r['R2'] is not None])) if valid_bpp else None
    avg_mape = float(np.mean([r['MAPE (%)'] for r in valid_bpp])) if valid_bpp else 0.0
    avg_mae  = float(np.mean([r['MAE'] for r in valid_bpp])) if valid_bpp else 0.0
    overall_best = Counter(r['Model'] for r in bpp_rows).most_common(1)[0][0] if bpp_rows else 'N/A'

    return {
        'per_prog': per_prog, 'best_per_program': best_per_prog,
        'detail': detail_df, 'results_df': pd.DataFrame(),
        'best_name': overall_best, 'best_r2': avg_r2,
        'best_mape': avg_mape, 'best_mae': avg_mae,
        'cat_enc': cat_enc, 'single': single,
        'n_lags': n_lags, 'target': target,
        'active_programs': active,
    }, None


def run_ml(df, target: str, n_lags: int, test_ratio: float):
    return train_best_per_program(df, target, n_lags, test_ratio)


# ── Forecast future ───────────────────────────────────────────────────────────
def forecast(df, ml: dict, n_years: int) -> pd.DataFrame:
    target   = ml['target']
    active   = ml['active_programs']
    per_prog = ml.get('per_prog', {})
    base_yr  = int(df['Tahun'].max())
    rows     = []
    STAT_METHODS = {'Holt Smoothing', 'Exp Smoothing', 'Weighted MA'}

    for cat in active:
        info    = per_prog.get(cat)
        history = list(df[df['Kategori'] == cat].sort_values('Tahun')[target].dropna().values.astype(float))
        if not history: continue
        best_nm = info.get('best_name', 'Holt Smoothing') if info else 'Holt Smoothing'

        def _ci_pct(fy): return min(0.05 * fy, 0.25)

        if info is None or info.get('single', True) or best_nm in STAT_METHODS:
            for fy in range(1, n_years + 1):
                try:
                    if best_nm == 'Holt Smoothing' or info is None:
                        pred = float(forecast_holt(history, 1)[0][0])
                    elif best_nm == 'Exp Smoothing':
                        pred = float(forecast_ses(history, 1)[0][0])
                    elif best_nm == 'Weighted MA':
                        pred = float(forecast_moving_avg(history, 1)[0])
                    else:
                        pred = float(forecast_holt(history, 1)[0][0])
                except Exception:
                    pred = history[-1] * 1.05
                pred = max(0.0, pred)
                ci   = _ci_pct(fy)
                rows.append({
                    'Kategori': cat, 'Tahun': base_yr + fy, target: pred,
                    f'{target}_upper': pred * (1 + ci),
                    f'{target}_lower': max(0.0, pred * (1 - ci)),
                    f'{target}_ci_pct': ci * 100,
                    'Type': f'Prediksi ({best_nm})',
                })
                history.append(pred)
            continue

        mdl       = info.get('best_model')
        sc        = info.get('scaler')
        cat_id    = info.get('cat_id', 0.0)
        nlags_use = info.get('n_lags_used', min(ml['n_lags'], 2))
        last_actual = history[-1]
        mape_val  = info.get('metrics', {}).get('MAPE (%)', None)
        mape_base = (mape_val / 100.0) if mape_val and mape_val > 0 else 0.10

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
                except Exception:
                    pred = None
            if pred is None:
                try:
                    pred = float(forecast_holt(history, 1)[0][0])
                except Exception:
                    pred = history[-1] * 1.05
            pred = max(0.0, pred)
            ci = min(mape_base * fy, 0.40)
            rows.append({
                'Kategori': cat, 'Tahun': base_yr + fy, target: pred,
                f'{target}_upper': pred * (1 + ci),
                f'{target}_lower': max(0.0, pred * (1 - ci)),
                f'{target}_ci_pct': ci * 100,
                'Type': 'Prediksi',
            })
            history.append(pred)

    return pd.DataFrame(rows)


def compute_monthly_breakdown(df_raw_monthly, yearly_pred_df, target: str) -> pd.DataFrame:
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
            weights = pd.Series({m: 1/12 for m in range(1, 13)})
        for m in range(1, 13):
            if m not in weights.index:
                weights[m] = 0.0
        weights = weights.sort_index()
        wsum = weights.sum()
        weights = weights / wsum if wsum > 0 else pd.Series({m: 1/12 for m in range(1, 13)})
        cat_pred = yearly_pred_df[yearly_pred_df['Kategori'] == cat]
        for _, row in cat_pred.iterrows():
            yr_total = float(row[target])
            yr_int   = int(row['Tahun'])
            for bulan, w in weights.items():
                rows.append({
                    'Kategori': cat, 'Tahun': yr_int, 'Bulan': int(bulan),
                    'Periode': f"{yr_int}-{int(bulan):02d}",
                    target: max(0.0, yr_total * w), 'Type': 'Prediksi Bulanan',
                })
    return pd.DataFrame(rows)


# ── Conclusion ────────────────────────────────────────────────────────────────
def build_conclusion(ml_result: dict, per_prog_result, df, target: str, n_future: int) -> list:
    lines = []
    if ml_result is None:
        return lines
    bpp      = ml_result.get('best_per_program', pd.DataFrame())
    per_prog = ml_result.get('per_prog', {})

    if not bpp.empty and 'R2' in bpp.columns:
        r2_val   = float(bpp['R2'].mean())
        mape_val = float(bpp['MAPE (%)'].mean())
    else:
        r2_val, mape_val = ml_result.get('best_r2', 0.0), ml_result.get('best_mape', 0.0)

    r2_grade   = ("Sangat Baik (>0.9)" if r2_val > 0.9 else "Baik (0.8–0.9)" if r2_val > 0.8
                  else "Cukup (0.6–0.8)" if r2_val > 0.6 else "Lemah (<0.6)")
    mape_grade = ("Sangat Akurat (<10%)" if mape_val < 10 else "Akurat (10–20%)" if mape_val < 20
                  else "Cukup (20–50%)" if mape_val < 50 else "Tidak Akurat (>50%)")

    lines.append(('🎯', 'Pendekatan Prediksi',
        f"Setiap program menggunakan **model terbaiknya sendiri**. "
        f"Rata-rata R² = **{r2_val:.4f}** ({r2_grade}), MAPE = **{mape_val:.2f}%** ({mape_grade})."))

    if not bpp.empty:
        prog_str = ', '.join(f"{r['Program']} → **{r['Model']}**" for _, r in bpp.iterrows())
        lines.append(('📊', 'Model Terbaik per Program', prog_str))
        if len(bpp) > 1:
            worst  = bpp.sort_values('R2').iloc[0]
            best_p = bpp.sort_values('R2', ascending=False).iloc[0]
            lines.append(('🔍', 'Akurasi per Program',
                f"**{best_p['Program']}** paling mudah (R²={best_p['R2']:.3f}, MAPE={best_p['MAPE (%)']:.1f}%). "
                f"**{worst['Program']}** paling sulit (R²={worst['R2']:.3f}, MAPE={worst['MAPE (%)']:.1f}%)."))

    base_yr = int(df['Tahun'].max())
    lines.append(('📅', 'Horizon Prediksi',
        f"Dilatih s/d **{base_yr}**, proyeksi hingga **{base_yr + n_future}** ({n_future} tahun ke depan)."))
    yrs = sorted(df['Tahun'].unique())
    lines.append(('📁', 'Kualitas Data',
        f"Dataset mencakup **{len(yrs)} tahun** ({yrs[0]}–{yrs[-1]}) "
        f"dengan **{len(get_active_programs(df))} program aktif**. "
        + ("✅ Cukup untuk model lag." if len(yrs) >= 4 else "⚠️ Tambah data historis.")))

    if r2_val >= 0.8 and mape_val <= 20:
        rec = "✅ Model layak untuk perencanaan anggaran dan proyeksi klaim BPJS."
    elif r2_val >= 0.6:
        rec = "⚠️ Cukup untuk proyeksi kasar. Validasi manual disarankan."
    else:
        rec = "❌ Akurasi belum optimal. Tambah data ≥5 tahun, atau gunakan Prophet."
    lines.append(('💡', 'Rekomendasi', rec))
    return lines


def run_prophet(df_monthly_raw, target: str, cat: str, n_months: int, use_holidays: bool = True):
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
        s_mode = 'multiplicative' if n_data >= 24 else 'additive'
        cp_scale = 0.05 if n_data >= 24 else 0.03
        m = Prophet(
            yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
            holidays=holidays_df, seasonality_mode=s_mode, interval_width=0.80,
            changepoint_prior_scale=cp_scale, seasonality_prior_scale=5.0,
            holidays_prior_scale=5.0, growth='flat' if n_data < 12 else 'linear',
        )
        m.fit(cat_df)
        future = m.make_future_dataframe(periods=n_months, freq='MS')
        fc = m.predict(future)
        fc['yhat']       = fc['yhat'].clip(lower=y_floor, upper=y_cap)
        fc['yhat_lower'] = fc['yhat_lower'].clip(lower=y_floor)
        fc['yhat_upper'] = fc['yhat_upper'].clip(lower=y_floor, upper=y_cap * 1.2)

        hist_pred = fc[fc['ds'].isin(cat_df['ds'])]
        if len(hist_pred) > 0:
            yt = cat_df.set_index('ds').loc[hist_pred['ds'], 'y'].values
            yp = hist_pred['yhat'].values
            r2_is   = float(r2_score(yt, yp)) if len(yt) > 1 else 0.0
            mape_is = float(np.mean(np.abs((yt - yp) / (np.abs(yt) + 1e-9))) * 100)
        else:
            r2_is = mape_is = 0.0

        n_hol     = len(holidays_df) if holidays_df is not None else 0
        h_col_map = _build_holiday_col_map(holidays_df) if holidays_df is not None else {}

        return {
            'model': m, 'forecast': fc, 'history': cat_df,
            'r2_insample': r2_is, 'mape_insample': mape_is,
            'n_holidays': n_hol, 'gcal_used': n_hol > 0,
            'holiday_col_map': h_col_map,
        }, None
    except Exception as e:
        return None, str(e)