"""
app.py — Entry point BPJS ML Dashboard.
Jalankan: streamlit run app.py
"""
import warnings
warnings.filterwarnings('ignore')

# ── Imports standar ───────────────────────────────────────────────────────────
import re
import hashlib
import io
import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Optional deps (agar tidak crash jika belum install) ───────────────────────
try:
    from xgboost import XGBRegressor
except ImportError:
    pass
try:
    from lightgbm import LGBMRegressor
except ImportError:
    pass

# ── Modul internal ────────────────────────────────────────────────────────────
from auth import check_password, render_user_badge, _secret
from config import inject_global_css, KEEPALIVE_JS, DARK, COLORS, styled_chart, hex_to_rgba
from data_utils import (load_raw, parse_dataset, merge_all,
                        _detect_cols_quick, _build_raw_monthly,
                        analyze_program_changes, get_active_programs, clean_num)
from history_utils import (load_history_meta, load_history_entry, save_history_meta,
                            delete_history_entry, add_to_history, init_session_defaults)
from ml_core import (run_ml, forecast, compute_monthly_breakdown, build_conclusion,
                     run_prophet, INDONESIAN_HOLIDAYS, PROPHET_OK, GCAL_KEY,
                     score_model, forecast_holt, forecast_ses, forecast_moving_avg,
                     build_features, SCALED_MODELS)
from export_utils import export_excel
from ai_utils import (_get_api_key, _detect_provider, _call_ai, _call_ai_groq, _call_ai_gemini,
                      _fetch_worldbank, _ai_analyze_peak_trough,
                      _web_search, _build_chat_data_ctx, _build_chat_wb_ctx,
                      _session_new, _session_save_current, _session_load, _session_delete,
                      _sessions_load, _qa_memory_get_relevant, _qa_memory_save,
                      _chat_answer, _qa_memory_stats)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG — harus dipanggil paling pertama sebelum check_password
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="BPJS ML Dashboard",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# ── Auth ──────────────────────────────────────────────────────────────────────
check_password()

# ── CSS & JS global ───────────────────────────────────────────────────────────
inject_global_css()
st.markdown(KEEPALIVE_JS, unsafe_allow_html=True)

# ── Session defaults ──────────────────────────────────────────────────────────
init_session_defaults()

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Upload, settings, riwayat
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
                        # Preserve df_raw_for_ai sebelum overwrite session state
                        _preserved_sebab = st.session_state.get('df_raw_for_ai')

                        st.session_state.active_data     = df_h
                        st.session_state.active_results  = res_h
                        st.session_state.active_entry_id = h['id']
                        for k, v in extra_h.items():
                            st.session_state[k] = v

                        # Restore df_raw_for_ai jika extra_h tidak menyimpannya
                        if st.session_state.get('df_raw_for_ai') is None and _preserved_sebab is not None:
                            st.session_state['df_raw_for_ai'] = _preserved_sebab

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

            # ── Bangun raw_monthly (data agregat bulanan) ─────────────────────
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

            # ── Deteksi & simpan file SEBAB_KLAIM untuk AI context ────────────
            # File ini dikenali dari keberadaan kolom "sebab" + "program"
            sebab_frames = []
            for yh, raw_df, fname in files_info:
                cols_lower = [c.lower().strip() for c in raw_df.columns]
                has_sebab   = any("sebab" in c or "cause" in c or "reason" in c for c in cols_lower)
                has_program = any("program" in c for c in cols_lower)

                if has_sebab and has_program:
                    df_sebab = raw_df.copy()

                    # Tambahkan kolom Periode dari nama file jika belum ada
                    has_periode = any(
                        "periode" in c or "period" in c or "date" in c or "tahun" in c
                        for c in cols_lower
                    )
                    if not has_periode:
                        df_sebab['Periode'] = yh

                    sebab_frames.append(df_sebab)
                    st.sidebar.success(
                        f"📋 {fname} → terdeteksi sebagai data Sebab Klaim ({len(df_sebab)} baris)"
                    )

            if sebab_frames:
                df_sebab_combined = pd.concat(sebab_frames, ignore_index=True)
                st.session_state['df_raw_for_ai']  = df_sebab_combined  # key utama untuk AI
                st.session_state['df_sebab_klaim'] = df_sebab_combined  # key fallback
                st.sidebar.success(f"🤖 AI context: {len(df_sebab_combined)} baris sebab klaim siap")
            else:
                # Fallback: pakai raw_monthly jika tidak ada file sebab klaim terpisah
                if st.session_state.get('raw_monthly') is not None:
                    st.session_state['df_raw_for_ai'] = st.session_state['raw_monthly']
                else:
                    st.session_state['df_raw_for_ai'] = None

        else:
            st.error("Gagal memproses data. Pastikan file punya kolom PROGRAM dan KASUS.")

df             = st.session_state.active_data
results_cache  = st.session_state.active_results
df_raw_monthly = st.session_state.get('raw_monthly', None)

# ══════════════════════════════════════════════════════════════════════════════
# EMPTY STATE — belum ada data
# ══════════════════════════════════════════════════════════════════════════════
has_nom: bool     = False
single_yr: bool   = False
years: list       = []
latest_year       = None
prev_year         = None
active_progs: set = set()

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
        (f1, "🤖", "Metode Adaptif",
         "Holt Smoothing, SES, WMA untuk data kecil. ML (XGBoost, RF) otomatis aktif untuk data ≥ 8 tahun."),
        (f2, "📅", "Kalender Indonesia",
         "Prophet + Google Calendar Indonesia. Semua hari libur nasional otomatis diambil dari API resmi Google."),
        (f3, "📥", "Export Excel",
         "Export prediksi tahunan & bulanan ke Excel dengan chart otomatis, siap untuk presentasi."),
    ]:
        with col:
            st.markdown(f'''
            <div style="background:#ffffff;border:1px solid #e2e8f0;
            border-radius:14px;padding:24px;text-align:center;height:160px;
            border-top:3px solid #3b82f6;">
              <div style="font-size:2rem;margin-bottom:8px;">{icon}</div>
              <div style="font-weight:700;color:#0f172a;margin-bottom:8px;font-size:.95rem;">{title}</div>
              <div style="color:#475569;font-size:.82rem;line-height:1.6;">{desc}</div>
            </div>''', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">📋 <b>Format kolom yang didukung:</b> '
        '<code>PROGRAM</code> · <code>KASUS</code> · <code>NOMINAL</code> · <code>DATE</code>'
        '<br>Upload 1+ file CSV/Excel.</div>',
        unsafe_allow_html=True
    )
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# DERIVED META
# ══════════════════════════════════════════════════════════════════════════════
years        = sorted(df['Tahun'].unique())
latest_year  = years[-1]
prev_year    = years[-2] if len(years) >= 2 else None
active_progs = get_active_programs(df)
all_progs    = sorted(df['Kategori'].unique())
has_nom      = 'Nominal' in df.columns
targets      = ['Kasus'] + (['Nominal'] if has_nom else [])
single_yr    = len(years) == 1
prog_changes = analyze_program_changes(df)

# ── Fallback chain: pastikan df_raw_for_ai selalu terisi ─────────────────────
if st.session_state.get('df_raw_for_ai') is None:
    for _fallback_key in ['df_sebab_klaim', 'raw_monthly']:
        _val = st.session_state.get(_fallback_key)
        if _val is not None:
            st.session_state['df_raw_for_ai'] = _val
            break

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
    ✅ <b>Program aktif (tahun {latest_year}) → diprediksi ({len(active_progs)}):</b> {', '.join(active_progs)}<br>
    📊 <b>Total baris setelah agregasi:</b> {len(df)}<br><br>
    <b>Perubahan Program per Tahun:</b><br>
    {change_html if change_html else 'Hanya 1 tahun data.'}
    </div>""", unsafe_allow_html=True)

    vdf = df[['Tahun', 'Kategori', 'Kasus'] + (['Nominal'] if has_nom else [])].copy()
    vdf = vdf.sort_values(['Tahun', 'Kategori'])
    if 'Nominal' in vdf.columns:
        vdf['Nominal (T)'] = (vdf['Nominal'] / 1e12).round(4)
        vdf['Nominal (B)'] = (vdf['Nominal'] / 1e9).round(2)
    st.dataframe(vdf, width='stretch', height=320)

    # ── Status AI context (untuk troubleshoot) ────────────────────────────────
    st.markdown("**🤖 Status AI Context (df_raw_for_ai)**")
    df_ai_ctx = st.session_state.get('df_raw_for_ai')
    if df_ai_ctx is not None:
        st.success(
            f"✅ df_raw_for_ai tersedia: {len(df_ai_ctx)} baris × {len(df_ai_ctx.columns)} kolom"
        )
        st.caption(f"Kolom: {list(df_ai_ctx.columns)}")
        st.dataframe(df_ai_ctx.head(5), width='stretch')
    else:
        st.error("❌ df_raw_for_ai = None → AI tidak akan bisa membaca data sebab klaim!")
        st.caption("Solusi: Upload file yang mengandung kolom 'Sebab Klaim' atau 'Sebab'")

# ══════════════════════════════════════════════════════════════════════════════
# HERO & KPI
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="hero-wrap">'
    '<div class="hero-logo">📊 Dashboard Prediksi Klaim BPJS Ketenagakerjaan</div>'
    '<div class="hero-sub">Analisis tren historis &amp; proyeksi — '
    'model adaptif statistik &amp; machine learning per program</div>'
    '</div>', unsafe_allow_html=True)

if single_yr:
    st.markdown("""
    <div class="warn">⚠️ <b>Mode 1 Tahun:</b>
    Prediksi menggunakan ekstrapolasi asumsi pertumbuhan 5%/tahun.
    Upload data multi-tahun untuk prediksi ML penuh.</div>
    """, unsafe_allow_html=True)

if prog_changes:
    last_change = list(prog_changes.items())[-1]
    (y0, y1), ch = last_change
    if ch['added'] or ch['removed']:
        added_str   = ', '.join(ch['added'])   if ch['added']   else '–'
        removed_str = ', '.join(ch['removed']) if ch['removed'] else '–'
        st.markdown(f"""
        <div class="warn">
        📌 <b>Perubahan program {y0}→{y1}:</b>
        &nbsp; Ditambah: <b style="color:#86efac">{added_str}</b>
        &nbsp;|&nbsp; Dihapus: <b style="color:#fca5a5">{removed_str}</b>
        &nbsp;→ Prediksi hanya untuk program aktif tahun {y1}.
        </div>""", unsafe_allow_html=True)

# ── Program filter ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("---")
    st.markdown("**🎯 Filter Program**")
    st.caption("Berlaku global untuk semua tab.")
    selected_filter = st.multiselect(
        "Tampilkan program:",
        options=list(active_progs),
        default=list(active_progs),
        key='prog_filter_widget',
    )
    if not selected_filter:
        selected_filter = list(active_progs)

filtered_progs = selected_filter
df_active_only = df[df['Kategori'].isin(active_progs)]

# ── KPI calcs ─────────────────────────────────────────────────────────────────
kpi_delta_k = kpi_delta_n = kpi_avg_growth = ""
if len(years) >= 2:
    yr_kasus = df_active_only.groupby('Tahun')['Kasus'].sum()
    if yr_kasus.iloc[-1] > 0 and yr_kasus.iloc[-2] > 0:
        delta_k_pct = (yr_kasus.iloc[-1] / yr_kasus.iloc[-2] - 1) * 100
        sign_k = "▲" if delta_k_pct >= 0 else "▼"
        cls_k  = "delta-pos" if delta_k_pct >= 0 else "delta-neg"
        kpi_delta_k = f'<div class="delta {cls_k}">{sign_k} {abs(delta_k_pct):.1f}% vs {prev_year}</div>'
    if has_nom:
        yr_nom = df_active_only.groupby('Tahun')['Nominal'].sum()
        if yr_nom.iloc[-1] > 0 and yr_nom.iloc[-2] > 0:
            delta_n_pct = (yr_nom.iloc[-1] / yr_nom.iloc[-2] - 1) * 100
            sign_n = "▲" if delta_n_pct >= 0 else "▼"
            cls_n  = "delta-pos" if delta_n_pct >= 0 else "delta-neg"
            kpi_delta_n = f'<div class="delta {cls_n}">{sign_n} {abs(delta_n_pct):.1f}% vs {prev_year}</div>'
    growths = []
    for i in range(1, len(years)):
        k_prev = df_active_only[df_active_only['Tahun'] == years[i-1]]['Kasus'].sum()
        k_curr = df_active_only[df_active_only['Tahun'] == years[i]]['Kasus'].sum()
        if k_prev > 0:
            growths.append((k_curr / k_prev - 1) * 100)
    avg_g  = np.mean(growths) if growths else 0
    sign_g = "▲" if avg_g >= 0 else "▼"
    cls_g  = "delta-pos" if avg_g >= 0 else "delta-neg"
    kpi_avg_growth = f'<div class="delta {cls_g}">{sign_g} {abs(avg_g):.1f}%/thn ({years[0]}–{years[-1]})</div>'

tk        = int(df_active_only['Kasus'].sum())
tk_latest = int(df_active_only[df_active_only['Tahun'] == latest_year]['Kasus'].sum())

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(
        f'<div class="kpi"><div class="val">{len(years)}</div>'
        f'<div class="lbl">📅 Tahun Data</div>'
        f'<div class="delta delta-neu">{years[0]} – {years[-1]}</div></div>',
        unsafe_allow_html=True)
with c2:
    st.markdown(
        f'<div class="kpi"><div class="val">{len(active_progs)}</div>'
        f'<div class="lbl">🏷️ Program Aktif</div>'
        f'<div class="delta delta-neu">{", ".join(active_progs)}</div></div>',
        unsafe_allow_html=True)
with c3:
    st.markdown(
        f'<div class="kpi"><div class="val">{tk_latest:,}</div>'
        f'<div class="lbl">📋 Kasus {latest_year}</div>{kpi_delta_k}</div>',
        unsafe_allow_html=True)
with c4:
    if has_nom:
        tn = df_active_only[df_active_only['Tahun'] == latest_year]['Nominal'].sum() / 1e9
        st.markdown(
            f'<div class="kpi"><div class="val">Rp{tn:,.1f}B</div>'
            f'<div class="lbl">💰 Nominal {latest_year}</div>{kpi_delta_n}</div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="kpi"><div class="val">{latest_year}</div>'
            f'<div class="lbl">📅 Data Terbaru</div></div>',
            unsafe_allow_html=True)
with c5:
    st.markdown(
        f'<div class="kpi"><div class="val">{tk:,}</div>'
        f'<div class="lbl">📊 Total Kasus</div>{kpi_avg_growth}</div>',
        unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
df_plot = df[df['Kategori'].isin(filtered_progs)].copy()

if len(filtered_progs) < len(active_progs):
    filter_str = ", ".join(filtered_progs)
    st.markdown(
        f'<div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;'
        f'padding:8px 14px;font-size:.82rem;color:#1e40af;margin-top:4px;">'
        f'🔵 <b>Filter aktif:</b> {len(filtered_progs)} dari {len(active_progs)} program'
        f' &nbsp;|&nbsp; <b>{filter_str}</b></div>',
        unsafe_allow_html=True)

_is_single_prog = len(filtered_progs) == 1
if _is_single_prog:
    _sp        = filtered_progs[0]
    _sp_total  = int(df_plot['Kasus'].sum())
    _sp_latest = int(df_plot[df_plot['Tahun'] == latest_year]['Kasus'].sum())
    st.markdown(
        f'<div style="background:linear-gradient(135deg,#eff6ff,#f0fdf4);border:1.5px solid #2563eb;'
        f'border-radius:14px;padding:14px 22px;margin-bottom:12px;">'
        f'<div style="font-size:1.05rem;font-weight:800;color:#0f172a;">'
        f'Mode: Program <span style="color:#2563eb">{_sp}</span></div>'
        f'<div style="font-size:.8rem;color:#64748b;margin-top:2px;">'
        f'Total historis: <b>{_sp_total:,}</b> · {latest_year}: <b>{_sp_latest:,}</b> kasus</div>'
        f'</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📈 Overview", "🤖 ML Analysis", "🔮 Prediksi", "📥 Export", "💬 AI Analyst"])

from tabs.tab_overview  import render_tab_overview
from tabs.tab_ml        import render_tab_ml
from tabs.tab_prediksi  import render_tab_prediksi
from tabs.tab_export    import render_tab_export
from tabs.tab_ai        import render_tab_ai

with tab1:
    render_tab_overview(
        df, df_plot, df_active_only, active_progs, filtered_progs,
        years, latest_year, prev_year, has_nom, single_yr,
        prog_changes, n_lags, n_future, COLORS, DARK, styled_chart, hex_to_rgba,
        _fetch_worldbank, _ai_analyze_peak_trough, _get_api_key
    )

with tab2:
    render_tab_ml(
        df, active_progs, filtered_progs, targets, n_lags, test_pct, n_future,
        years, single_yr, has_nom, latest_year,
        results_cache, COLORS, DARK, styled_chart, hex_to_rgba,
        run_ml, build_conclusion, run_prophet, PROPHET_OK, GCAL_KEY, INDONESIAN_HOLIDAYS,
        add_to_history, df_raw_monthly
    )

with tab3:
    render_tab_prediksi(
        df, df_plot, active_progs, filtered_progs, targets, n_lags, test_pct, n_future,
        years, latest_year, single_yr, has_nom,
        results_cache, COLORS, DARK, styled_chart, hex_to_rgba,
        run_ml, forecast, compute_monthly_breakdown, add_to_history, df_raw_monthly
    )

with tab4:
    render_tab_export(
        df, active_progs, years, latest_year, has_nom,
        results_cache, export_excel
    )

with tab5:
    render_tab_ai(
        df, active_progs, years, latest_year, prev_year, has_nom,
        COLORS, DARK,
        _get_api_key, _session_new, _session_save_current, _session_load,
        _session_delete, _sessions_load, _chat_answer, _qa_memory_stats,
        _fetch_worldbank
    )

# ── User badge di sidebar (logout) ────────────────────────────────────────────
render_user_badge()