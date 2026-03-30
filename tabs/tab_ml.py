"""tabs/tab_ml.py — Tab 2: ML Analysis."""
import hashlib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime


def render_tab_ml(df, active_progs, filtered_progs, targets, n_lags, test_pct, n_future,
                  years, single_yr, has_nom, latest_year,
                  results_cache, COLORS, DARK, styled_chart, hex_to_rgba,
                  run_ml, build_conclusion, run_prophet, PROPHET_OK, GCAL_KEY,
                  INDONESIAN_HOLIDAYS, add_to_history, df_raw_monthly):

    n_yrs = len(years)
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
            st.session_state[f'per_prog_{target_ml}'] = ml_res

            # ── Simpan ke session_state agar AI tab bisa akses ────────────────
            # Menyimpan hasil ML terbaru untuk digunakan sebagai context di chatbot AI.
            # Tidak ada data raw yang tersimpan — hanya metadata model dan metrik akurasi.
            st.session_state['ml_result'] = ml_res
            st.session_state['ml_result_target'] = target_ml

            data_hash = hashlib.md5(df.to_csv().encode()).hexdigest()[:8]
            eid   = f"{data_hash}_{target_ml}_L{n_lags}_T{test_pct}"
            label = f"📁 {datetime.now().strftime('%d/%m %H:%M')} | {target_ml} | {len(years)}yr"
            extra_snapshot = {k: st.session_state[k]
                              for k in ['raw_monthly','forecast_Kasus','forecast_Nominal',
                                        'forecast_monthly_Kasus','forecast_monthly_Nominal',
                                        'forecast_annual_Kasus','forecast_annual_Nominal',
                                        'last_forecast','last_forecast_monthly']
                              if k in st.session_state and st.session_state[k] is not None}
            add_to_history(label, eid, df.copy(), dict(results_cache), extra_snapshot)

            # ── Notifikasi ke user bahwa AI sudah bisa akses hasil ML ─────────
            st.success(
                f"✅ Model selesai dilatih! "
                f"AI Analyst di tab **💬 AI** sekarang bisa menjawab pertanyaan "
                f"prediksi berdasarkan hasil model ini."
            )

    if ml_res:
        bpp      = ml_res.get('best_per_program', pd.DataFrame())
        per_prog = ml_res.get('per_prog', {})

        mtab1, mtab2, mtab3, mtab4 = st.tabs(
            ["📊 Performa Model", "📋 Tabel Detail", "💡 Kesimpulan", "🔮 Prophet + Kalender"])

        # ── Sub-tab 1: Performa ───────────────────────────────────────────────
        with mtab1:
            if bpp.empty:
                st.info("Klik **Jalankan Analisis ML** untuk melihat hasil.")
            else:
                avg_mape = bpp['MAPE (%)'].mean() if 'MAPE (%)' in bpp.columns else None
                overall_grade = ("🟢 Sangat Baik" if avg_mape and avg_mape < 10 else
                                 "🔵 Baik" if avg_mape and avg_mape < 20 else
                                 "🟡 Cukup" if avg_mape and avg_mape < 50 else "🔴 Perlu Data Lebih Banyak")

                if avg_mape:
                    best_prog  = bpp.loc[bpp['MAPE (%)'].idxmin(), 'Program']
                    worst_prog = bpp.loc[bpp['MAPE (%)'].idxmax(), 'Program']
                    st.markdown(f"""<div class="success-box">
                    🔍 <b>Auto-Insight:</b> Kualitas prediksi: <b>{overall_grade}</b> (Avg MAPE {avg_mape:.1f}%).
                    Terbaik: <b>{best_prog}</b> · Perlu perhatian: <b>{worst_prog}</b>.
                    {"Data " + str(n_yrs) + " tahun → metode statistik." if n_yrs < 8 else f"Data {n_yrs} tahun → ML tersedia."}
                    </div>""", unsafe_allow_html=True)

                # Model cards
                cards_html = '<div style="display:flex;flex-wrap:wrap;gap:10px;margin:12px 0;">'
                for _, row in bpp.iterrows():
                    mape_v = row.get('MAPE (%)', None)
                    pill_cls = ("mpill-green" if mape_v and mape_v < 10 else
                                "mpill-blue" if mape_v and mape_v < 20 else
                                "mpill-yellow" if mape_v and mape_v < 50 else "mpill-red")
                    mape_txt = f"MAPE: {mape_v:.1f}%" if mape_v else "–"
                    cards_html += (f'<div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;'
                                   f'padding:12px 14px;min-width:140px;">'
                                   f'<div style="font-size:.65rem;color:#64748b;font-weight:700;text-transform:uppercase;">{row["Program"]}</div>'
                                   f'<div style="font-size:.88rem;font-weight:600;color:#0f172a;margin:4px 0;">{row["Model"]}</div>'
                                   f'<span class="mpill {pill_cls}">{mape_txt}</span></div>')
                cards_html += '</div>'
                st.markdown(cards_html, unsafe_allow_html=True)

                # MAPE chart
                bpp_plot = bpp.dropna(subset=['MAPE (%)'])
                if not bpp_plot.empty:
                    fig_mp = px.bar(bpp_plot, x='Program', y='MAPE (%)', color='Model',
                                    color_discrete_sequence=COLORS,
                                    title='MAPE % per Program (lebih rendah = lebih baik)')
                    fig_mp.add_hline(y=20, line_dash='dash', line_color='#fbbf24',
                                     annotation_text='Threshold 20%')
                    fig_mp.update_layout(**DARK, height=360, margin=dict(b=60, t=40))
                    st.plotly_chart(fig_mp, width='stretch')

        # ── Sub-tab 2: Tabel ──────────────────────────────────────────────────
        with mtab2:
            det = ml_res.get('detail', pd.DataFrame())
            if bpp.empty:
                st.info("Klik **Jalankan Analisis ML** untuk melihat model terbaik per program.")
            else:
                st.markdown('<div class="sec">Model Terbaik per Program</div>', unsafe_allow_html=True)
                st.dataframe(
                    bpp.style
                       .format({'R2': '{:.4f}', 'MAPE (%)': '{:.2f}', 'MAE': '{:,.0f}', 'RMSE': '{:,.0f}'}),
                    width='stretch', height=260)

                if not det.empty:
                    with st.expander("📋 Tabel Detail Semua Model × Semua Program"):
                        st.dataframe(
                            det.sort_values(['Program','R2'], ascending=[True,False])
                               .style.format({'R2': '{:.4f}', 'MAPE (%)': '{:.2f}',
                                              'MAE': '{:,.0f}', 'RMSE': '{:,.0f}'}),
                            width='stretch', height=400)

        # ── Sub-tab 3: Kesimpulan ─────────────────────────────────────────────
        with mtab3:
            conclusions = build_conclusion(ml_res, ml_res, df, target_ml, n_future)
            if conclusions:
                st.markdown('<div class="sec">Kesimpulan & Rekomendasi Otomatis</div>', unsafe_allow_html=True)
                for icon, title, text in conclusions:
                    st.markdown(f"""
                    <div style="background:#fff;border:1px solid #e2e8f0;border-radius:14px;
                                padding:18px 22px;margin:10px 0;border-left:3px solid #3b82f6;">
                        <div style="font-size:.65rem;font-weight:700;color:#334155;
                                    text-transform:uppercase;letter-spacing:2px;margin-bottom:8px;">{icon} {title}</div>
                        <div style="color:#334155;font-size:.9rem;line-height:1.75;">{text}</div>
                    </div>""", unsafe_allow_html=True)
            else:
                st.info("Jalankan analisis ML untuk mendapatkan kesimpulan otomatis.")

        # ── Sub-tab 4: Prophet ────────────────────────────────────────────────
        with mtab4:
            df_raw_m_p = st.session_state.get('raw_monthly', None)
            if not PROPHET_OK:
                st.warning("**Prophet belum terinstall.** Tambahkan `prophet` ke `requirements.txt`.")
            elif df_raw_m_p is None or len(df_raw_m_p) == 0:
                st.warning("Upload dataset dengan data bulanan terlebih dahulu untuk menggunakan Prophet.")
            else:
                n_holidays = len(INDONESIAN_HOLIDAYS)
                n_htypes   = INDONESIAN_HOLIDAYS['holiday'].nunique() if n_holidays > 0 else 0
                gcal_status = (
                    f"✅ <b>{n_holidays} hari libur</b>, <b>{n_htypes} jenis</b> dari Google Calendar API."
                    if n_holidays > 0 else
                    "⚠️ <b>GCAL_KEY belum diset</b> atau API tidak mengembalikan data. Prophet tetap jalan tanpa holiday."
                )
                st.markdown(f"""<div class="info-box">
                🔮 <b>Prophet</b> menangani efek hari libur secara eksplisit dari Google Calendar Indonesia.<br>
                📅 Sumber kalender: Google Calendar API (2019–2028, auto-refresh 24 jam).<br>
                {gcal_status}
                </div>""", unsafe_allow_html=True)

                pc1, pc2 = st.columns(2)
                with pc1:
                    target_prophet = st.selectbox("Target", targets, key='prophet_target')
                with pc2:
                    n_months_prophet = st.slider("Prediksi (bulan)", 6, 36, 12, 6)

                use_holidays = st.checkbox(
                    f"Gunakan kalender hari libur Indonesia ({n_holidays} hari libur, {n_htypes} jenis)",
                    value=(n_holidays > 0))

                _prophet_progs = filtered_progs if filtered_progs else active_progs
                _btn_lbl = (f"🔮 Jalankan Prophet — {_prophet_progs[0]}"
                            if len(_prophet_progs) == 1
                            else f"🔮 Jalankan Prophet ({len(_prophet_progs)} Program)")

                if st.button(_btn_lbl, type="primary", width='stretch'):
                    all_p_results = {}
                    prog_errors   = {}
                    with st.spinner(f"Melatih Prophet untuk {len(_prophet_progs)} program..."):
                        for cp in _prophet_progs:
                            pr, pe = run_prophet(df_raw_m_p, target_prophet, cp, n_months_prophet, use_holidays)
                            if pe:
                                prog_errors[cp] = pe
                            else:
                                all_p_results[cp] = pr
                    for cp, pe in prog_errors.items():
                        st.warning(f"⚠️ {cp}: {pe}")
                    st.session_state['prophet_results'] = all_p_results

                    # ── Simpan prophet results ke session agar AI bisa akses ──
                    if all_p_results:
                        st.session_state['prophet_result'] = all_p_results
                        st.session_state['prophet_result_target'] = target_prophet

                _all_p = st.session_state.get('prophet_results', {})
                if _all_p:
                    for cp, pr in _all_p.items():
                        fc = pr['forecast']
                        hist_pr = pr['history']
                        st.markdown(f'<div class="sec">Hasil Prophet — {cp}</div>', unsafe_allow_html=True)
                        r2_is   = pr.get('r2_insample', None)
                        mape_is = pr.get('mape_insample', None)
                        st.markdown(
                            f'<div class="info-box">📊 <b>{cp}</b> — In-sample: '
                            f'R² = {r2_is:.4f if r2_is is not None else "N/A"} | '
                            f'MAPE = {mape_is:.2f if mape_is is not None else "N/A"}%</div>',
                            unsafe_allow_html=True)
                        fig_p = go.Figure()
                        col_p = COLORS[list(_all_p.keys()).index(cp) % len(COLORS)]
                        future_only = fc[~fc['ds'].isin(hist_pr['ds'])]
                        fig_p.add_trace(go.Scatter(x=hist_pr['ds'], y=hist_pr['y'],
                            name='Aktual', mode='lines+markers', line=dict(color=col_p, width=2)))
                        fig_p.add_trace(go.Scatter(
                            x=list(future_only['ds'])+list(future_only['ds'][::-1]),
                            y=list(future_only['yhat_upper'])+list(future_only['yhat_lower'][::-1]),
                            fill='toself', fillcolor=f'rgba(37,99,235,0.10)',
                            line=dict(color='rgba(0,0,0,0)'), showlegend=False))
                        fig_p.add_trace(go.Scatter(x=future_only['ds'], y=future_only['yhat'],
                            name='Prediksi Prophet', mode='lines',
                            line=dict(color='#2563eb', width=2, dash='dash')))
                        fig_p.update_layout(**DARK, height=380, xaxis_tickangle=-30,
                            margin=dict(t=20, b=60, l=60, r=20))
                        st.plotly_chart(fig_p, width='stretch')
    else:
        st.info("Klik **Jalankan Analisis ML** untuk memulai.")