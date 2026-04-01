"""tabs/tab_ml.py — Tab 2: ML Analysis."""
import hashlib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime


def _build_holiday_program_matrix(prophet_results: dict):
    if not prophet_results:
        return pd.DataFrame(), pd.DataFrame()
    all_data = {}
    for prog, pr in prophet_results.items():
        effects = pr.get('holiday_effects', {})
        clean = {k: v for k, v in effects.items()
                 if k != '_error' and isinstance(v, dict)
                 and v.get('mean_abs_effect', 0) > 1e-6}
        if clean:
            all_data[prog] = clean
    if not all_data:
        return pd.DataFrame(), pd.DataFrame()
    all_holidays = sorted(set(h for pe in all_data.values() for h in pe.keys()))
    rows_abs, rows_signed = [], []
    for h in all_holidays:
        row_a = {'Hari Libur': h}
        row_s = {'Hari Libur': h}
        for prog in sorted(all_data.keys()):
            eff = all_data[prog].get(h, {})
            abs_val = eff.get('mean_abs_effect', 0.0)
            max_e   = eff.get('max_effect', 0.0)
            min_e   = eff.get('min_effect', 0.0)
            signed  = max_e if abs(max_e) >= abs(min_e) else min_e
            row_a[prog] = round(abs_val, 4)
            row_s[prog] = round(signed, 4)
        rows_abs.append(row_a)
        rows_signed.append(row_s)
    df_abs    = pd.DataFrame(rows_abs).set_index('Hari Libur')
    df_signed = pd.DataFrame(rows_signed).set_index('Hari Libur')
    order = df_abs.sum(axis=1).sort_values(ascending=False).index
    return df_abs.loc[order], df_signed.loc[order]


def _build_top_holidays_per_program(prophet_results: dict, top_n: int = 8) -> dict:
    result = {}
    for prog, pr in prophet_results.items():
        effects = pr.get('holiday_effects', {})
        clean = {k: v for k, v in effects.items()
                 if k != '_error' and isinstance(v, dict)
                 and v.get('mean_abs_effect', 0) > 1e-6}
        if not clean:
            result[prog] = []
            continue
        sorted_eff = sorted(clean.items(), key=lambda x: x[1]['mean_abs_effect'], reverse=True)[:top_n]
        prog_list = []
        for h_name, eff in sorted_eff:
            max_e = eff.get('max_effect', 0)
            min_e = eff.get('min_effect', 0)
            direction = "naik" if abs(max_e) >= abs(min_e) else "turun"
            dominant  = max_e if direction == "naik" else min_e
            prog_list.append({
                'holiday': h_name, 'mean_abs': eff['mean_abs_effect'],
                'max_effect': max_e, 'min_effect': min_e,
                'dominant': dominant, 'direction': direction,
            })
        result[prog] = prog_list
    return result


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
            st.session_state.active_results      = results_cache
            st.session_state[f'per_prog_{target_ml}'] = ml_res
            st.session_state['ml_result']        = ml_res
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

            _df_raw_m = st.session_state.get('raw_monthly', None)
            if PROPHET_OK and _df_raw_m is not None and len(_df_raw_m) > 0:
                _prophet_progs = filtered_progs if filtered_progs else active_progs
                _use_holidays  = len(INDONESIAN_HOLIDAYS) > 0
                with st.spinner(f"🔮 Auto-menjalankan Prophet untuk {len(_prophet_progs)} program..."):
                    _auto_prophet, _auto_errors = {}, {}
                    for _cp in _prophet_progs:
                        _pr, _pe = run_prophet(_df_raw_m, target_ml, _cp, 12, _use_holidays)
                        if _pe: _auto_errors[_cp] = _pe
                        else:   _auto_prophet[_cp] = _pr
                for _cp, _pe in _auto_errors.items():
                    st.warning(f"⚠️ Prophet {_cp}: {_pe}")
                if _auto_prophet:
                    st.session_state['prophet_results']       = _auto_prophet
                    st.session_state['prophet_result']        = _auto_prophet
                    st.session_state['prophet_result_target'] = target_ml
                    st.session_state['prophet_auto_target']   = target_ml

            _prophet_done = (PROPHET_OK and st.session_state.get('raw_monthly') is not None
                             and bool(st.session_state.get('prophet_results')))
            _note = " Prophet + efek hari libur juga sudah dianalisis." if _prophet_done else ""
            st.success(f"✅ Model **{target_ml}** selesai!{_note} Tab **💬 AI** siap menjawab prediksi.")

    if ml_res:
        bpp      = ml_res.get('best_per_program', pd.DataFrame())
        per_prog = ml_res.get('per_prog', {})

        mtab1, mtab2, mtab3, mtab4, mtab5 = st.tabs([
            "📊 Performa Model", "📋 Tabel Detail", "💡 Kesimpulan",
            "🔮 Prophet + Kalender", "🗺️ Efek Hari Libur per Program",
        ])

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
                cards_html = '<div style="display:flex;flex-wrap:wrap;gap:10px;margin:12px 0;">'
                for _, row in bpp.iterrows():
                    mape_v   = row.get('MAPE (%)', None)
                    pill_cls = ("mpill-green" if mape_v and mape_v < 10 else
                                "mpill-blue"  if mape_v and mape_v < 20 else
                                "mpill-yellow" if mape_v and mape_v < 50 else "mpill-red")
                    mape_txt = f"MAPE: {mape_v:.1f}%" if mape_v else "–"
                    cards_html += (f'<div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:12px 14px;min-width:140px;">'
                                   f'<div style="font-size:.65rem;color:#64748b;font-weight:700;text-transform:uppercase;">{row["Program"]}</div>'
                                   f'<div style="font-size:.88rem;font-weight:600;color:#0f172a;margin:4px 0;">{row["Model"]}</div>'
                                   f'<span class="mpill {pill_cls}">{mape_txt}</span></div>')
                cards_html += '</div>'
                st.markdown(cards_html, unsafe_allow_html=True)
                bpp_plot = bpp.dropna(subset=['MAPE (%)'])
                if not bpp_plot.empty:
                    fig_mp = px.bar(bpp_plot, x='Program', y='MAPE (%)', color='Model',
                                    color_discrete_sequence=COLORS,
                                    title='MAPE % per Program (lebih rendah = lebih baik)')
                    fig_mp.add_hline(y=20, line_dash='dash', line_color='#fbbf24', annotation_text='Threshold 20%')
                    fig_mp.update_layout(**DARK, height=360, margin=dict(b=60, t=40))
                    st.plotly_chart(fig_mp, width='stretch')

        with mtab2:
            det = ml_res.get('detail', pd.DataFrame())
            if bpp.empty:
                st.info("Klik **Jalankan Analisis ML** untuk melihat hasil.")
            else:
                st.markdown('<div class="sec">Model Terbaik per Program</div>', unsafe_allow_html=True)
                st.dataframe(bpp.style.format({'R2': '{:.4f}', 'MAPE (%)': '{:.2f}', 'MAE': '{:,.0f}', 'RMSE': '{:,.0f}'}),
                             width='stretch', height=260)
                if not det.empty:
                    with st.expander("📋 Tabel Detail Semua Model × Semua Program"):
                        st.dataframe(det.sort_values(['Program','R2'], ascending=[True,False])
                                       .style.format({'R2': '{:.4f}', 'MAPE (%)': '{:.2f}', 'MAE': '{:,.0f}', 'RMSE': '{:,.0f}'}),
                                     width='stretch', height=400)

        with mtab3:
            conclusions = build_conclusion(ml_res, ml_res, df, target_ml, n_future)
            if conclusions:
                st.markdown('<div class="sec">Kesimpulan & Rekomendasi Otomatis</div>', unsafe_allow_html=True)
                for icon, title, text in conclusions:
                    st.markdown(f"""<div style="background:#fff;border:1px solid #e2e8f0;border-radius:14px;
                                padding:18px 22px;margin:10px 0;border-left:3px solid #3b82f6;">
                        <div style="font-size:.65rem;font-weight:700;color:#334155;text-transform:uppercase;
                                    letter-spacing:2px;margin-bottom:8px;">{icon} {title}</div>
                        <div style="color:#334155;font-size:.9rem;line-height:1.75;">{text}</div>
                    </div>""", unsafe_allow_html=True)
            else:
                st.info("Jalankan analisis ML untuk mendapatkan kesimpulan otomatis.")

        with mtab4:
            df_raw_m_p = st.session_state.get('raw_monthly', None)
            if not PROPHET_OK:
                st.warning("**Prophet belum terinstall.**")
            elif df_raw_m_p is None or len(df_raw_m_p) == 0:
                st.warning("Upload dataset bulanan terlebih dahulu.")
            else:
                n_holidays = len(INDONESIAN_HOLIDAYS)
                n_htypes   = INDONESIAN_HOLIDAYS['holiday'].nunique() if n_holidays > 0 else 0
                gcal_status = (f"✅ <b>{n_holidays} hari libur</b>, <b>{n_htypes} jenis</b> dari Google Calendar."
                               if n_holidays > 0 else "⚠️ <b>GCAL_KEY belum diset</b>.")
                st.markdown(f'<div class="info-box">🔮 <b>Prophet + Kalender Indonesia</b><br>{gcal_status}</div>',
                            unsafe_allow_html=True)
                target_prophet = target_ml
                st.markdown(f'<div class="info-box" style="margin-bottom:6px;">🎯 Target terkunci: <b>{target_prophet}</b></div>',
                            unsafe_allow_html=True)
                n_months_prophet = st.slider("Prediksi (bulan)", 6, 36, 12, 6)
                use_holidays = st.checkbox(f"Gunakan kalender ({n_holidays} hari libur)", value=(n_holidays > 0))
                _prophet_progs = filtered_progs if filtered_progs else active_progs
                _btn_lbl = (f"🔮 Jalankan Prophet — {_prophet_progs[0]}" if len(_prophet_progs) == 1
                            else f"🔮 Jalankan Prophet ({len(_prophet_progs)} Program)")

                with st.expander("🔍 Cek Data Bulanan", expanded=False):
                    _debug_rows = []
                    for _cp in _prophet_progs:
                        _cp_data = df_raw_m_p[df_raw_m_p['Kategori'] == _cp] if 'Kategori' in df_raw_m_p.columns else pd.DataFrame()
                        _has_target = target_prophet in _cp_data.columns if not _cp_data.empty else False
                        _valid_rows = int(_cp_data[target_prophet].notna().sum()) if _has_target else 0
                        _debug_rows.append({'Program': _cp, 'Total Baris': len(_cp_data),
                                            f'Valid ({target_prophet})': _valid_rows,
                                            'Status': '✅' if _valid_rows >= 6 else f'⚠️ ({_valid_rows})'})
                    if _debug_rows:
                        st.dataframe(pd.DataFrame(_debug_rows), width='stretch')

                if st.button(_btn_lbl, type="primary", width='stretch'):
                    all_p_results, prog_errors = {}, {}
                    with st.spinner(f"Melatih Prophet untuk {len(_prophet_progs)} program..."):
                        for cp in _prophet_progs:
                            pr, pe = run_prophet(df_raw_m_p, target_prophet, cp, n_months_prophet, use_holidays)
                            if pe: prog_errors[cp] = pe
                            else:  all_p_results[cp] = pr
                    for cp, pe in prog_errors.items():
                        st.warning(f"⚠️ {cp}: {pe}")
                    st.session_state['prophet_results']       = all_p_results
                    st.session_state['prophet_result']        = all_p_results
                    st.session_state['prophet_result_target'] = target_prophet

                _all_p = st.session_state.get('prophet_results', {})
                if _all_p:
                    _auto_tgt = st.session_state.get('prophet_auto_target', '')
                    if _auto_tgt:
                        st.markdown(f'<div class="info-box">✅ Auto-run selesai (target: <b>{_auto_tgt}</b>).</div>',
                                    unsafe_allow_html=True)

                if _all_p:
                    for cp, pr in _all_p.items():
                        fc      = pr['forecast']
                        hist_pr = pr['history']
                        col_p   = COLORS[list(_all_p.keys()).index(cp) % len(COLORS)]
                        st.markdown(f'<div class="sec">Hasil Prophet — {cp}</div>', unsafe_allow_html=True)
                        r2_is   = pr.get('r2_insample', None)
                        mape_is = pr.get('mape_insample', None)
                        st.markdown(
                            f'<div class="info-box">📊 <b>{cp}</b> — '
                            f'R² = <b>{"N/A" if r2_is is None else f"{r2_is:.4f}"}</b> | '
                            f'MAPE = <b>{"N/A" if mape_is is None else f"{mape_is:.2f}%"}</b> | '
                            f'Data: <b>{pr.get("n_data","?")} bln</b></div>',
                            unsafe_allow_html=True)

                        future_only = fc[~fc['ds'].isin(hist_pr['ds'])]
                        all_fc      = fc[fc['ds'].isin(hist_pr['ds'])]
                        fig_p = go.Figure()
                        fig_p.add_trace(go.Scatter(
                            x=list(future_only['ds'])+list(future_only['ds'][::-1]),
                            y=list(future_only['yhat_upper'])+list(future_only['yhat_lower'][::-1]),
                            fill='toself', fillcolor=hex_to_rgba(col_p,0.12),
                            line=dict(color='rgba(0,0,0,0)'), showlegend=False))
                        fig_p.add_trace(go.Scatter(x=all_fc['ds'], y=all_fc['yhat'], name='Fitted',
                            mode='lines', line=dict(color=hex_to_rgba(col_p,0.5), width=1.5, dash='dot')))
                        fig_p.add_trace(go.Scatter(x=hist_pr['ds'], y=hist_pr['y'], name='Aktual',
                            mode='lines+markers', line=dict(color=col_p, width=2.5), marker=dict(size=6)))
                        fig_p.add_trace(go.Scatter(x=future_only['ds'], y=future_only['yhat'],
                            name='Prediksi', mode='lines+markers',
                            line=dict(color='#2563eb', width=2.5, dash='dash'), marker=dict(size=7, symbol='diamond')))
                        if len(hist_pr) > 0:
                            fig_p.add_vline(x=hist_pr['ds'].max(), line_dash='dot',
                                line_color=hex_to_rgba('#60a5fa',0.5), line_width=1.5)
                        fig_p.update_layout(**DARK, height=380, xaxis_tickangle=-30,
                            margin=dict(t=30,b=60,l=70,r=20),
                            legend=dict(orientation='h', y=-0.25),
                            yaxis_title=target_prophet, hovermode='x unified')
                        st.plotly_chart(fig_p, width='stretch')

                        # Komponen
                        st.markdown(f'<div class="sec" style="font-size:.78rem;">📈 Komponen — {cp}</div>', unsafe_allow_html=True)
                        comp_labels = ['📉 Trend']
                        if 'yearly'  in fc.columns: comp_labels.append('📅 Musiman Tahunan')
                        if 'monthly' in fc.columns: comp_labels.append('🗓️ Musiman Bulanan')
                        has_hol = pr.get('n_holidays', 0) > 0
                        comp_tabs = st.tabs(comp_labels + (['🎌 Efek Hari Libur (detail)'] if has_hol else []))
                        cidx = 0
                        with comp_tabs[cidx]:
                            fig_tr = go.Figure()
                            fig_tr.add_trace(go.Scatter(x=fc['ds'], y=fc['trend'], mode='lines',
                                name='Trend', line=dict(color='#f59e0b', width=2.5)))
                            if 'trend_lower' in fc.columns:
                                fig_tr.add_trace(go.Scatter(
                                    x=list(fc['ds'])+list(fc['ds'][::-1]),
                                    y=list(fc['trend_upper'])+list(fc['trend_lower'][::-1]),
                                    fill='toself', fillcolor='rgba(245,158,11,0.10)',
                                    line=dict(color='rgba(0,0,0,0)'), showlegend=False))
                            fig_tr.update_layout(**DARK, height=260, margin=dict(t=10,b=50,l=70,r=20),
                                yaxis_title='Trend', xaxis_tickangle=-30)
                            st.plotly_chart(fig_tr, width='stretch')
                        cidx += 1
                        if 'yearly' in fc.columns and cidx < len(comp_tabs):
                            with comp_tabs[cidx]:
                                _fh = fc[fc['ds'].isin(hist_pr['ds'])].copy()
                                _fh['bulan'] = _fh['ds'].dt.month
                                _mo = _fh.groupby('bulan')['yearly'].mean().reset_index()
                                _mo.columns = ['Bulan', 'Efek']
                                _bn = ['Jan','Feb','Mar','Apr','Mei','Jun','Jul','Agu','Sep','Okt','Nov','Des']
                                _mo['Nama'] = _mo['Bulan'].apply(lambda x: _bn[x-1])
                                fig_yr = go.Figure(go.Bar(x=_mo['Nama'], y=_mo['Efek'],
                                    marker_color=['#10b981' if v >= 0 else '#ef4444' for v in _mo['Efek']],
                                    text=_mo['Efek'].round(2), textposition='outside'))
                                fig_yr.add_hline(y=0, line_dash='dot', line_color='rgba(255,255,255,0.3)')
                                fig_yr.update_layout(**DARK, height=260, margin=dict(t=10,b=40,l=70,r=20),
                                    yaxis_title='Efek Musiman', xaxis_title='Bulan')
                                st.plotly_chart(fig_yr, width='stretch')
                            cidx += 1
                        if 'monthly' in fc.columns and cidx < len(comp_tabs):
                            with comp_tabs[cidx]:
                                _mc = fc[fc['ds'].isin(hist_pr['ds'])][['ds','monthly']].copy()
                                fig_ms = go.Figure(go.Scatter(x=_mc['ds'], y=_mc['monthly'],
                                    mode='lines', line=dict(color='#a78bfa', width=2)))
                                fig_ms.add_hline(y=0, line_dash='dot', line_color='rgba(255,255,255,0.3)')
                                fig_ms.update_layout(**DARK, height=240, margin=dict(t=10,b=50,l=70,r=20),
                                    yaxis_title='Efek Bulanan', xaxis_tickangle=-30)
                                st.plotly_chart(fig_ms, width='stretch')
                            cidx += 1
                        if has_hol and cidx < len(comp_tabs):
                            with comp_tabs[cidx]:
                                h_eff = pr.get('holiday_effects', {})
                                fc_h  = pr['forecast']
                                if 'holidays' in fc_h.columns:
                                    st.markdown(f"**Efek gabungan hari libur per bulan — {cp}:**")
                                    _fhh = fc_h[fc_h['ds'].isin(pr['history']['ds'])].copy()
                                    _fhh['Bulan'] = _fhh['ds'].dt.strftime('%Y-%m')
                                    _hv = _fhh['holidays'].values
                                    fig_ht = go.Figure(go.Bar(x=_fhh['Bulan'], y=_fhh['holidays'],
                                        marker_color=['#10b981' if v >= 0 else '#ef4444' for v in _hv],
                                        text=[f"{v:+.2f}" for v in _hv], textposition='outside',
                                        hovertemplate='%{x}<br>Efek: %{y:+.3f}<extra></extra>'))
                                    fig_ht.add_hline(y=0, line_dash='dot', line_color='rgba(255,255,255,0.4)')
                                    fig_ht.update_layout(**DARK, height=300, xaxis_tickangle=-45,
                                        margin=dict(t=20,b=80,l=60,r=20),
                                        yaxis_title=f'Efek pada {target_prophet}')
                                    st.plotly_chart(fig_ht, width='stretch')
                                    st.caption("Hijau = hari libur meningkatkan klaim | Merah = menurunkan")
                                    clean_eff = {k: v for k,v in h_eff.items()
                                                 if k != '_error' and isinstance(v,dict) and v.get('mean_abs_effect',0) > 1e-6}
                                    if clean_eff:
                                        st.markdown(f"**Top {min(15,len(clean_eff))} hari libur spesifik:**")
                                        top_h = list(clean_eff.items())[:15]
                                        fig_h2 = go.Figure()
                                        fig_h2.add_trace(go.Bar(
                                            y=[h[0][:40] for h in top_h],
                                            x=[h[1]['mean_abs_effect'] for h in top_h],
                                            orientation='h', marker_color='#3b82f6',
                                            text=[f"{h[1]['mean_abs_effect']:.3f}" for h in top_h],
                                            textposition='outside'))
                                        fig_h2.update_layout(**DARK,
                                            height=max(280, len(top_h)*30+80),
                                            margin=dict(t=10,b=40,l=260,r=80),
                                            xaxis_title='Magnitude efek')
                                        st.plotly_chart(fig_h2, width='stretch')
                                    with st.expander("📋 Tabel efek lengkap"):
                                        if clean_eff:
                                            st.dataframe(pd.DataFrame([
                                                {'Hari Libur': k,
                                                 'Efek Rata-rata (abs)': round(v['mean_abs_effect'],4),
                                                 'Efek Maks': round(v['max_effect'],4),
                                                 'Efek Min': round(v['min_effect'],4)}
                                                for k,v in clean_eff.items()
                                            ]), width='stretch')

        # ══════════════════════════════════════════════════════════════════════
        # SUB-TAB 5: EFEK HARI LIBUR PER PROGRAM (BARU)
        # ══════════════════════════════════════════════════════════════════════
        with mtab5:
            _all_p = st.session_state.get('prophet_results', {})

            if not _all_p:
                st.info("Jalankan **Analisis ML** terlebih dahulu. Hasil efek hari libur per program akan muncul di sini.")
                st.stop()

            has_any = any(
                bool({k:v for k,v in pr.get('holiday_effects',{}).items()
                      if k != '_error' and isinstance(v,dict) and v.get('mean_abs_effect',0) > 1e-6})
                for pr in _all_p.values()
            )
            if not has_any:
                st.warning("Prophet sudah jalan tapi efek hari libur tidak terdeteksi. Pastikan GCAL_KEY sudah diset.")
                st.stop()

            st.markdown(
                '<div class="info-box">Analisis ini menunjukkan <b>hari libur apa yang paling berefek '
                'pada masing-masing program</b>. Satu hari libur bisa berefek berbeda di tiap program — '
                'misalnya Idul Fitri mungkin berefek besar ke JHT tapi kecil ke JKK. '
                'Nilai menunjukkan perubahan relatif jumlah klaim di bulan yang mengandung hari libur tersebut.</div>',
                unsafe_allow_html=True
            )

            top_per_prog          = _build_top_holidays_per_program(_all_p, top_n=8)
            df_abs, df_signed     = _build_holiday_program_matrix(_all_p)
            prog_list_sorted      = sorted(top_per_prog.keys())

            # ── Bagian 1: Card per program ────────────────────────────────────
            st.markdown('<div class="sec">Top Hari Libur Paling Berefek — per Program</div>', unsafe_allow_html=True)
            st.caption("Angka menunjukkan seberapa besar hari libur itu mengubah klaim program tersebut. ▲ = menaikkan, ▼ = menurunkan.")

            cols_per_row = 2
            for row_i in range(0, len(prog_list_sorted), cols_per_row):
                row_progs = prog_list_sorted[row_i:row_i + cols_per_row]
                cols = st.columns(len(row_progs))
                for col_j, prog in enumerate(row_progs):
                    prog_effects = top_per_prog[prog]
                    with cols[col_j]:
                        card_inner = ""
                        if not prog_effects:
                            card_inner = '<div style="color:#94a3b8;font-size:.82rem;padding:8px 0;">Tidak ada efek terdeteksi.</div>'
                        else:
                            max_mag = prog_effects[0]['mean_abs'] if prog_effects else 1
                            for rank, eff in enumerate(prog_effects, 1):
                                h_name    = eff['holiday'][:42]
                                mag       = eff['mean_abs']
                                dominant  = eff['dominant']
                                direction = eff['direction']
                                arrow     = "▲" if direction == "naik" else "▼"
                                color     = "#10b981" if direction == "naik" else "#ef4444"
                                bar_w     = int(mag / (max_mag + 1e-9) * 100)
                                # Konversi efek ke persen untuk lebih intuitif
                                pct_label = f"{abs(dominant)*100:.1f}%" if abs(dominant) < 10 else f"{abs(dominant):.2f}"
                                card_inner += (
                                    f'<div style="margin-bottom:10px;">'
                                    f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:3px;">'
                                    f'<span style="font-size:.75rem;color:#334155;font-weight:500;max-width:75%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">'
                                    f'{rank}. {h_name}</span>'
                                    f'<span style="font-size:.8rem;font-weight:700;color:{color};white-space:nowrap;margin-left:6px;">'
                                    f'{arrow} {pct_label}</span>'
                                    f'</div>'
                                    f'<div style="background:#f1f5f9;border-radius:4px;height:6px;">'
                                    f'<div style="background:{color};width:{bar_w}%;height:6px;border-radius:4px;"></div>'
                                    f'</div>'
                                    f'</div>'
                                )
                            card_inner += '<div style="font-size:.67rem;color:#94a3b8;margin-top:6px;border-top:1px solid #f1f5f9;padding-top:6px;">▲ klaim naik | ▼ klaim turun saat hari libur ini</div>'

                        st.markdown(
                            f'<div style="background:#fff;border:1px solid #e2e8f0;border-radius:14px;'
                            f'padding:18px;margin-bottom:14px;border-top:3px solid #3b82f6;">'
                            f'<div style="font-weight:800;color:#0f172a;font-size:.9rem;margin-bottom:12px;">'
                            f'🏷️ {prog}</div>'
                            f'{card_inner}</div>',
                            unsafe_allow_html=True
                        )

            # ── Bagian 2: Heatmap matrix ──────────────────────────────────────
            st.markdown('<div class="sec" style="margin-top:20px;">Heatmap — Magnitude Efek Hari Libur × Program</div>',
                        unsafe_allow_html=True)
            st.caption("Semakin gelap = efek semakin besar. Nilai = mean absolute effect dari model Prophet.")

            if not df_abs.empty:
                df_heat    = df_abs.head(20)
                progs_heat = df_heat.columns.tolist()
                fig_heat = go.Figure(go.Heatmap(
                    z=df_heat.values,
                    x=progs_heat,
                    y=[h[:50] for h in df_heat.index.tolist()],
                    colorscale='Blues',
                    text=[[f"{v:.3f}" for v in row] for row in df_heat.values],
                    texttemplate="%{text}",
                    textfont=dict(size=10),
                    hovertemplate="Hari Libur: %{y}<br>Program: %{x}<br>Efek: %{z:.4f}<extra></extra>",
                    colorbar=dict(title="Magnitude"),
                ))
                fig_heat.update_layout(
                    **DARK,
                    height=max(420, len(df_heat) * 30 + 120),
                    margin=dict(t=20, b=60, l=300, r=40),
                    xaxis=dict(side='top', tickfont=dict(size=13)),
                    yaxis=dict(tickfont=dict(size=10)),
                )
                st.plotly_chart(fig_heat, width='stretch')

            # ── Bagian 3: Grouped bar arah efek ──────────────────────────────
            st.markdown('<div class="sec" style="margin-top:20px;">Arah Efek — Positif vs Negatif per Program</div>',
                        unsafe_allow_html=True)
            st.caption("▲ Positif = hari libur cenderung menaikkan klaim program itu | ▼ Negatif = menurunkan")

            if not df_signed.empty:
                df_chart    = df_signed.head(12)
                progs_chart = df_chart.columns.tolist()
                fig_dir = go.Figure()
                for i, prog in enumerate(progs_chart):
                    vals = df_chart[prog].tolist()
                    fig_dir.add_trace(go.Bar(
                        name=prog,
                        x=[h[:38] for h in df_chart.index.tolist()],
                        y=vals,
                        marker_color=COLORS[i % len(COLORS)],
                        text=[f"{v:+.3f}" for v in vals],
                        textposition='outside',
                    ))
                fig_dir.add_hline(y=0, line_dash='dot', line_color='rgba(255,255,255,0.5)', line_width=1.5)
                fig_dir.update_layout(
                    **DARK, barmode='group', height=460,
                    margin=dict(t=20, b=140, l=60, r=20),
                    legend=dict(orientation='h', y=-0.42),
                    xaxis=dict(tickangle=-40, tickfont=dict(size=9)),
                    yaxis_title='Efek (+ = klaim naik, - = klaim turun)',
                )
                st.plotly_chart(fig_dir, width='stretch')

            # ── Bagian 4: Tabel ekspor lengkap ───────────────────────────────
            with st.expander("📋 Tabel Lengkap — Semua Hari Libur × Semua Program"):
                if not df_abs.empty:
                    df_tbl = df_abs.copy()
                    df_tbl['Program Terbesar Efek'] = df_tbl.idxmax(axis=1)
                    df_tbl['Efek Maks (abs)']       = df_tbl[df_abs.columns.tolist()].max(axis=1).round(4)
                    st.dataframe(
                        df_tbl.style.format({c: '{:.4f}' for c in df_abs.columns}),
                        width='stretch', height=420
                    )

    else:
        st.info("Klik **Jalankan Analisis ML** untuk memulai.")