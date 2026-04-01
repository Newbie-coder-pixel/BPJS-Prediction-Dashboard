"""tabs/tab_prediksi.py — Tab 3: Prediksi tahunan & bulanan."""
import hashlib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime


def render_tab_prediksi(df, df_plot, active_progs, filtered_progs, targets,
                        n_lags, test_pct, n_future, years, latest_year,
                        single_yr, has_nom, results_cache,
                        COLORS, DARK, styled_chart, hex_to_rgba,
                        run_ml, forecast, compute_monthly_breakdown,
                        add_to_history, df_raw_monthly):

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
                st.error(err); ml_pred = None
            else:
                results_cache[ck_p] = ml_pred
                st.session_state.active_results = results_cache

    fut = fut_monthly = None

    if ml_pred:
        with st.spinner("Menghitung proyeksi..."):
            fut = forecast(df, ml_pred, n_future)

        df_raw_m = st.session_state.get('raw_monthly', None)
        if df_raw_m is not None and len(df_raw_m) > 0 and target_pred in df_raw_m.columns:
            try:
                fut_monthly = compute_monthly_breakdown(df_raw_m, fut, target_pred)
            except Exception as e:
                st.warning(f"Gagal hitung prediksi bulanan: {e}")
                fut_monthly = None

        st.session_state['last_forecast']                    = fut
        st.session_state['last_forecast_monthly']            = fut_monthly
        st.session_state[f'forecast_{target_pred}']         = fut
        st.session_state[f'forecast_monthly_{target_pred}'] = fut_monthly
        st.session_state[f'forecast_annual_{target_pred}']  = fut

        # ── Simpan ml_pred & target agar AI Analyst bisa akses hasil prediksi ──
        st.session_state['ml_result']            = ml_pred
        st.session_state['ml_result_target']     = target_pred
        st.session_state['ml_forecast_ready']    = True   # flag untuk invalidasi cache AI

        data_hash  = hashlib.md5(df.to_csv().encode()).hexdigest()[:8]
        eid_pred   = f"{data_hash}_{target_pred}_L{n_lags}_T{test_pct}"
        label_pred = f"📁 {datetime.now().strftime('%d/%m %H:%M')} | {target_pred} | {len(years)}yr"
        extra_snapshot = {k: st.session_state[k]
                          for k in ['raw_monthly','forecast_Kasus','forecast_Nominal',
                                    'forecast_monthly_Kasus','forecast_monthly_Nominal',
                                    'forecast_annual_Kasus','forecast_annual_Nominal',
                                    'last_forecast','last_forecast_monthly']
                          if k in st.session_state and st.session_state[k] is not None}
        add_to_history(label_pred, eid_pred, df.copy(), dict(results_cache), extra_snapshot)

        future_yrs = sorted(fut['Tahun'].unique())
        yr_range   = f"{future_yrs[0]}-{future_yrs[-1]}"

        per_prog_info = ml_pred.get('per_prog', {})
        if per_prog_info:
            model_parts = " | ".join(f"<b>{cat}</b>→{info['best_name']}"
                                     for cat, info in per_prog_info.items())
            badge_html = (f'<div class="badge">🎯 <b>Per-Program Model</b>: {model_parts}'
                          f' &nbsp;|&nbsp; Proyeksi <b>{n_future} tahun</b> ({yr_range})</div>')
        else:
            badge_html = (f'<div class="badge">Model: <b>{ml_pred["best_name"]}</b>'
                          f' &nbsp;|&nbsp; Proyeksi <b>{n_future} tahun</b> ({yr_range})</div>')
        st.markdown(badge_html, unsafe_allow_html=True)

        ptab_yr, ptab_mo = st.tabs(["📅 Prediksi Tahunan", "📆 Prediksi Bulanan"])

        with ptab_yr:
            hist = df_plot.groupby(['Tahun','Kategori'])[target_pred].sum().reset_index()
            fut_yr = fut.copy(); fut_yr['Jenis'] = 'Prediksi'

            st.markdown('<div class="sec">Tren Historis (Aktual) vs Prediksi</div>', unsafe_allow_html=True)
            fig_main = go.Figure()
            cat_color = {c: COLORS[i % len(COLORS)] for i, c in enumerate(sorted(hist['Kategori'].unique()))}
            per_prog_bt = ml_pred.get('per_prog', {})

            for cat in sorted(hist['Kategori'].unique()):
                col = cat_color[cat]
                h = hist[hist['Kategori']==cat].sort_values('Tahun')
                if len(h):
                    fig_main.add_trace(go.Scatter(
                        x=h['Tahun'], y=h[target_pred], name=cat+" (Aktual)",
                        mode='lines+markers', line=dict(color=col, width=2.5),
                        marker=dict(size=8, symbol='circle'), legendgroup=cat))

                    if cat in per_prog_bt and len(h) >= 3:
                        try:
                            y_arr  = h[target_pred].values.astype(float)
                            yr_arr = h['Tahun'].values
                            best_model_name = per_prog_bt[cat].get('best_name', 'Weighted MA')
                            fitted_vals, fitted_years, fitted_errors = [], [], []
                            for i in range(1, len(y_arr)):
                                history = y_arr[:i]; actual = y_arr[i]; yr = int(yr_arr[i])
                                if 'Exp' in best_model_name or 'SES' in best_model_name:
                                    alpha = 0.6; s = history[0]
                                    for v in history[1:]: s = alpha * v + (1-alpha) * s
                                    pred = s
                                elif 'Holt' in best_model_name and len(history) >= 2:
                                    alpha, beta = 0.6, 0.3; l, b = history[0], history[1]-history[0]
                                    for v in history[1:]:
                                        l_new = alpha*v + (1-alpha)*(l+b); b = beta*(l_new-l)+(1-beta)*b; l = l_new
                                    pred = l + b
                                else:
                                    w_len = min(len(history), 3); window = history[-w_len:]
                                    weights = list(range(1, w_len+1))
                                    pred = float(np.average(window, weights=weights))
                                err_pct = abs(pred-actual)/(abs(actual)+1e-9)*100
                                fitted_vals.append(pred); fitted_years.append(yr); fitted_errors.append(err_pct)
                            if fitted_vals:
                                fig_main.add_trace(go.Scatter(
                                    x=fitted_years, y=fitted_vals, name=cat+" (Prediksi Historis)",
                                    mode='lines+markers', line=dict(color=col, width=1.8, dash='dot'),
                                    marker=dict(size=7, symbol='diamond-open', color=col),
                                    legendgroup=cat))
                        except Exception:
                            pass

                p = fut_yr[fut_yr['Kategori']==cat].sort_values('Tahun')
                if len(p):
                    x_p = list(p['Tahun']); y_p = list(p[target_pred])
                    upper_col = f'{target_pred}_upper'; lower_col = f'{target_pred}_lower'
                    has_ci = upper_col in p.columns and lower_col in p.columns
                    if len(h):
                        last_h = h.sort_values('Tahun').iloc[-1]
                        x_p = [int(last_h['Tahun'])] + x_p; y_p = [float(last_h[target_pred])] + y_p
                        if has_ci:
                            y_upper = [float(last_h[target_pred])] + list(p[upper_col])
                            y_lower = [float(last_h[target_pred])] + list(p[lower_col])
                    elif has_ci:
                        y_upper = list(p[upper_col]); y_lower = list(p[lower_col])
                    if has_ci:
                        fig_main.add_trace(go.Scatter(
                            x=x_p+x_p[::-1], y=y_upper+y_lower[::-1],
                            fill='toself', fillcolor=hex_to_rgba(col, 0.12),
                            line=dict(color='rgba(0,0,0,0)'), hoverinfo='skip',
                            showlegend=False, legendgroup=cat))
                    fig_main.add_trace(go.Scatter(
                        x=x_p, y=y_p, name=cat+" (Prediksi)",
                        mode='lines+markers', line=dict(color=col, width=2.5, dash='dash'),
                        marker=dict(size=10, symbol='diamond'), legendgroup=cat))

            if future_yrs:
                fig_main.add_vrect(x0=latest_year+0.3, x1=future_yrs[-1]+0.7,
                    fillcolor=hex_to_rgba('#3b82f6', 0.05), line_width=0,
                    annotation_text="▶ Zona Prediksi",
                    annotation_font=dict(color='#60a5fa', size=11))
                fig_main.add_vline(x=latest_year+0.5, line_dash='dot',
                    line_color=hex_to_rgba('#60a5fa', 0.6), line_width=1.5)

            styled_chart(fig_main, height=540, legend_bottom=True, margin_b=140)
            fig_main.update_layout(xaxis=dict(dtick=1), yaxis_title=target_pred)
            st.plotly_chart(fig_main, width='stretch')

            # Tabel CI
            upper_col = f'{target_pred}_upper'; lower_col = f'{target_pred}_lower'; ci_col = f'{target_pred}_ci_pct'
            has_ci = all(c in fut.columns for c in [upper_col, lower_col, ci_col])
            if has_ci:
                tbl = fut[['Tahun','Kategori',target_pred,upper_col,lower_col,ci_col]].copy()
                tbl.columns = ['Tahun','Program','Prediksi','Batas Atas','Batas Bawah','CI (%)']
                tbl['Prediksi']    = tbl['Prediksi'].round(0).astype(int)
                tbl['Batas Atas']  = tbl['Batas Atas'].round(0).astype(int)
                tbl['Batas Bawah'] = tbl['Batas Bawah'].round(0).astype(int)
                tbl['CI (%)']      = tbl['CI (%)'].round(1).astype(str) + '%'
                tbl['Rentang']     = tbl.apply(lambda r: f"{int(r['Batas Bawah']):,}  ↔  {int(r['Batas Atas']):,}", axis=1)
                st.dataframe(tbl[['Tahun','Program','Prediksi','Rentang','CI (%)']].sort_values(['Tahun','Program']),
                             width='stretch')

            # Bar prediksi
            fig_bar = px.bar(fut, x='Tahun', y=target_pred, color='Kategori',
                             barmode='group', color_discrete_sequence=COLORS, text=target_pred)
            fig_bar.update_traces(texttemplate='%{text:,.0f}', textposition='outside', marker_line_width=0)
            styled_chart(fig_bar, height=420)
            st.plotly_chart(fig_bar, width='stretch')

            # Tabel prediksi
            st.markdown('<div class="sec">Tabel Prediksi Tahunan</div>', unsafe_allow_html=True)
            tbl2 = fut[['Kategori','Tahun',target_pred]].copy().sort_values(['Kategori','Tahun'])
            fmt  = 'Rp {:,.0f}' if target_pred == 'Nominal' else '{:,.0f}'
            tbl2[target_pred] = tbl2[target_pred].apply(lambda x: fmt.format(x))
            st.dataframe(tbl2, width='stretch')

        with ptab_mo:
            if fut_monthly is not None and len(fut_monthly) > 0:
                st.markdown('<div class="info-box">Prediksi bulanan dihitung dengan mendistribusikan total tahunan menggunakan pola musiman dari data historis.</div>', unsafe_allow_html=True)
                st.markdown('<div class="sec">Tren Bulanan: Aktual vs Prediksi</div>', unsafe_allow_html=True)

                df_raw_m2 = st.session_state.get('raw_monthly', None)
                fig_mo = go.Figure()
                _mo_progs = [p for p in sorted(ml_pred['active_programs']) if p in filtered_progs] or sorted(ml_pred['active_programs'])

                for i, cat in enumerate(_mo_progs):
                    col = COLORS[i % len(COLORS)]
                    if df_raw_m2 is not None and target_pred in df_raw_m2.columns:
                        cat_raw = df_raw_m2[df_raw_m2['Kategori']==cat].sort_values(['Tahun','Bulan']).copy()
                        if len(cat_raw):
                            cat_raw['Periode'] = cat_raw['Tahun'].astype(str)+'-'+cat_raw['Bulan'].astype(str).str.zfill(2)
                            fig_mo.add_trace(go.Scatter(x=cat_raw['Periode'], y=cat_raw[target_pred],
                                name=cat+" (Aktual)", mode='lines+markers', line=dict(color=col, width=2),
                                marker=dict(size=5), legendgroup=cat))
                    cat_pred_mo = fut_monthly[fut_monthly['Kategori']==cat].sort_values(['Tahun','Bulan'])
                    if len(cat_pred_mo):
                        fig_mo.add_trace(go.Scatter(x=cat_pred_mo['Periode'], y=cat_pred_mo[target_pred],
                            name=cat+" (Prediksi)", mode='lines+markers', line=dict(color=col, width=2, dash='dash'),
                            marker=dict(size=7, symbol='diamond'), legendgroup=cat))

                fig_mo.update_layout(**DARK, height=520, hovermode='x unified', xaxis_tickangle=-45,
                    legend=dict(orientation='h', y=-0.35, font=dict(size=11)),
                    margin=dict(b=150, t=20, l=70, r=20), yaxis_title=target_pred)
                st.plotly_chart(fig_mo, width='stretch')

                st.markdown('<div class="sec">Tabel Prediksi Bulanan Lengkap</div>', unsafe_allow_html=True)
                tbl_mo = fut_monthly[['Kategori','Tahun','Bulan','Periode',target_pred]].copy().sort_values(['Kategori','Tahun','Bulan'])
                fmt = 'Rp {:,.0f}' if target_pred == 'Nominal' else '{:,.0f}'
                tbl_mo[target_pred] = tbl_mo[target_pred].apply(lambda x: fmt.format(x))
                st.dataframe(tbl_mo, width='stretch', height=420)
            else:
                st.warning("Data bulanan tidak tersedia. Upload dataset dengan data bulanan atau klik Hitung Prediksi ulang.")
    else:
        st.info("Klik **Hitung Prediksi** — model ML akan dilatih otomatis jika belum ada.")