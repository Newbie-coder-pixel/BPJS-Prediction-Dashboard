"""tabs/tab_overview.py — Tab 1: Overview charts & insights."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render_tab_overview(df, df_plot, df_active_only, active_progs, filtered_progs,
                        years, latest_year, prev_year, has_nom, single_yr,
                        prog_changes, n_lags, n_future, COLORS, DARK,
                        styled_chart, hex_to_rgba,
                        _fetch_worldbank, _ai_analyze_peak_trough, _get_api_key):

    df_lat = df_plot[df_plot['Tahun'] == latest_year]

    if not single_yr:
        top_prog = df_lat.groupby('Kategori')['Kasus'].sum().idxmax()
        growth_by_prog = {}
        growth_prev_yr = {}
        for cp in active_progs:
            cd = df_plot[df_plot['Kategori']==cp].sort_values('Tahun')
            if len(cd) >= 2 and cd['Kasus'].iloc[-2] > 0:
                growth_by_prog[cp] = (cd['Kasus'].iloc[-1]/cd['Kasus'].iloc[-2]-1)*100
                growth_prev_yr[cp] = int(cd.iloc[-2]['Tahun'])
        fastest  = max(growth_by_prog, key=growth_by_prog.get) if growth_by_prog else "-"
        fastest_g = growth_by_prog.get(fastest, 0)
        slowest  = min(growth_by_prog, key=growth_by_prog.get) if growth_by_prog else "-"
        slowest_g = growth_by_prog.get(slowest, 0)
        total_latest = int(df_lat['Kasus'].sum())

        ia, ib, ic, id_ = st.columns(4)
        with ia:
            st.markdown(f'<div class="insight-card"><div class="ic-title">🏆 Program Terbesar</div>'
                        f'<div class="ic-val">{top_prog}</div>'
                        f'<div class="ic-sub">{int(df_lat[df_lat["Kategori"]==top_prog]["Kasus"].sum()):,} kasus {latest_year}</div></div>',
                        unsafe_allow_html=True)
        with ib:
            st.markdown(f'<div class="insight-card"><div class="ic-title">📈 Pertumbuhan Tertinggi</div>'
                        f'<div class="ic-val" style="color:#22c55e">{fastest}</div>'
                        f'<div class="ic-sub">+{fastest_g:.1f}% vs {growth_prev_yr.get(fastest, prev_year)}</div></div>',
                        unsafe_allow_html=True)
        with ic:
            st.markdown(f'<div class="insight-card"><div class="ic-title">📉 Pertumbuhan Terendah</div>'
                        f'<div class="ic-val" style="color:#f87171">{slowest}</div>'
                        f'<div class="ic-sub">{slowest_g:+.1f}% vs {growth_prev_yr.get(slowest, prev_year)}</div></div>',
                        unsafe_allow_html=True)
        with id_:
            st.markdown(f'<div class="insight-card"><div class="ic-title">📋 Total Kasus {latest_year}</div>'
                        f'<div class="ic-val">{total_latest:,}</div>'
                        f'<div class="ic-sub">{len(active_progs)} program aktif</div></div>',
                        unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # ── Pie + Bar ──────────────────────────────────────────────────────────────
    r1, r2 = st.columns(2)
    with r1:
        st.markdown('<div class="sec">Distribusi Kasus — Semua Tahun (Program Aktif)</div>', unsafe_allow_html=True)
        pie_d = df_plot.groupby('Kategori')['Kasus'].sum().reset_index()
        fig = px.pie(pie_d, names='Kategori', values='Kasus', hole=0.5,
                     color_discrete_sequence=COLORS)
        fig.update_traces(textinfo='label+percent', textposition='outside', textfont_size=11)
        fig.update_layout(**DARK, showlegend=True, height=400,
                          legend=dict(orientation='h', y=-0.1, font=dict(size=10)),
                          margin=dict(t=10, b=60, l=10, r=10))
        total_kasus = int(pie_d['Kasus'].sum())
        fig.add_annotation(text=f"<b>{total_kasus:,}</b><br><span style='font-size:10px'>Total</span>",
            showarrow=False, font=dict(size=13, color='#334155'), align='center')
        st.plotly_chart(fig, width='stretch')

    with r2:
        st.markdown(f'<div class="sec">Market Share per Program — {latest_year}</div>', unsafe_allow_html=True)
        bar_d = df_lat.groupby('Kategori')['Kasus'].sum().sort_values(ascending=True).reset_index()
        total_bar = bar_d['Kasus'].sum()
        bar_d['Share'] = (bar_d['Kasus']/total_bar*100).round(1)
        fig2 = go.Figure()
        for i, row in bar_d.iterrows():
            col_c = COLORS[i % len(COLORS)]
            fig2.add_trace(go.Bar(
                x=[row['Kasus']], y=[row['Kategori']], orientation='h',
                name=row['Kategori'], marker_color=col_c, marker_line_width=0,
                text=f"{row['Kasus']:,} ({row['Share']}%)", textposition='outside',
                showlegend=False,
            ))
        fig2.update_layout(**DARK, height=400, showlegend=False,
                           margin=dict(t=10, b=10, l=10, r=120))
        st.plotly_chart(fig2, width='stretch')

    if not single_yr:
        trend = df_plot.groupby(['Tahun','Kategori'])['Kasus'].sum().reset_index()

        # World Bank data
        all_yrs_data = sorted(trend['Tahun'].unique().tolist())
        with st.spinner("🌐 Mengambil data ekonomi dari World Bank API..."):
            wb_ekon = _fetch_worldbank(tuple(all_yrs_data))
        st.session_state['_wb_ekon_cache'] = wb_ekon

        EKON_CONTEXT = {}
        for _yr, _d in wb_ekon.items():
            _g = _d.get('gdp_pct')
            if _g is None: continue
            _icon = ("⚠️ Kontraksi" if _g < 0 else "🔄 Pemulihan" if _g < 3
                     else "📊 Moderat" if _g < 5 else "🚀 Ekspansi")
            _parts = []
            if 'gdp_pct'          in _d: _parts.append(f"PDB {_d['gdp_pct']:+.2f}%")
            if 'inflation_pct'    in _d: _parts.append(f"Inflasi {_d['inflation_pct']:.1f}%")
            if 'unemployment_pct' in _d: _parts.append(f"Pengangguran {_d['unemployment_pct']:.1f}%")
            EKON_CONTEXT[_yr] = (_icon, ", ".join(_parts) + ".")

        # Tren chart
        st.markdown('<div class="sec">Tren Kasus per Tahun — dengan Konteks Ekonomi AI</div>', unsafe_allow_html=True)
        fig3 = go.Figure()
        for i, cat in enumerate(sorted(df_plot['Kategori'].unique())):
            cd = trend[trend['Kategori']==cat].sort_values('Tahun')
            col_c = COLORS[i % len(COLORS)]
            fig3.add_trace(go.Scatter(
                x=cd['Tahun'], y=cd['Kasus'], name=cat, mode='lines+markers',
                line=dict(color=col_c, width=2.5),
                marker=dict(size=9, color=col_c, line=dict(color='rgba(255,255,255,0.7)', width=1.5)),
                fill='tozeroy', fillcolor=hex_to_rgba(col_c, 0.07),
                hovertemplate=f"<b>{cat}</b><br>Tahun: %{{x}}<br>Kasus: %{{y:,}}<extra></extra>"
            ))
        tot_yr = trend.groupby('Tahun')['Kasus'].sum()
        yr_max_t = int(tot_yr.max()) if len(tot_yr) > 0 else 1
        for _yr, (_icon_c, _desc_c) in EKON_CONTEXT.items():
            if _yr in all_yrs_data:
                _g = wb_ekon.get(_yr, {}).get('gdp_pct')
                if _g is not None and (_g < 0 or _g > 5.5):
                    _clr = '#dc2626' if _g < 0 else '#16a34a'
                    _bg  = 'rgba(254,226,226,0.92)' if _g < 0 else 'rgba(220,252,231,0.92)'
                    _brd = '#fca5a5' if _g < 0 else '#86efac'
                    fig3.add_annotation(x=_yr, y=yr_max_t * 0.88,
                        text=f"{_icon_c}<br>PDB {_g:+.1f}%", showarrow=False,
                        font=dict(size=9, color=_clr, family='Inter'),
                        bgcolor=_bg, bordercolor=_brd, borderwidth=1, borderpad=4)
        styled_chart(fig3, height=480)
        fig3.update_layout(xaxis=dict(dtick=1))
        st.plotly_chart(fig3, width='stretch')

        # Auto-insight
        _trend_total = trend.groupby('Tahun')['Kasus'].sum()
        if len(_trend_total) >= 2:
            _yr_max_k = int(_trend_total.idxmax())
            _yr_min_k = int(_trend_total.idxmin())
            _yr_last2 = sorted(_trend_total.index)[-2:]
            _growth_last = (_trend_total[_yr_last2[1]] / (_trend_total[_yr_last2[0]] + 1e-9) - 1) * 100
            _ctx_peak = EKON_CONTEXT.get(_yr_max_k, (None, None))
            _peak_desc = _ctx_peak[1] if _ctx_peak and _ctx_peak[0] else ""
            st.markdown(
                f'<div class="info-box">📊 <b>Auto-Insight Tren:</b> '
                f'Klaim tertinggi di <b>{_yr_max_k}</b> ({int(_trend_total[_yr_max_k]):,} kasus). '
                f'Klaim terendah di <b>{_yr_min_k}</b> ({int(_trend_total[_yr_min_k]):,} kasus). '
                f'Pertumbuhan {_yr_last2[0]}→{_yr_last2[1]}: '
                f'<b style="color:{"#16a34a" if _growth_last >= 0 else "#dc2626"}">{_growth_last:+.1f}%</b>.'
                + (f' <b>Konteks {_yr_max_k}:</b> {_peak_desc}' if _peak_desc else '')
                + f'</div>', unsafe_allow_html=True)

        # Peak & Trough
        st.markdown('<div class="sec">📍 Analisis Peak & Trough per Program — Dianalisis oleh AI</div>', unsafe_allow_html=True)
        _api_key = _get_api_key()
        progs_to_analyze = sorted(filtered_progs)
        n_pcols = min(len(progs_to_analyze), 3)

        if n_pcols > 0:
            peak_cols_list = st.columns(n_pcols)
            for pi, prog in enumerate(progs_to_analyze):
                prog_trend = trend[trend['Kategori']==prog].sort_values('Tahun')
                if len(prog_trend) < 2: continue
                peak_yr    = int(prog_trend.loc[prog_trend['Kasus'].idxmax(), 'Tahun'])
                trough_yr  = int(prog_trend.loc[prog_trend['Kasus'].idxmin(), 'Tahun'])
                peak_val   = int(prog_trend['Kasus'].max())
                trough_val = int(prog_trend['Kasus'].min())
                range_pct  = abs((peak_val - trough_val) / (trough_val + 1e-9) * 100)
                _cache_key = f"ai_pt_{prog}_{peak_yr}_{trough_yr}"
                if _cache_key not in st.session_state:
                    st.session_state[_cache_key] = None
                ai_res = st.session_state.get(_cache_key)
                if ai_res and isinstance(ai_res, dict):
                    peak_label   = ai_res.get("peak_label",   "📊 Puncak")
                    peak_desc    = ai_res.get("peak_desc",    EKON_CONTEXT.get(peak_yr, ("", "Data terbatas."))[1])
                    trough_label = ai_res.get("trough_label", "📊 Lembah")
                    trough_desc  = ai_res.get("trough_desc",  EKON_CONTEXT.get(trough_yr, ("", "Data terbatas."))[1])
                else:
                    _pk = EKON_CONTEXT.get(peak_yr,   ("📊 Data WB", "Set GROQ_API_KEY di Secrets untuk analisis AI."))
                    _tr = EKON_CONTEXT.get(trough_yr, ("📊 Data WB", "Set GROQ_API_KEY di Secrets untuk analisis AI."))
                    peak_label, peak_desc     = _pk
                    trough_label, trough_desc = _tr
                with peak_cols_list[pi % n_pcols]:
                    if _api_key and _api_key[1] and st.session_state.get(_cache_key) is None:
                        if st.button(f"🤖 Analisis AI: {prog}", key=f"btn_ai_{prog}_{pi}"):
                            with st.spinner(f"AI menganalisis {prog}..."):
                                st.session_state[_cache_key] = _ai_analyze_peak_trough(
                                    prog, peak_yr, peak_val, trough_yr, trough_val, wb_ekon, _api_key)
                            st.rerun()
                    ai_res = st.session_state.get(_cache_key)
                    st.markdown(
                        f'<div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;'
                        f'padding:16px 18px;margin-bottom:10px;">'
                        f'<div style="font-size:.68rem;font-weight:700;color:#64748b;text-transform:uppercase;margin-bottom:10px;">📊 {prog}</div>'
                        f'<div style="background:#fef9c3;border-left:3px solid #eab308;border-radius:6px;padding:10px 12px;margin-bottom:8px;">'
                        f'<div style="font-size:.7rem;font-weight:700;color:#854d0e;margin-bottom:3px;">🔺 PEAK — {peak_yr} &nbsp;{peak_label}</div>'
                        f'<div style="font-size:.95rem;font-weight:800;color:#713f12">{peak_val:,} kasus</div>'
                        f'<div style="font-size:.75rem;color:#92400e;margin-top:4px;line-height:1.6">{peak_desc}</div></div>'
                        f'<div style="background:#eff6ff;border-left:3px solid #3b82f6;border-radius:6px;padding:10px 12px;">'
                        f'<div style="font-size:.7rem;font-weight:700;color:#1e3a8a;margin-bottom:3px;">🔻 TROUGH — {trough_yr} &nbsp;{trough_label}</div>'
                        f'<div style="font-size:.95rem;font-weight:800;color:#1e40af">{trough_val:,} kasus</div>'
                        f'<div style="font-size:.75rem;color:#1d4ed8;margin-top:4px;line-height:1.6">{trough_desc}</div></div>'
                        f'<div style="margin-top:8px;font-size:.75rem;color:#64748b;">Range peak↔trough: <b style="color:#0f172a">{range_pct:.0f}%</b></div>'
                        f'</div>', unsafe_allow_html=True)

        # Stacked bar + Heatmap
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
            st.markdown('<div class="sec">Heatmap Kasus (Program × Tahun)</div>', unsafe_allow_html=True)
            hm_p = (df_plot.groupby(['Kategori','Tahun'])['Kasus'].sum()
                    .reset_index().pivot(index='Kategori', columns='Tahun', values='Kasus').fillna(0))
            fig5 = px.imshow(hm_p, color_continuous_scale='Blues', aspect='auto', text_auto=',')
            fig5.update_layout(**DARK, height=360, margin=dict(t=10,b=10,l=10,r=10))
            st.plotly_chart(fig5, width='stretch')

        # YoY Growth
        st.markdown('<div class="sec">Year-over-Year Growth & CAGR per Program</div>', unsafe_allow_html=True)
        yoy = []
        for cat in active_progs:
            cd = df_plot[df_plot['Kategori']==cat].sort_values('Tahun')
            for i in range(1, len(cd)):
                prev = cd.iloc[i-1]['Kasus']
                curr = cd.iloc[i]['Kasus']
                yoy.append({
                    'Kategori': cat, 'Tahun': int(cd.iloc[i]['Tahun']),
                    'Label': f"{int(cd.iloc[i]['Tahun'])} vs {int(cd.iloc[i-1]['Tahun'])}",
                    'Growth (%)': round((curr/(prev+1e-9)-1)*100, 2)
                })
        if yoy:
            ydf = pd.DataFrame(yoy)
            fig_y = px.bar(ydf, x='Tahun', y='Growth (%)', color='Kategori',
                           barmode='group', color_discrete_sequence=COLORS, text='Growth (%)')
            fig_y.update_traces(texttemplate='%{text:.1f}%', textposition='outside',
                                textfont_size=9, marker_line_width=0)
            fig_y.add_hline(y=0, line_color='#334155', line_width=1.5)
            styled_chart(fig_y, height=360)
            fig_y.update_layout(xaxis=dict(dtick=1))
            st.plotly_chart(fig_y, width='stretch')

    # Nominal
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
            fp.update_layout(**DARK, showlegend=False, height=360, margin=dict(t=10,b=10,l=10,r=10))
            fp.add_annotation(text=f"<b>Rp{total_nom:,.1f}B</b><br><span style='font-size:9px'>Total</span>",
                showarrow=False, font=dict(size=12, color='#334155'), align='center')
            st.plotly_chart(fp, width='stretch')
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
                        fillcolor=hex_to_rgba(col_c, 0.3),
                    ))
                styled_chart(fn, height=360)
                fn.update_layout(xaxis=dict(dtick=1), yaxis_title='Rp Miliar')
                st.plotly_chart(fn, width='stretch')