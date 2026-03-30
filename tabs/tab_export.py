"""tabs/tab_export.py — Tab 4: Export Excel & CSV."""
import streamlit as st
import pandas as pd
from datetime import datetime


def render_tab_export(df, active_progs, years, latest_year, has_nom,
                      results_cache, export_excel):
    st.markdown('<div class="sec">Export Laporan</div>', unsafe_allow_html=True)
    ec1, ec2 = st.columns(2)

    best_ml         = next((v for v in results_cache.values() if v), None)
    last_fut        = st.session_state.get('last_forecast', None)
    fut_ann_kasus   = st.session_state.get('forecast_annual_Kasus',    None)
    fut_ann_nominal = st.session_state.get('forecast_annual_Nominal',  None)
    fut_mo_kasus    = st.session_state.get('forecast_monthly_Kasus',   None)
    fut_mo_nominal  = st.session_state.get('forecast_monthly_Nominal', None)

    if fut_ann_kasus is None and fut_ann_nominal is None and last_fut is not None:
        tc_lf = [c for c in last_fut.columns if c not in ['Kategori','Tahun','Type']]
        if tc_lf:
            if tc_lf[0] == 'Kasus':
                fut_ann_kasus = last_fut
            else:
                fut_ann_nominal = last_fut

    has_ann_k  = fut_ann_kasus   is not None and len(fut_ann_kasus)   > 0
    has_ann_n  = fut_ann_nominal is not None and len(fut_ann_nominal) > 0
    has_mo_k   = fut_mo_kasus    is not None and len(fut_mo_kasus)    > 0
    has_mo_n   = fut_mo_nominal  is not None and len(fut_mo_nominal)  > 0

    with ec1:
        st.markdown("**📊 Excel dengan Chart Terintegrasi**")
        st.caption("Sheet: Data Gabungan · Pivot Kasus · Prediksi Tahunan · Prediksi Bulanan · ML Results")

        status_html = '<div class="info-box"><b>Prediksi Tahunan:</b><br>'
        status_html += (f'✅ Kasus tahunan siap ({len(fut_ann_kasus)} baris)<br>'
                        if has_ann_k else '⚠️ Kasus tahunan belum ada<br>')
        status_html += (f'✅ Nominal tahunan siap ({len(fut_ann_nominal)} baris)<br>'
                        if has_ann_n else '⚠️ Nominal tahunan belum ada<br>')
        status_html += '<br><b>Prediksi Bulanan:</b><br>'
        status_html += (f'✅ Kasus bulanan siap ({len(fut_mo_kasus)} baris)<br>'
                        if has_mo_k else '⚠️ Kasus bulanan belum ada<br>')
        status_html += (f'✅ Nominal bulanan siap ({len(fut_mo_nominal)} baris)'
                        if has_mo_n else '⚠️ Nominal bulanan belum ada')
        status_html += '</div>'
        st.markdown(status_html, unsafe_allow_html=True)

        if st.button("⚙️ Generate Excel", width='stretch'):
            with st.spinner("Membuat Excel..."):
                xlsx = export_excel(
                    df, best_ml, last_fut,
                    fut_kasus=fut_ann_kasus, fut_nominal=fut_ann_nominal,
                    fut_monthly_kasus=fut_mo_kasus, fut_monthly_nominal=fut_mo_nominal,
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

        if has_mo_k or has_mo_n:
            st.markdown("**📄 CSV Prediksi Bulanan**")
            frames_csv = []
            if has_mo_k:
                frames_csv.append(fut_mo_kasus.assign(Target='Kasus'))
            if has_mo_n:
                frames_csv.append(fut_mo_nominal.assign(Target='Nominal'))
            combined_csv = (pd.concat(frames_csv, ignore_index=True)
                            .sort_values(['Tahun','Bulan','Kategori','Target'])
                            .reset_index(drop=True))
            st.download_button(
                "⬇️ Download CSV Bulanan", data=combined_csv.to_csv(index=False),
                file_name=f"BPJS_Bulanan_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv", width='stretch')

    st.markdown('<div class="sec">Preview Data Aktif</div>', unsafe_allow_html=True)
    st.info(f"**{len(df)} baris** | **{len(active_progs)} program aktif** "
            f"({', '.join(active_progs)}) | **Tahun: {', '.join(map(str, years))}**")
    st.dataframe(df, width='stretch', height=360)