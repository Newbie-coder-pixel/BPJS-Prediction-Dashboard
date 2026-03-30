"""
export_utils.py — Export data dan prediksi ke file Excel (XlsxWriter).
"""
import io
import pandas as pd


def xl_col_to_name(col_idx: int) -> str:
    name = ''
    col_idx += 1
    while col_idx:
        col_idx, remainder = divmod(col_idx - 1, 26)
        name = chr(65 + remainder) + name
    return name


def export_excel(df, ml_result, fut_df,
                 fut_kasus=None, fut_nominal=None,
                 fut_monthly_kasus=None, fut_monthly_nominal=None) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
        wb      = writer.book
        hdr     = wb.add_format({'bold': True, 'bg_color': '#1e3a5f',
                                  'font_color': 'white', 'border': 1})
        num_fmt = wb.add_format({'num_format': '#,##0', 'border': 1})
        sec_fmt = wb.add_format({'bold': True, 'bg_color': '#0f2744',
                                  'font_color': '#93c5fd', 'font_size': 12, 'border': 0})

        # ── Sheet 1: Data Gabungan ────────────────────────────────────────────
        df_sorted = df.sort_values(['Tahun', 'Kategori']).reset_index(drop=True)
        df_sorted.to_excel(writer, sheet_name='Data Gabungan', index=False)
        ws1 = writer.sheets['Data Gabungan']
        for i, c in enumerate(df_sorted.columns):
            ws1.write(0, i, c, hdr)
            ws1.set_column(i, i, 22)

        # ── Sheet 2: Pivot Kasus ──────────────────────────────────────────────
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

        # ── Fallback fut_df ───────────────────────────────────────────────────
        has_fut_k = fut_kasus   is not None and len(fut_kasus)   > 0
        has_fut_n = fut_nominal is not None and len(fut_nominal) > 0
        if not has_fut_k and not has_fut_n and fut_df is not None and len(fut_df) > 0:
            tc_fb = [c for c in fut_df.columns if c not in ['Kategori','Tahun','Type']]
            if tc_fb:
                if tc_fb[0] == 'Kasus':
                    fut_kasus = fut_df; has_fut_k = True
                else:
                    fut_nominal = fut_df; has_fut_n = True

        # ── Sheet 3: Prediksi Tahunan ─────────────────────────────────────────
        if has_fut_k or has_fut_n:
            ws3_name = 'Prediksi Tahunan'
            writer.book.add_worksheet(ws3_name)
            ws3 = writer.sheets[ws3_name]

            hdr_sec  = wb.add_format({'bold': True, 'bg_color': '#0f2744',
                                       'font_color': '#93c5fd', 'font_size': 12})
            hdr_yr   = wb.add_format({'bold': True, 'bg_color': '#1e3a5f',
                                       'font_color': 'white', 'border': 1, 'align': 'center'})
            num_k    = wb.add_format({'num_format': '#,##0', 'border': 1})
            num_hist = wb.add_format({'num_format': '#,##0', 'border': 1, 'bg_color': '#1a1a2e'})
            CHART_COLORS = ['#4472C4','#ED7D31','#A9D18E','#FF0000',
                            '#7030A0','#00B0F0','#92D050','#FFC000']

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
                cats   = [c for c in combined.columns if c not in ['Tahun','_type']]
                n_rows = len(combined)
                n_cats = len(cats)
                ws.merge_range(cursor, 0, cursor, n_cats + 1, title_txt, hdr_sec)
                hdr_row = cursor + 1
                ws.write(hdr_row, 0, 'Tahun', hdr_yr); ws.set_column(0, 0, 10)
                ws.write(hdr_row, 1, 'Tipe',  hdr_yr); ws.set_column(1, 1, 12)
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
                aktual_rows = [data_start + i for i, r in combined.iterrows() if r['_type'] == 'Aktual']
                pred_rows   = [data_start + i for i, r in combined.iterrows() if r['_type'] == 'Prediksi']
                for ci, cat in enumerate(cats):
                    col_excel = ci + 2
                    color     = CHART_COLORS[ci % len(CHART_COLORS)]
                    if aktual_rows:
                        ch.add_series({
                            'name': cat + ' Aktual',
                            'categories': [ws_name, aktual_rows[0], 0, aktual_rows[-1], 0],
                            'values':     [ws_name, aktual_rows[0], col_excel, aktual_rows[-1], col_excel],
                            'line': {'color': color, 'width': 2.25},
                            'marker': {'type': 'circle', 'size': 6,
                                       'fill': {'color': color}, 'border': {'color': color}},
                        })
                    if pred_rows:
                        ch.add_series({
                            'name': cat + ' Prediksi',
                            'categories': [ws_name, pred_rows[0], 0, pred_rows[-1], 0],
                            'values':     [ws_name, pred_rows[0], col_excel, pred_rows[-1], col_excel],
                            'line': {'color': color, 'width': 2.25, 'dash_type': 'dash'},
                            'marker': {'type': 'diamond', 'size': 7,
                                       'fill': {'color': color}, 'border': {'color': color}},
                        })
                ch.set_title({'name': title_txt})
                ch.set_x_axis({'name': 'Tahun', 'major_gridlines': {'visible': True, 'line': {'color': '#e0e0e0'}}, 'num_font': {'size': 10}})
                ch.set_y_axis({'name': value_col, 'major_gridlines': {'visible': True, 'line': {'color': '#e0e0e0'}}, 'num_format': '#,##0'})
                ch.set_legend({'position': 'bottom', 'font': {'size': 9}})
                ch.set_size({'width': 800, 'height': 450})
                ch.set_style(10)
                chart_col = xl_col_to_name(n_cats + 3)
                ws.insert_chart(f'{chart_col}{hdr_row + 1}', ch)
                return last_data_row + 4

            df_hist_k = df[['Tahun','Kategori','Kasus']].copy() if 'Kasus' in df.columns else None
            df_hist_n = df[['Tahun','Kategori','Nominal']].copy() if 'Nominal' in df.columns else None

            if has_fut_k and df_hist_k is not None:
                cursor = _write_annual_block(
                    ws3, wb, df_hist_k, fut_kasus, 'Kasus',
                    '📊 PREDIKSI KASUS (TAHUNAN) — Aktual vs Proyeksi', cursor, ws3_name)
            if has_fut_n and df_hist_n is not None:
                cursor = _write_annual_block(
                    ws3, wb, df_hist_n, fut_nominal, 'Nominal',
                    '💰 PREDIKSI NOMINAL (TAHUNAN) — Aktual vs Proyeksi', cursor, ws3_name)

        # ── Sheet 4: Prediksi Bulanan ─────────────────────────────────────────
        has_kasus   = fut_monthly_kasus   is not None and len(fut_monthly_kasus)   > 0
        has_nominal = fut_monthly_nominal is not None and len(fut_monthly_nominal) > 0

        if has_kasus or has_nominal:
            ws4_name = 'Prediksi Bulanan'
            writer.book.add_worksheet(ws4_name)
            ws4 = writer.sheets[ws4_name]
            cursor = 0

            def _write_monthly(ws, wb, ws_name, df_mo, val_col, title, sec_fmt, hdr, num_fmt, cursor):
                ws.merge_range(cursor, 0, cursor, 7, title, sec_fmt)
                cursor += 1
                piv = (df_mo.sort_values(['Tahun','Bulan','Kategori'])
                       .pivot_table(index='Periode', columns='Kategori', values=val_col, aggfunc='sum')
                       .reset_index().sort_values('Periode').reset_index(drop=True))
                nrow = len(piv); ncat = len(piv.columns) - 1
                hdr_row = cursor
                for ci, cn in enumerate(piv.columns):
                    ws.write(cursor, ci, str(cn), hdr); ws.set_column(ci, ci, 16)
                cursor += 1
                for ri in range(nrow):
                    ws.write(cursor + ri, 0, piv.iloc[ri, 0])
                    for ci in range(1, ncat + 1):
                        ws.write(cursor + ri, ci, piv.iloc[ri, ci], num_fmt)
                last_row = cursor + nrow - 1
                ch = wb.add_chart({'type': 'line'})
                for ci in range(1, ncat + 1):
                    ch.add_series({
                        'name':       [ws_name, hdr_row, ci],
                        'categories': [ws_name, hdr_row + 1, 0, last_row, 0],
                        'values':     [ws_name, hdr_row + 1, ci, last_row, ci],
                        'marker':     {'type': 'circle', 'size': 4},
                    })
                ch.set_title({'name': f'Prediksi {val_col} per Program (Bulanan)'})
                ch.set_x_axis({'name': 'Periode (YYYY-MM)', 'num_font': {'rotation': -45}})
                ch.set_y_axis({'name': val_col})
                ch.set_legend({'position': 'bottom'})
                ch.set_size({'width': 760, 'height': 420})
                ws.insert_chart(f'{xl_col_to_name(ncat + 2)}{hdr_row + 1}', ch)
                return last_row + 3

            if has_kasus:
                cursor = _write_monthly(ws4, wb, ws4_name, fut_monthly_kasus,
                                        'Kasus', '📊 PREDIKSI KASUS (BULANAN)',
                                        sec_fmt, hdr, num_fmt, cursor)
            if has_nominal:
                cursor = _write_monthly(ws4, wb, ws4_name, fut_monthly_nominal,
                                        'Nominal', '💰 PREDIKSI NOMINAL (BULANAN)',
                                        sec_fmt, hdr, num_fmt, cursor)

        # ── Sheet 5: Bulanan Detail ───────────────────────────────────────────
        detail_frames = []
        if has_kasus:
            detail_frames.append(fut_monthly_kasus[['Periode','Tahun','Bulan','Kategori','Kasus']].copy())
        if has_nominal:
            detail_frames.append(fut_monthly_nominal[['Periode','Tahun','Bulan','Kategori','Nominal']].copy())
        if detail_frames:
            detail_all = (detail_frames[0].merge(detail_frames[1],
                          on=['Periode','Tahun','Bulan','Kategori'], how='outer')
                          if len(detail_frames) == 2 else detail_frames[0])
            detail_all = detail_all.sort_values(['Tahun','Bulan','Kategori']).reset_index(drop=True)
            detail_all.to_excel(writer, sheet_name='Bulanan Detail', index=False)
            ws5 = writer.sheets['Bulanan Detail']
            for i, c in enumerate(detail_all.columns):
                ws5.write(0, i, c, hdr); ws5.set_column(i, i, 18)
            for ci, col_name in enumerate(detail_all.columns):
                if col_name in ('Kasus', 'Nominal'):
                    for ri in range(len(detail_all)):
                        ws5.write(ri + 1, ci, detail_all.iloc[ri][col_name], num_fmt)

        # ── Sheet 6: ML Results ───────────────────────────────────────────────
        if ml_result:
            rdf = ml_result['results_df']
            rdf.to_excel(writer, sheet_name='ML Results', index=False)
            ws6 = writer.sheets['ML Results']
            for i, c in enumerate(rdf.columns):
                ws6.write(0, i, c, hdr); ws6.set_column(i, i, 18)
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