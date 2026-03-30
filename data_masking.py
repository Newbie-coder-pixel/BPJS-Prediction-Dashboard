"""
data_masking.py — Proteksi data sebelum dikirim ke LLM.
Tidak ada raw data / nominal per-individu yang keluar.
"""
import numpy as np
import pandas as pd
from typing import Any


def _bucket_nominal(val_billions: float) -> str:
    if val_billions < 1:    return "<1 M"
    if val_billions < 10:   return "1–10 M"
    if val_billions < 50:   return "10–50 M"
    if val_billions < 100:  return "50–100 M"
    if val_billions < 500:  return "100–500 M"
    return ">500 M"


def build_safe_context(
    df: pd.DataFrame,
    active_progs: list,
    years: list,
    has_nom: bool,
    latest_year: int,
    question: str,
) -> str:
    if df is None or len(df) == 0:
        return "Belum ada data yang diupload."

    q = question.lower()
    parts = []

    parts.append(f"DATA KLAIM BPJS KETENAGAKERJAAN — {latest_year}")
    parts.append(f"Program aktif ({len(active_progs)}): {', '.join(active_progs)}")
    parts.append(f"Rentang data: {min(years)}–{max(years)} ({len(years)} tahun)")
    parts.append("")

    for prog in active_progs:
        sub = df[df['Kategori'] == prog].sort_values('Tahun')
        if sub.empty:
            continue

        first, last = sub.iloc[0], sub.iloc[-1]
        kasus_growth = (last['Kasus'] - first['Kasus']) / (first['Kasus'] + 1e-9) * 100
        trend = "naik" if kasus_growth > 5 else ("turun" if kasus_growth < -5 else "stabil")

        line = (
            f"{prog}: {int(first['Kasus']):,} kasus ({int(first['Tahun'])}) → "
            f"{int(last['Kasus']):,} kasus ({int(last['Tahun'])}) "
            f"[{kasus_growth:+.1f}%, {trend}]"
        )

        if has_nom and 'Nominal' in sub.columns:
            n_first = float(first.get('Nominal', 0)) / 1e9
            n_last  = float(last.get('Nominal', 0)) / 1e9
            nom_growth = (n_last - n_first) / (n_first + 1e-9) * 100
            line += (
                f" | nominal: {_bucket_nominal(n_first)} → "
                f"{_bucket_nominal(n_last)} [{nom_growth:+.1f}%]"
            )

        parts.append(line)

    parts.append("")

    if any(w in q for w in ["tren", "trend", "naik", "turun", "pertumbuhan"]):
        parts.append("--- Tren Tahunan (total semua program) ---")
        yearly = df.groupby('Tahun')['Kasus'].sum().sort_index()
        for yr, val in yearly.items():
            parts.append(f"  {yr}: {int(val):,} kasus")

    if any(w in q for w in ["perbandingan", "bandingkan", "terbesar", "terkecil", "ranking"]):
        parts.append("--- Ranking Program Berdasarkan Kasus Terbaru ---")
        latest_df = df[df['Tahun'] == latest_year].sort_values('Kasus', ascending=False)
        for rank, (_, row) in enumerate(latest_df.iterrows(), 1):
            parts.append(f"  {rank}. {row['Kategori']}: {int(row['Kasus']):,} kasus")

    if any(w in q for w in ["total", "keseluruhan", "semua", "agregat"]):
        total_kasus = int(df[df['Tahun'] == latest_year]['Kasus'].sum())
        parts.append(f"--- Total Semua Program {latest_year}: {total_kasus:,} kasus ---")

    parts.append("")
    parts.append(
        "CATATAN: Data ini adalah agregasi per program per tahun. "
        "Tidak ada data individu/NIK/nama yang tersedia."
    )

    return "\n".join(parts)


def safe_ml_summary(ml_result: dict) -> str:
    """
    Ringkasan metrik akurasi model ML — tanpa expose internal model params.
    Hanya kirim nama model dan akurasi, BUKAN angka prediksi.
    Angka prediksi diambil terpisah via build_forecast_context().
    """
    if not ml_result:
        return ""
    bpp = ml_result.get('best_per_program')
    if bpp is None or (hasattr(bpp, 'empty') and bpp.empty):
        return ""

    lines = ["--- Model ML yang Digunakan per Program ---"]
    for _, row in bpp.iterrows():
        mape = row.get('MAPE (%)')
        r2   = row.get('R2')
        mape_str = f"{mape:.1f}%" if mape is not None and not np.isnan(float(mape if mape else 0)) else "N/A"
        r2_str   = f"{r2:.3f}"   if r2   is not None and not np.isnan(float(r2 if r2 else 0))   else "N/A"
        lines.append(
            f"  {row['Program']}: model={row['Model']}, MAPE={mape_str}, R²={r2_str}"
        )
    return "\n".join(lines)


def build_forecast_context(
    forecast_df: pd.DataFrame,
    target: str,
    latest_year: int,
    has_nom: bool = False,
    forecast_nom_df: pd.DataFrame = None,
) -> str:
    """
    Fungsi BARU — ambil angka prediksi dari hasil forecast dan
    format jadi teks yang bisa dibaca LLM.

    forecast_df = st.session_state.get('forecast_Kasus')
    forecast_nom_df = st.session_state.get('forecast_Nominal')

    Hanya kirim angka prediksi agregat per program per tahun.
    Tidak ada data individu.
    """
    if forecast_df is None or (hasattr(forecast_df, 'empty') and forecast_df.empty):
        return ""

    try:
        lines = [f"--- PREDIKSI KASUS (dari model ML, base year={latest_year}) ---"]

        pred_df = forecast_df[forecast_df['Tahun'] > latest_year].copy()
        if pred_df.empty:
            # Coba ambil semua data jika filter kosong
            pred_df = forecast_df.copy()

        if 'Kategori' not in pred_df.columns or target not in pred_df.columns:
            return ""

        # Ringkas per program per tahun prediksi
        for prog in sorted(pred_df['Kategori'].unique()):
            prog_pred = pred_df[pred_df['Kategori'] == prog].sort_values('Tahun')
            if prog_pred.empty:
                continue
            prog_lines = []
            for _, row in prog_pred.iterrows():
                yr   = int(row['Tahun'])
                val  = float(row[target])
                ci_pct = float(row.get(f'{target}_ci_pct', 0))
                lo   = float(row.get(f'{target}_lower', val * (1 - ci_pct/100)))
                hi   = float(row.get(f'{target}_upper', val * (1 + ci_pct/100)))
                prog_lines.append(
                    f"    {yr}: {val:,.0f} kasus "
                    f"[rentang: {lo:,.0f}–{hi:,.0f}, ±{ci_pct:.0f}%]"
                )
            if prog_lines:
                lines.append(f"  {prog}:")
                lines.extend(prog_lines)

        # Total semua program per tahun prediksi
        lines.append("")
        lines.append("  TOTAL semua program:")
        yearly_total = pred_df.groupby('Tahun')[target].sum().sort_index()
        for yr, total in yearly_total.items():
            if int(yr) > latest_year:
                lines.append(f"    {int(yr)}: {total:,.0f} kasus (total)")

        # Nominal jika tersedia (hanya bucket, bukan exact)
        if has_nom and forecast_nom_df is not None and not forecast_nom_df.empty:
            lines.append("")
            lines.append("--- PREDIKSI NOMINAL (rentang/bucket, bukan angka exact) ---")
            nom_target = 'Nominal'
            pred_nom = forecast_nom_df[forecast_nom_df['Tahun'] > latest_year].copy()
            if not pred_nom.empty and nom_target in pred_nom.columns:
                for prog in sorted(pred_nom['Kategori'].unique()):
                    prog_nom = pred_nom[pred_nom['Kategori'] == prog].sort_values('Tahun')
                    nom_parts = []
                    for _, row in prog_nom.iterrows():
                        val_b = float(row[nom_target]) / 1e9
                        nom_parts.append(
                            f"{int(row['Tahun'])}: {_bucket_nominal(val_b)}"
                        )
                    if nom_parts:
                        lines.append(f"  {prog}: {' → '.join(nom_parts)}")

        lines.append("")
        lines.append(
            "PENTING: Angka prediksi di atas adalah estimasi model ML. "
            "Rentang menunjukkan ketidakpastian. Semakin jauh tahun, "
            "semakin lebar rentangnya."
        )

        return "\n".join(lines)

    except Exception as e:
        return f"[Gagal memuat data prediksi: {str(e)}]"