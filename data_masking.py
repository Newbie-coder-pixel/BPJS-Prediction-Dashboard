"""
data_masking.py — Proteksi data sebelum dikirim ke LLM.
Tidak ada raw data / nominal per-individu yang keluar.
"""
import numpy as np
import pandas as pd
from typing import Any


# Nominal dalam miliar rupiah, dibucket agar tidak expose angka exact
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
    """
    Transformasi df → teks ringkasan aman untuk LLM.
    TIDAK ada row-level data, TIDAK ada nominal exact per-individu.
    """
    if df is None or len(df) == 0:
        return "Belum ada data yang diupload."

    q = question.lower()
    parts = []

    # ── Header ────────────────────────────────────────────────────────────────
    parts.append(f"DATA KLAIM BPJS KETENAGAKERJAAN — {latest_year}")
    parts.append(f"Program aktif ({len(active_progs)}): {', '.join(active_progs)}")
    parts.append(f"Rentang data: {min(years)}–{max(years)} ({len(years)} tahun)")
    parts.append("")

    # ── Per-program summary ───────────────────────────────────────────────────
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

        # Nominal hanya ditampilkan sebagai bucket + growth %, TIDAK exact
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

    # ── Statistik agregat tambahan (intent-based) ─────────────────────────────
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
    Ringkasan hasil ML untuk context LLM — tanpa expose internal model params.
    """
    if not ml_result:
        return ""
    bpp = ml_result.get('best_per_program')
    if bpp is None or (hasattr(bpp, 'empty') and bpp.empty):
        return ""

    lines = ["--- Ringkasan Model ML ---"]
    for _, row in bpp.iterrows():
        mape = row.get('MAPE (%)')
        r2   = row.get('R2')
        mape_str = f"{mape:.1f}%" if mape is not None and not np.isnan(float(mape if mape else 0)) else "N/A"
        r2_str   = f"{r2:.3f}"   if r2   is not None and not np.isnan(float(r2 if r2 else 0))   else "N/A"
        lines.append(
            f"  {row['Program']}: model={row['Model']}, "
            f"akurasi MAPE={mape_str}, R²={r2_str}"
        )
    return "\n".join(lines)