"""
data_masking.py — Proteksi data sebelum dikirim ke LLM.
Tidak ada raw data / nominal per-individu yang keluar.
Versi upgrade: mendukung kolom Sebab Klaim untuk analisis penyebab klaim.
PERBAIKAN:
- build_safe_context: inject multi-year data untuk pertanyaan tren/dampak
- build_safe_context: inject pengetahuan timeline COVID-19
- _build_sebab_context: mode multi-tahun untuk pertanyaan tren/dampak
"""
import re
import numpy as np
import pandas as pd
from typing import Any


def _bucket_nominal(val_billions: float) -> str:
    if val_billions < 0.001: return "<1 M"
    if val_billions < 1:     return f"{val_billions*1000:.0f} M"
    if val_billions < 10:    return f"Rp {val_billions:.1f} T"
    if val_billions < 100:   return f"Rp {val_billions:.0f} T"
    return f"Rp {val_billions:.0f} T"


def _fmt_nominal(val: float) -> str:
    """Format nominal ke Rp T / M / B yang readable."""
    if val >= 1e12:  return f"Rp {val/1e12:.2f} T"
    if val >= 1e9:   return f"Rp {val/1e9:.1f} M"
    if val >= 1e6:   return f"Rp {val/1e6:.0f} jt"
    return f"Rp {val:,.0f}"


# ── Keyword groups ────────────────────────────────────────────────────────────

_TREND_KEYWORDS = [
    "tren", "trend", "naik", "turun", "pertumbuhan", "historis",
    "dampak", "pengaruh", "covid", "pandemi", "krisis", "efek",
    "perubahan", "sebelum", "sesudah", "pasca", "setelah",
    "bandingkan", "komparasi", "dibanding", "vs", "versus",
]

_SEBAB_KEYWORDS = [
    "sebab", "penyebab", "alasan", "kenapa", "mengapa", "karena",
    "cause", "reason", "faktor", "terbesar", "terbanyak", "dominan",
    "mengundurkan", "phk", "pensiun", "kecelakaan", "meninggal",
]

_COVID_KEYWORDS = [
    "covid", "pandemi", "corona", "wabah", "lockdown",
    "psbb", "pembatasan", "new normal",
]


def _is_trend_question(q: str) -> bool:
    return any(k in q for k in _TREND_KEYWORDS)


def _is_covid_question(q: str) -> bool:
    return any(k in q for k in _COVID_KEYWORDS)


def _is_sebab_question(q: str) -> bool:
    return any(k in q for k in _SEBAB_KEYWORDS)


def _extract_years_from_question(question: str, latest_year: int) -> list[int] | None:
    """
    Ekstrak tahun dari pertanyaan.
    Return None = tidak ada tahun spesifik (pakai semua / latest).
    Return list = tahun yang diminta.
    """
    q = question.lower()
    found = [int(y) for y in re.findall(r'20\d{2}', question)]
    if found:
        return found
    if 'tahun ini' in q or 'terbaru' in q:
        return [latest_year]
    if 'tahun lalu' in q or 'sebelumnya' in q:
        return [latest_year - 1]
    return None


# ── COVID timeline knowledge — diinjeksi ke prompt ───────────────────────────

COVID_TIMELINE_CONTEXT = """
[KONTEKS COVID-19 — PENGETAHUAN UMUM]
- COVID-19 pertama kali terdeteksi: Desember 2019 (Wuhan, China)
- WHO menyatakan pandemi global: 11 Maret 2020
- Indonesia kasus pertama: 2 Maret 2020
- PSBB (Pembatasan Sosial Berskala Besar) Indonesia: April–Mei 2020
- Gelombang Delta (varian B.1.617.2): Juli–Agustus 2021 (puncak kasus Indonesia)
- New Normal / transisi endemi: pertengahan 2022
- WHO mencabut status darurat kesehatan global: 5 Mei 2023
- Dampak ke ketenagakerjaan: PHK massal 2020–2021, pemulihan bertahap 2022–2023
INSTRUKSI: Gunakan timeline di atas untuk konteks, bukan angka klaim dari sini.
Angka klaim HARUS dari [DATA HISTORIS] atau [DETAIL SEBAB KLAIM].
""".strip()


def build_safe_context(
    df: pd.DataFrame,
    active_progs: list,
    years: list,
    has_nom: bool,
    latest_year: int,
    question: str,
    df_raw: pd.DataFrame = None,
) -> str:
    """
    Transformasi df → teks ringkasan aman untuk LLM.
    df_raw = raw dataframe sebelum agregasi, dipakai untuk analisis Sebab Klaim.
    """
    if df is None or len(df) == 0:
        return "Belum ada data yang diupload."

    q = question.lower()
    parts = []

    # ── Header ────────────────────────────────────────────────────────────────
    parts.append("DATA KLAIM BPJS KETENAGAKERJAAN (SUMBER: DATASET INTERNAL)")
    parts.append(f"Program aktif ({len(active_progs)}): {', '.join(active_progs)}")
    parts.append(f"Rentang data: {min(years)}–{max(years)} ({len(years)} tahun)")
    parts.append("")

    # ── COVID timeline — inject jika relevan ─────────────────────────────────
    if _is_covid_question(q):
        parts.append(COVID_TIMELINE_CONTEXT)
        parts.append("")

    # ── Per-program summary (selalu ditampilkan) ──────────────────────────────
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
            n_first = float(first.get('Nominal', 0))
            n_last  = float(last.get('Nominal', 0))
            nom_growth = (n_last - n_first) / (n_first + 1e-9) * 100
            line += (
                f" | nominal: {_fmt_nominal(n_first)} → "
                f"{_fmt_nominal(n_last)} [{nom_growth:+.1f}%]"
            )

        parts.append(line)

    parts.append("")

    # ── Intent-based aggregations ─────────────────────────────────────────────

    # Ranking / terbesar
    if any(w in q for w in ["terbesar", "terkecil", "ranking", "terbanyak",
                             "tertinggi", "terendah", "berapa", "data klaim"]):
        parts.append(f"--- Ranking Program Tahun {latest_year} ---")
        latest_df = df[df['Tahun'] == latest_year].sort_values('Kasus', ascending=False)
        for rank, (_, row) in enumerate(latest_df.iterrows(), 1):
            line = f"  {rank}. {row['Kategori']}: {int(row['Kasus']):,} kasus"
            if has_nom and 'Nominal' in row:
                line += f" | {_fmt_nominal(float(row['Nominal']))}"
            parts.append(line)
        parts.append("")

    # ── Tren tahunan lengkap — untuk pertanyaan tren, dampak, COVID ───────────
    if _is_trend_question(q) or _is_covid_question(q):
        parts.append("--- DATA HISTORIS PER TAHUN — SEMUA PROGRAM ---")
        parts.append("(Gunakan ini untuk analisis perubahan antar tahun)")
        parts.append("")

        # Total per tahun
        yearly_total = df.groupby('Tahun')['Kasus'].sum().sort_index()
        parts.append("  Total semua program per tahun:")
        prev_val = None
        for yr, val in yearly_total.items():
            yoy_str = ""
            if prev_val is not None and prev_val > 0:
                yoy = (val - prev_val) / prev_val * 100
                yoy_str = f" ({yoy:+.1f}% YoY)"
            parts.append(f"    {int(yr)}: {int(val):,} kasus{yoy_str}")
            prev_val = val
        parts.append("")

        # Per program per tahun
        parts.append("  Per program per tahun:")
        for prog in active_progs:
            sub = df[df['Kategori'] == prog].sort_values('Tahun')
            if sub.empty:
                continue
            row_parts = []
            prev_k = None
            for _, row in sub.iterrows():
                k = int(row['Kasus'])
                yr = int(row['Tahun'])
                if prev_k is not None and prev_k > 0:
                    yoy = (k - prev_k) / prev_k * 100
                    row_parts.append(f"{yr}: {k:,} ({yoy:+.1f}%)")
                else:
                    row_parts.append(f"{yr}: {k:,}")
                prev_k = k
            parts.append(f"    {prog}: {' → '.join(row_parts)}")

        if has_nom and 'Nominal' in df.columns:
            parts.append("")
            parts.append("  Nominal per program per tahun:")
            for prog in active_progs:
                sub = df[df['Kategori'] == prog].sort_values('Tahun')
                if sub.empty or 'Nominal' not in sub.columns:
                    continue
                nom_parts = [
                    f"{int(r['Tahun'])}: {_fmt_nominal(float(r['Nominal']))}"
                    for _, r in sub.iterrows()
                ]
                parts.append(f"    {prog}: {' → '.join(nom_parts)}")

        parts.append("")

    # Total agregat
    if any(w in q for w in ["total", "keseluruhan", "semua", "agregat"]):
        total_kasus = int(df[df['Tahun'] == latest_year]['Kasus'].sum())
        parts.append(f"--- Total Semua Program {latest_year}: {total_kasus:,} kasus ---")
        if has_nom and 'Nominal' in df.columns:
            total_nom = df[df['Tahun'] == latest_year]['Nominal'].sum()
            parts.append(f"--- Total Nominal {latest_year}: {_fmt_nominal(total_nom)} ---")
        parts.append("")

    # ── SEBAB KLAIM ───────────────────────────────────────────────────────────
    raw_for_sebab = df_raw
    if raw_for_sebab is None:
        try:
            import streamlit as st
            raw_for_sebab = st.session_state.get('df_raw_for_ai', None)
        except Exception:
            pass

    if raw_for_sebab is not None and (_is_sebab_question(q) or _is_covid_question(q)):
        # Untuk pertanyaan tren/COVID: tampilkan multi-tahun
        multi_year = _is_trend_question(q) or _is_covid_question(q)
        q_years    = _extract_years_from_question(question, latest_year)
        sebab_ctx  = _build_sebab_context(
            raw_for_sebab, q, latest_year, has_nom,
            multi_year=multi_year, specific_years=q_years
        )
        if sebab_ctx:
            parts.append(sebab_ctx)

    parts.append(
        "CATATAN: Data di atas adalah agregasi dari dataset internal. "
        "PRIORITASKAN data ini di atas hasil web search."
    )

    return "\n".join(parts)


def _build_sebab_context(
    df_raw: pd.DataFrame,
    question: str,
    latest_year: int,
    has_nom: bool,
    multi_year: bool = False,
    specific_years: list[int] | None = None,
) -> str:
    """
    Buat ringkasan sebab klaim dari raw dataframe.
    multi_year=True: tampilkan semua tahun (untuk pertanyaan tren/dampak).
    specific_years: filter ke tahun tertentu jika ada.
    """
    if df_raw is None or df_raw.empty:
        return ""

    cols_upper = {c.upper(): c for c in df_raw.columns}

    # ── Deteksi kolom ─────────────────────────────────────────────────────────
    sebab_col = None
    for candidate in ['SEBAB KLAIM', 'SEBAB_KLAIM', 'SEBAB', 'ALASAN', 'REASON', 'CAUSE']:
        if candidate in cols_upper:
            sebab_col = cols_upper[candidate]
            break
    if sebab_col is None:
        return ""

    prog_col = None
    for candidate in ['PROGRAM', 'KATEGORI', 'CATEGORY', 'JENIS']:
        if candidate in cols_upper:
            prog_col = cols_upper[candidate]
            break
    if prog_col is None:
        return ""

    kasus_col = None
    for candidate in ['TOTAL KASUS', 'KASUS', 'AKTUAL_KASUS', 'TOTAL_KASUS', 'COUNT']:
        if candidate in cols_upper:
            kasus_col = cols_upper[candidate]
            break
    if kasus_col is None:
        return ""

    nom_col = None
    if has_nom:
        for candidate in ['TOTAL NOMINAL', 'NOMINAL', 'AKTUAL_NOMINAL', 'TOTAL_NOMINAL', 'AMOUNT']:
            if candidate in cols_upper:
                nom_col = cols_upper[candidate]
                break

    tahun_col = None
    for candidate in ['PERIODE', 'TAHUN', 'DATE', 'YEAR']:
        if candidate in cols_upper:
            tahun_col = cols_upper[candidate]
            break

    try:
        work = df_raw.copy()
        work[kasus_col] = pd.to_numeric(work[kasus_col], errors='coerce').fillna(0)
        if nom_col:
            work[nom_col] = pd.to_numeric(work[nom_col], errors='coerce').fillna(0)

        # ── Parsing tahun ─────────────────────────────────────────────────────
        if tahun_col:
            raw_date = work[tahun_col].astype(str).str.strip()
            yyyymmdd = raw_date.str.match(r'^\d{8}$')
            if yyyymmdd.any():
                work['_tahun'] = raw_date.str[:4].astype(int, errors='ignore')
            else:
                ym = raw_date.str.extract(r'(\d{4})', expand=False)
                work['_tahun'] = pd.to_numeric(ym, errors='coerce')

            if specific_years:
                # Filter ke tahun spesifik yang diminta
                work = work[work['_tahun'].isin(specific_years)]
            elif not multi_year:
                # Mode single-year: hanya latest
                work = work[work['_tahun'] == latest_year]
            # multi_year tanpa specific_years: pakai semua data

        if work.empty:
            return ""

        # ── Tentukan label tahun untuk header ─────────────────────────────────
        if tahun_col and '_tahun' in work.columns:
            available = sorted(work['_tahun'].dropna().unique().astype(int))
            if len(available) == 1:
                year_label = str(available[0])
            elif len(available) > 1:
                year_label = f"{available[0]}–{available[-1]} (semua tahun tersedia)"
            else:
                year_label = str(latest_year)
        else:
            year_label = str(latest_year)

        lines = [f"--- DETAIL SEBAB KLAIM (dataset internal, {year_label}) ---"]

        programs = work[prog_col].str.strip().str.upper().unique()

        for prog in sorted(programs):
            prog_df = work[work[prog_col].str.strip().str.upper() == prog]
            if prog_df.empty:
                continue

            if multi_year and tahun_col and '_tahun' in work.columns:
                # ── Mode multi-tahun: tampilkan top sebab per tahun ───────────
                prog_years = sorted(prog_df['_tahun'].dropna().unique().astype(int))
                total_prog = int(prog_df[kasus_col].sum())
                lines.append(f"\n  {prog} (total semua tahun: {total_prog:,} kasus):")

                for yr in prog_years:
                    yr_df = prog_df[prog_df['_tahun'] == yr]
                    if yr_df.empty:
                        continue
                    total_yr = int(yr_df[kasus_col].sum())
                    agg_dict = {kasus_col: 'sum'}
                    if nom_col:
                        agg_dict[nom_col] = 'sum'
                    top = (
                        yr_df.groupby(sebab_col, as_index=False)
                        .agg(agg_dict)
                        .sort_values(kasus_col, ascending=False)
                        .head(3)
                    )
                    lines.append(f"    Tahun {yr} (total: {total_yr:,} kasus):")
                    for _, row in top.iterrows():
                        sebab = str(row[sebab_col])[:60]
                        kasus = int(row[kasus_col])
                        pct   = kasus / (total_yr + 1e-9) * 100
                        line  = f"      • {sebab}: {kasus:,} kasus ({pct:.1f}%)"
                        if nom_col and row[nom_col] > 0:
                            line += f" | {_fmt_nominal(float(row[nom_col]))}"
                        lines.append(line)
            else:
                # ── Mode single-year: top-5 sebab ─────────────────────────────
                agg_dict = {kasus_col: 'sum'}
                if nom_col:
                    agg_dict[nom_col] = 'sum'
                total_prog = int(prog_df[kasus_col].sum())
                top = (
                    prog_df.groupby(sebab_col, as_index=False)
                    .agg(agg_dict)
                    .sort_values(kasus_col, ascending=False)
                    .head(5)
                )
                lines.append(f"\n  {prog} (total: {total_prog:,} kasus):")
                for _, row in top.iterrows():
                    sebab = str(row[sebab_col])[:60]
                    kasus = int(row[kasus_col])
                    pct   = kasus / (total_prog + 1e-9) * 100
                    line  = f"    • {sebab}: {kasus:,} kasus ({pct:.1f}%)"
                    if nom_col and row[nom_col] > 0:
                        line += f" | {_fmt_nominal(float(row[nom_col]))}"
                    lines.append(line)

        lines.append("")
        return "\n".join(lines)

    except Exception as e:
        return f"[Gagal memuat sebab klaim: {str(e)[:100]}]"


def safe_ml_summary(ml_result: dict) -> str:
    if not ml_result:
        return ""
    bpp = ml_result.get('best_per_program')
    if bpp is None or (hasattr(bpp, 'empty') and bpp.empty):
        return ""

    lines = ["--- Model ML per Program ---"]
    for _, row in bpp.iterrows():
        mape = row.get('MAPE (%)')
        r2   = row.get('R2')
        mape_str = f"{mape:.1f}%" if mape is not None and not np.isnan(float(mape or 0)) else "N/A"
        r2_str   = f"{r2:.3f}"   if r2   is not None and not np.isnan(float(r2   or 0)) else "N/A"
        lines.append(f"  {row['Program']}: model={row['Model']}, MAPE={mape_str}, R²={r2_str}")
    return "\n".join(lines)


def build_forecast_context(
    forecast_df: pd.DataFrame,
    target: str,
    latest_year: int,
    has_nom: bool = False,
    forecast_nom_df: pd.DataFrame = None,
) -> str:
    if forecast_df is None or (hasattr(forecast_df, 'empty') and forecast_df.empty):
        return ""

    try:
        lines = [f"--- PREDIKSI KASUS (model ML, base year={latest_year}) ---"]
        pred_df = forecast_df[forecast_df['Tahun'] > latest_year].copy()
        if pred_df.empty:
            pred_df = forecast_df.copy()
        if 'Kategori' not in pred_df.columns or target not in pred_df.columns:
            return ""

        for prog in sorted(pred_df['Kategori'].unique()):
            prog_pred = pred_df[pred_df['Kategori'] == prog].sort_values('Tahun')
            if prog_pred.empty:
                continue
            lines.append(f"  {prog}:")
            for _, row in prog_pred.iterrows():
                yr  = int(row['Tahun'])
                val = float(row[target])
                ci  = float(row.get(f'{target}_ci_pct', 0))
                lo  = float(row.get(f'{target}_lower', val * (1 - ci / 100)))
                hi  = float(row.get(f'{target}_upper', val * (1 + ci / 100)))
                lines.append(
                    f"    {yr}: {val:,.0f} kasus "
                    f"[{lo:,.0f}–{hi:,.0f}, ±{ci:.0f}%]"
                )

        lines.append("")
        lines.append("  TOTAL per tahun:")
        yearly = pred_df.groupby('Tahun')[target].sum().sort_index()
        for yr, total in yearly.items():
            if int(yr) > latest_year:
                lines.append(f"    {int(yr)}: {total:,.0f} kasus")

        if has_nom and forecast_nom_df is not None and not forecast_nom_df.empty:
            lines.append("")
            lines.append("--- PREDIKSI NOMINAL (bucket) ---")
            pred_nom = forecast_nom_df[forecast_nom_df['Tahun'] > latest_year].copy()
            if not pred_nom.empty and 'Nominal' in pred_nom.columns:
                for prog in sorted(pred_nom['Kategori'].unique()):
                    pn    = pred_nom[pred_nom['Kategori'] == prog].sort_values('Tahun')
                    parts = [
                        f"{int(r['Tahun'])}: {_fmt_nominal(float(r['Nominal']))}"
                        for _, r in pn.iterrows()
                    ]
                    if parts:
                        lines.append(f"  {prog}: {' → '.join(parts)}")

        lines.append("")
        lines.append("PENTING: Angka prediksi adalah estimasi. Semakin jauh tahun, semakin lebar rentang.")
        return "\n".join(lines)

    except Exception as e:
        return f"[Gagal memuat prediksi: {str(e)}]"