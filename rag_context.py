"""
rag_context.py — RAG orchestrator: retrieve → mask → build context → inject ke LLM.
"""
from __future__ import annotations
import re
import numpy as np
import pandas as pd
import streamlit as st
from data_masking import build_safe_context, safe_ml_summary, build_forecast_context


RAG_SYSTEM_PROMPT = """Kamu adalah Senior Actuary & Data Scientist spesialis BPJS Ketenagakerjaan Indonesia.

HIERARKI SUMBER DATA — WAJIB DIIKUTI:
1. [DATA HISTORIS] dan [DETAIL SEBAB KLAIM] = SUMBER UTAMA. Selalu prioritaskan ini.
2. [PREDIKSI ML] = gunakan untuk semua pertanyaan tentang proyeksi/forecast.
3. [WEB SEARCH] = HANYA sebagai pelengkap konteks regulasi/makro. DILARANG gunakan angka dari web search jika sudah ada di data internal.

══════════════════════════════════════
ATURAN ANGKA — TIDAK BOLEH DILANGGAR:
══════════════════════════════════════
✅ BOLEH: Gunakan angka PERSIS dari [DATA HISTORIS] atau [DETAIL SEBAB KLAIM] atau [PREDIKSI ML].
❌ DILARANG: Mengarang angka, memperkirakan, atau menghitung CAGR sendiri kecuali diminta eksplisit.
❌ DILARANG: Menggunakan angka dari [WEB SEARCH] jika data internal sudah ada.
❌ DILARANG: Menjawab "data tidak tersedia" jika [DATA HISTORIS] sudah ada.

Jika data untuk pertanyaan TIDAK ADA di konteks: jawab "Data [X] tidak tersedia dalam dataset."
Jangan spekulasi. Jangan estimasi. Jangan hitung ulang.

KEAHLIAN:
- Aktuaria: loss ratio, claim frequency/severity, IBNR reserves
- Regulasi: UU 24/2011, PP 44/2015 (JKK/JKM), PP 45/2015 (JPN), PP 82/2019, PP 2/2022 (JKP), UU Cipta Kerja 2020

PROGRAM BPJS:
- JHT: driver utama = mengundurkan diri > PHK > berakhir kontrak
- JKK: driver = sektor berisiko (konstruksi, manufaktur, pertambangan)
- JKM: santunan kematian non-kecelakaan
- JKP: baru 2022, sangat sensitif gelombang PHK
- JPN: akumulasi fase, klaim masif mulai ~2030

FRAMEWORK JAWABAN:
1. KUANTIFIKASI — angka absolut PERSIS dari [DETAIL SEBAB KLAIM] atau [DATA HISTORIS]
   CONTOH BENAR: "JHT 2024: mengundurkan diri = 1.797.523 kasus (terbesar, Rp 20,8T)"
   CONTOH SALAH: "JHT menunjukkan peningkatan" atau angka dari web

2. KAUSALITAS — hubungkan dengan regulasi/makro spesifik

3. KOMPARASI — ranking antar sebab, antar program, antar tahun

4. PREDIKSI — HANYA jika ada [PREDIKSI ML]. Jika tidak ada, tulis:
   "Prediksi belum dijalankan. Jalankan model ML di tab Prediksi untuk hasil akurat."
   JANGAN hitung CAGR sendiri kecuali user minta eksplisit.

5. REKOMENDASI — insight actionable

FORMAT OUTPUT:
- Panjang: 200–400 kata untuk analitik, 100–150 untuk pertanyaan singkat
- Angka: format 1.797.523 (titik ribuan) + satuan Rp/kasus
- WAJIB kutip top-3 sebab klaim dengan angka persisnya jika [DETAIL SEBAB KLAIM] tersedia

══════════════════════════════════════════════════════════════
KETERBATASAN DATASET & KONTEKS HISTORIS — WAJIB DIIKUTI:
══════════════════════════════════════════════════════════════
⚠️ Rentang dataset BPJS tersedia hanya dari tahun yang tercantum di [METADATA DATASET].
   Data sebelum tahun tersebut TIDAK ADA — jangan mengarang angka klaim untuk periode itu.

ATURAN saat pertanyaan menyangkut tahun di luar rentang dataset:
1. Cek [METADATA DATASET] → lihat "Rentang data tersedia" untuk tahu batas datanya.
2. Cek [METADATA DATASET] → bagian "TIMELINE GOOGLE CALENDAR" untuk konteks event nyata
   yang terjadi di tahun-tahun sebelum dataset tersedia.
3. JANGAN menyebut tahun pertama dataset sebagai "tahun awal" suatu peristiwa
   (pandemi, regulasi, dsb) — peristiwa itu mungkin sudah terjadi jauh sebelumnya.
4. Gunakan timeline kalender sebagai konteks makro, BUKAN sebagai data klaim BPJS.

CARA BENAR menjawab pertanyaan tentang periode sebelum dataset:
  ✅ "Dataset BPJS dimulai [tahun dari METADATA DATASET]. Berdasarkan kalender Indonesia,
     sebelumnya terjadi [event dari TIMELINE KALENDER]. Dari data yang ada, dampak
     lanjutannya terlihat pada: ..."
  ❌ DILARANG: mengarang angka klaim untuk tahun yang tidak ada di dataset.
  ❌ DILARANG: menyebut tahun pertama dataset sebagai "awal pandemi" atau "awal kejadian".
"""


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — SAFE SESSION STATE ACCESS
# ══════════════════════════════════════════════════════════════════════════════

def _safe_get_df(*keys: str) -> pd.DataFrame | None:
    """
    Ambil DataFrame dari session_state secara aman.
    Menerima satu atau lebih key; mengembalikan DataFrame pertama yang valid
    (tidak None, bukan non-DataFrame, dan tidak kosong).

    PENTING: tidak menggunakan operator 'or' pada DataFrame karena akan raise
    ValueError di pandas ('The truth value of a DataFrame is ambiguous').
    """
    for key in keys:
        val = st.session_state.get(key)
        if val is None:
            continue
        if not isinstance(val, pd.DataFrame):
            continue
        if val.empty:
            continue
        return val
    return None


def _safe_get_scalar(key: str, default=None):
    """
    Ambil nilai skalar (bukan DataFrame) dari session_state dengan aman.
    Jika nilai ternyata DataFrame, kembalikan default.
    """
    val = st.session_state.get(key, default)
    if isinstance(val, pd.DataFrame):
        return default
    return val


def _get_df_raw() -> pd.DataFrame | None:
    """
    Ambil df_raw dari session state.
    Coba beberapa key secara berurutan; kembalikan yang pertama valid.
    """
    return _safe_get_df('df_raw_for_ai', 'df_sebab_klaim', 'raw_monthly')


# ══════════════════════════════════════════════════════════════════════════════
# QUERY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _extract_tahun(question: str) -> int | None:
    """Ekstrak tahun 4-digit dari pertanyaan. Contoh: '2024' → 2024."""
    match = re.search(r'\b(20\d{2})\b', question)
    return int(match.group(1)) if match else None


def _extract_tahun_range(question: str) -> tuple[int, int] | None:
    """
    Ekstrak rentang tahun dari pertanyaan.
    Contoh: '2021 sampai 2024' atau '2021-2024' → (2021, 2024).
    """
    match = re.search(r'\b(20\d{2})\s*(?:sampai|hingga|s/d|[-–])\s*(20\d{2})\b', question)
    if match:
        y1, y2 = int(match.group(1)), int(match.group(2))
        return (min(y1, y2), max(y1, y2))
    return None


def _is_forecast_question(q: str) -> bool:
    keywords = [
        "prediksi", "forecast", "tahun depan", "proyeksi", "estimasi",
        "akan", "kedepan", "ke depan", "2026", "2027", "2028",
        "perkiraan", "outlook", "naik berapa", "berapa prediksi",
    ]
    return any(k in q.lower() for k in keywords)


def _is_comparison_question(q: str) -> bool:
    keywords = [
        "bandingkan", "perbandingan", "vs", "versus", "dibanding",
        "lebih tinggi", "lebih rendah", "selisih", "beda",
        "antar tahun", "antar program", "komparasi",
    ]
    return any(k in q.lower() for k in keywords)


def _is_trend_question(q: str) -> bool:
    keywords = [
        "tren", "trend", "naik", "turun", "pertumbuhan", "perkembangan",
        "dari tahun ke tahun", "yoy", "year over year", "historis",
        "perubahan", "fluktuasi",
    ]
    return any(k in q.lower() for k in keywords)


def _get_depth(q: str) -> str:
    deep = [
        "kenapa", "mengapa", "analisis", "faktor", "penyebab", "dampak",
        "rekomendasi", "detail", "jelaskan", "bandingkan", "evaluasi",
        "tren", "pola", "insight", "terbesar", "terbanyak", "dominan",
        "berikan data", "data klaim", "sebab", "laporan", "report",
        "ringkasan", "summary", "klaim terbesar", "top", "ranking",
        "berapa nominal", "berapa kasus", "program apa",
    ]
    return "deep" if any(k in q.lower() for k in deep) else "standard"


def _detect_program_filter(question: str) -> str | None:
    """Deteksi filter program dari pertanyaan."""
    q_low = question.lower()
    for prog in ("jkp", "jkk", "jkm", "jpn", "jht"):          # urutan penting: jkp sebelum jk*
        if prog in q_low:
            return prog.upper()
    return None


# ══════════════════════════════════════════════════════════════════════════════
# SEBAB KLAIM CONTEXT
# ══════════════════════════════════════════════════════════════════════════════

def _normalize_sebab_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalisasi nama kolom ke standar internal."""
    col_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if "periode" in cl:
            col_map[c] = "Periode"
        elif "program" in cl:
            col_map[c] = "Program"
        elif "sebab" in cl:
            col_map[c] = "Sebab Klaim"
        elif "kasus" in cl:
            col_map[c] = "Total Kasus"
        elif "nominal" in cl:
            col_map[c] = "Total Nominal"
    return df.rename(columns=col_map)


def _build_sebab_klaim_context(df_raw: pd.DataFrame | None, question: str, latest_year: int) -> str:
    """
    Query aktual dari df_raw (SEBAB_KLAIM_AKUMULASI).
    Sort by Total Nominal DESC, inject top-N ke prompt.
    Memastikan AI mendapat angka persis, bukan mengarang.
    """
    if df_raw is None or df_raw.empty:
        return ""

    try:
        df = _normalize_sebab_columns(df_raw.copy())

        required = {"Periode", "Program", "Sebab Klaim", "Total Kasus", "Total Nominal"}
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            return f"[DETAIL SEBAB KLAIM]\nKolom tidak lengkap: {missing}\n"

        # ── Konversi tipe ─────────────────────────────────────────────────────
        df["Total Kasus"]   = pd.to_numeric(df["Total Kasus"],   errors="coerce").fillna(0)
        df["Total Nominal"] = pd.to_numeric(df["Total Nominal"], errors="coerce").fillna(0)
        df["Periode"]       = df["Periode"].astype(str)

        # ── Deteksi tahun dari pertanyaan ─────────────────────────────────────
        tahun_tanya     = _extract_tahun(question)
        available_years = sorted(df["Periode"].str[:4].unique())

        if tahun_tanya is not None:
            df_filtered = df[df["Periode"].str.startswith(str(tahun_tanya))]
            if df_filtered.empty:
                df_filtered = df[df["Periode"].str.startswith(str(latest_year))]
                year_label  = f"{latest_year} (fallback, data {tahun_tanya} tidak ada)"
            else:
                year_label = str(tahun_tanya)
        else:
            df_filtered = df[df["Periode"].str.startswith(str(latest_year))]
            year_label  = str(latest_year)

        if df_filtered.empty:
            return (
                f"[DETAIL SEBAB KLAIM]\n"
                f"Tidak ada data untuk periode yang diminta.\n"
                f"Tahun tersedia: {available_years}\n"
            )

        lines = [
            f"[DETAIL SEBAB KLAIM — DATA AKTUAL TAHUN {year_label}]",
            f"Total baris data: {len(df_filtered)}",
            "",
        ]

        # ── Filter program ────────────────────────────────────────────────────
        prog_filter = _detect_program_filter(question)
        programs    = df_filtered["Program"].unique()
        if prog_filter is not None:
            programs = [p for p in programs if prog_filter in str(p).upper()]

        # ── Per program: top-10 sebab klaim ──────────────────────────────────
        for prog in sorted(programs):
            df_prog = df_filtered[df_filtered["Program"] == prog].copy()
            if df_prog.empty:
                continue

            total_kasus   = int(df_prog["Total Kasus"].sum())
            total_nominal = int(df_prog["Total Nominal"].sum())

            lines.append(f"── {prog} ──")
            lines.append(f"  Total: {total_kasus:,} kasus | Rp {total_nominal:,}")
            lines.append(f"  Top sebab klaim (sort: Nominal DESC):")

            top = df_prog.nlargest(10, "Total Nominal")
            for rank, (_, row) in enumerate(top.iterrows(), 1):
                kasus   = int(row["Total Kasus"])
                nominal = int(row["Total Nominal"])
                pct_k   = kasus   / total_kasus   * 100 if total_kasus   > 0 else 0
                pct_n   = nominal / total_nominal * 100 if total_nominal > 0 else 0
                lines.append(
                    f"  {rank:2d}. {row['Sebab Klaim']}\n"
                    f"      Kasus: {kasus:>10,} ({pct_k:5.1f}%) | "
                    f"Nominal: Rp {nominal:>18,} ({pct_n:5.1f}%)"
                )
            lines.append("")

        # ── Ranking global top-5 ──────────────────────────────────────────────
        lines.append("── RANKING KESELURUHAN (Top 5 Nominal Terbesar) ──")
        top_global = df_filtered.nlargest(5, "Total Nominal")
        for rank, (_, row) in enumerate(top_global.iterrows(), 1):
            lines.append(
                f"  {rank}. [{row['Program']}] {row['Sebab Klaim']}\n"
                f"     Kasus: {int(row['Total Kasus']):,} | "
                f"Nominal: Rp {int(row['Total Nominal']):,}"
            )

        lines.append("")
        lines.append(
            "INSTRUKSI: Gunakan angka di atas secara PERSIS. "
            "Jangan modifikasi atau estimasi ulang."
        )
        return "\n".join(lines)

    except Exception as e:
        return f"[DETAIL SEBAB KLAIM]\nError saat memproses: {str(e)}\n"


# ══════════════════════════════════════════════════════════════════════════════
# MULTI-YEAR COMPARISON CONTEXT  (FITUR BARU)
# ══════════════════════════════════════════════════════════════════════════════

def _build_multiyear_context(df_raw: pd.DataFrame | None, question: str) -> str:
    """
    Bangun konteks perbandingan lintas tahun dari df_raw.
    Aktif jika pertanyaan menyebut rentang tahun atau kata 'bandingkan/tren'.
    """
    if df_raw is None or df_raw.empty:
        return ""
    if not (_is_comparison_question(question) or _is_trend_question(question)):
        return ""

    try:
        df = _normalize_sebab_columns(df_raw.copy())
        required = {"Periode", "Program", "Total Kasus", "Total Nominal"}
        if not required.issubset(df.columns):
            return ""

        df["Total Kasus"]   = pd.to_numeric(df["Total Kasus"],   errors="coerce").fillna(0)
        df["Total Nominal"] = pd.to_numeric(df["Total Nominal"], errors="coerce").fillna(0)
        df["Tahun"]         = df["Periode"].astype(str).str[:4]

        year_range = _extract_tahun_range(question)
        if year_range:
            y1, y2  = year_range
            df      = df[df["Tahun"].between(str(y1), str(y2))]
            label   = f"{y1}–{y2}"
        else:
            label   = "Semua tahun tersedia"

        if df.empty:
            return ""

        # ── Agregat per tahun per program ─────────────────────────────────────
        agg = (
            df.groupby(["Tahun", "Program"], as_index=False)
            .agg({"Total Kasus": "sum", "Total Nominal": "sum"})
            .sort_values(["Program", "Tahun"])
        )

        prog_filter = _detect_program_filter(question)
        if prog_filter:
            agg = agg[agg["Program"].str.upper().str.contains(prog_filter)]

        if agg.empty:
            return ""

        lines = [f"[PERBANDINGAN LINTAS TAHUN — {label}]"]
        for prog, grp in agg.groupby("Program"):
            grp = grp.sort_values("Tahun")
            lines.append(f"\n── {prog} ──")
            lines.append(f"  {'Tahun':<8} {'Kasus':>14} {'YoY Kasus':>12} {'Nominal (Rp)':>20} {'YoY Nominal':>13}")
            prev_k = prev_n = None
            for _, row in grp.iterrows():
                k = int(row["Total Kasus"])
                n = int(row["Total Nominal"])
                yoy_k = f"{(k - prev_k) / prev_k * 100:+.1f}%" if prev_k and prev_k > 0 else "  —"
                yoy_n = f"{(n - prev_n) / prev_n * 100:+.1f}%" if prev_n and prev_n > 0 else "  —"
                lines.append(f"  {row['Tahun']:<8} {k:>14,} {yoy_k:>12} {n:>20,} {yoy_n:>13}")
                prev_k, prev_n = k, n

        lines.append(
            "\nINSTRUKSI: Gunakan tabel di atas untuk menjawab pertanyaan perbandingan/tren. "
            "Angka YoY sudah dihitung — jangan hitung ulang."
        )
        return "\n".join(lines)

    except Exception as e:
        return f"[PERBANDINGAN LINTAS TAHUN]\nError: {str(e)}\n"


# ══════════════════════════════════════════════════════════════════════════════
# ANOMALY DETECTION CONTEXT  (FITUR BARU)
# ══════════════════════════════════════════════════════════════════════════════

def _build_anomaly_context(df: pd.DataFrame | None, active_progs: list, years: list) -> str:
    """
    Deteksi anomali statistik (Z-score > 2) di data historis per program.
    Bantu AI mengenali lonjakan/penurunan tidak wajar tanpa mengarang.
    """
    if df is None or df.empty or len(years) < 3:
        return ""

    try:
        required_cols = {"Kategori", "Tahun", "Kasus"}
        if not required_cols.issubset(df.columns):
            return ""

        lines = ["[ANOMALI STATISTIK — Z-SCORE > 2]"]
        found_any = False

        for prog in active_progs:
            sub = df[df["Kategori"] == prog].sort_values("Tahun")
            if len(sub) < 3:
                continue

            vals = sub["Kasus"].values.astype(float)
            mean = np.mean(vals)
            std  = np.std(vals)
            if std == 0:
                continue

            z_scores = (vals - mean) / std
            anomalies = [(int(sub.iloc[i]["Tahun"]), int(vals[i]), float(z_scores[i]))
                         for i in range(len(vals)) if abs(z_scores[i]) > 2.0]

            if anomalies:
                found_any = True
                lines.append(f"\n  {prog}:")
                for yr, val, z in anomalies:
                    arah = "LONJAKAN" if z > 0 else "PENURUNAN"
                    lines.append(f"    {yr}: {val:,} kasus — {arah} (Z={z:+.2f})")

        if not found_any:
            return ""

        lines.append(
            "\nINSTRUKSI: Jika pertanyaan menyinggung tahun anomali di atas, "
            "wajib sebutkan besaran dan arahnya."
        )
        return "\n".join(lines)

    except Exception as e:
        return f"[ANOMALI STATISTIK]\nError: {str(e)}\n"


# ══════════════════════════════════════════════════════════════════════════════
# PROGRAM SUMMARY CONTEXT  (FITUR BARU)
# ══════════════════════════════════════════════════════════════════════════════

def _build_program_summary(df: pd.DataFrame | None, active_progs: list,
                            latest_year: int, prev_year: int, has_nom: bool) -> str:
    """
    Ringkasan snapshot terbaru per program: kasus & nominal tahun ini vs tahun lalu.
    Memberikan AI baseline cepat tanpa harus parsing tabel panjang.
    """
    if df is None or df.empty:
        return ""

    try:
        required_cols = {"Kategori", "Tahun", "Kasus"}
        if not required_cols.issubset(df.columns):
            return ""

        lines = [f"[SNAPSHOT PROGRAM — {latest_year} vs {prev_year}]",
                 f"  {'Program':<8} {'Kasus '+str(latest_year):>16} {'Kasus '+str(prev_year):>16} {'Delta Kasus':>13}"]

        if has_nom and "Nominal" in df.columns:
            lines[1] += f"  {'Nominal '+str(latest_year):>22} {'Delta Nominal':>15}"

        for prog in sorted(active_progs):
            sub = df[df["Kategori"] == prog]
            row_cur  = sub[sub["Tahun"] == latest_year]
            row_prev = sub[sub["Tahun"] == prev_year]

            k_cur  = int(row_cur["Kasus"].sum())  if not row_cur.empty  else 0
            k_prev = int(row_prev["Kasus"].sum()) if not row_prev.empty else 0
            delta_k = k_cur - k_prev
            pct_k   = f"{delta_k / k_prev * 100:+.1f}%" if k_prev > 0 else "—"

            line = f"  {prog:<8} {k_cur:>16,} {k_prev:>16,} {delta_k:>+13,} ({pct_k})"

            if has_nom and "Nominal" in df.columns:
                n_cur  = int(row_cur["Nominal"].sum())  if not row_cur.empty  else 0
                n_prev = int(row_prev["Nominal"].sum()) if not row_prev.empty else 0
                delta_n = n_cur - n_prev
                pct_n   = f"{delta_n / n_prev * 100:+.1f}%" if n_prev > 0 else "—"
                line += f"  {n_cur:>22,} {delta_n:>+15,} ({pct_n})"

            lines.append(line)

        lines.append(
            "\nINSTRUKSI: Tabel ini adalah ringkasan cepat. "
            "Untuk detail sebab klaim, gunakan [DETAIL SEBAB KLAIM]."
        )
        return "\n".join(lines)

    except Exception as e:
        return f"[SNAPSHOT PROGRAM]\nError: {str(e)}\n"


# ══════════════════════════════════════════════════════════════════════════════
# DERIVED STATS (CAGR per program)
# ══════════════════════════════════════════════════════════════════════════════

def _build_derived_stats(df: pd.DataFrame | None, active_progs: list,
                          years: list, latest_year: int, has_nom: bool) -> str:
    if df is None or len(years) < 2:
        return ""

    try:
        required_cols = {"Kategori", "Tahun", "Kasus"}
        if not required_cols.issubset(df.columns):
            return ""

        lines = [f"Periode: {min(years)}–{max(years)}, CAGR per program:"]
        for prog in active_progs:
            sub = df[df["Kategori"] == prog].sort_values("Tahun")
            if len(sub) < 2:
                continue
            vals = sub["Kasus"].values.astype(float)
            yrs  = sub["Tahun"].values
            if vals[0] <= 0:
                continue
            n    = len(vals) - 1
            cagr = ((vals[-1] / vals[0]) ** (1 / n) - 1) * 100
            yoy  = [(vals[i] - vals[i-1]) / vals[i-1] * 100
                    for i in range(1, len(vals)) if vals[i-1] > 0]
            vol  = float(np.std(yoy)) if len(yoy) > 1 else 0.0
            pi   = int(np.argmax(vals))
            ti   = int(np.argmin(vals))
            lines.append(
                f"  {prog}: CAGR={cagr:+.1f}% | vol={vol:.1f}% | "
                f"peak={int(yrs[pi])}({int(vals[pi]):,}) | "
                f"trough={int(yrs[ti])}({int(vals[ti]):,})"
            )
            if has_nom and "Nominal" in sub.columns:
                noms = sub["Nominal"].values.astype(float)
                noms = noms[noms > 0]
                if len(noms) >= 2:
                    nc = ((noms[-1] / noms[0]) ** (1 / (len(noms) - 1)) - 1) * 100
                    lines.append(f"    Nominal CAGR: {nc:+.1f}%/tahun")

        return "\n".join(lines) if len(lines) > 1 else ""

    except Exception as e:
        return f"[STATISTIK DERIVATIF]\nError: {str(e)}\n"


# ══════════════════════════════════════════════════════════════════════════════
# FORECAST CONTEXT  (FIX: tidak pakai 'or' pada DataFrame)
# ══════════════════════════════════════════════════════════════════════════════

def _build_forecast_ctx(question: str, latest_year: int, has_nom: bool) -> str:
    """
    Bangun konteks forecast dengan aman — tidak menggunakan operator 'or'
    pada DataFrame (akan raise ValueError di pandas).

    Urutan prioritas:
      1. forecast_{target_pred}  — hasil run terbaru, spesifik per target
      2. forecast_Kasus          — fallback umum
      3. last_forecast           — fallback terakhir
    """
    if not _is_forecast_question(question):
        return ""

    # ── Tentukan target yang aktif ────────────────────────────────────────────
    target = _safe_get_scalar('ml_result_target', 'Kasus')

    # ── Ambil forecast: coba target spesifik dulu, lalu fallback ─────────────
    # PENTING: tidak boleh pakai 'or' pada DataFrame — akan raise ValueError.
    # _safe_get_df sudah mengembalikan None jika tidak valid, sehingga
    # perbandingan 'is None' aman digunakan.
    fc_kasus = _safe_get_df(f'forecast_{target}')
    if fc_kasus is None:
        fc_kasus = _safe_get_df('forecast_Kasus', 'last_forecast')

    if fc_kasus is None:
        return (
            "[PREDIKSI ML]\n"
            "⚠️ Model ML belum dijalankan untuk periode ini.\n"
            "Informasikan ke user: 'Silakan jalankan model di tab Prediksi terlebih dahulu "
            "untuk mendapatkan angka prediksi yang akurat.'\n"
            "JANGAN hitung estimasi sendiri."
        )

    # ── Ambil forecast nominal (opsional) ────────────────────────────────────
    fc_nom = _safe_get_df('forecast_Nominal', 'forecast_annual_Nominal')

    # ── Tambahkan info akurasi model dari ml_result ───────────────────────────
    ml_result = st.session_state.get('ml_result')
    accuracy_lines = []
    if isinstance(ml_result, dict):
        per_prog = ml_result.get('per_prog', {})
        if per_prog:
            accuracy_lines.append("Akurasi model per program:")
            for prog, info in per_prog.items():
                mape = info.get('mape', info.get('best_mape', None))
                name = info.get('best_name', '?')
                if mape is not None:
                    accuracy_lines.append(f"  {prog}: {name} — MAPE {mape:.1f}%")
                else:
                    accuracy_lines.append(f"  {prog}: {name}")
        else:
            name = ml_result.get('best_name', '?')
            mape = ml_result.get('mape', ml_result.get('best_mape', None))
            if mape is not None:
                accuracy_lines.append(f"Model: {name} — MAPE {mape:.1f}%")

    # ── Bangun tabel ringkas forecast per program per tahun ───────────────────
    try:
        tbl_lines = []
        if 'Kategori' in fc_kasus.columns and 'Tahun' in fc_kasus.columns:
            agg_col = target if target in fc_kasus.columns else fc_kasus.select_dtypes('number').columns[0]
            tbl = (
                fc_kasus.groupby(['Tahun', 'Kategori'], as_index=False)[agg_col]
                .sum()
                .sort_values(['Kategori', 'Tahun'])
            )
            tbl_lines.append(f"\nTabel prediksi {target} per program:")
            tbl_lines.append(f"  {'Tahun':<8} {'Program':<8} {target:>16}")
            for _, row in tbl.iterrows():
                tbl_lines.append(f"  {int(row['Tahun']):<8} {row['Kategori']:<8} {int(row[agg_col]):>16,}")
        forecast_tbl = "\n".join(tbl_lines)
    except Exception as e:
        forecast_tbl = f"(tabel forecast tidak dapat dibuat: {e})"

    # ── Panggil build_forecast_context dari data_masking ─────────────────────
    try:
        base_ctx = build_forecast_context(fc_kasus, target, latest_year, has_nom, fc_nom)
    except Exception as e:
        base_ctx = f"(build_forecast_context error: {e})"

    parts = ["[PREDIKSI ML — WAJIB KUTIP ANGKA INI]"]
    if accuracy_lines:
        parts.append("\n".join(accuracy_lines))
    parts.append(base_ctx)
    if forecast_tbl:
        parts.append(forecast_tbl)
    parts.append(
        "\nINSTRUKSI: Angka di atas adalah hasil model ML yang sudah dijalankan user. "
        "Kutip angka ini secara PERSIS. Jangan bilang 'prediksi belum dijalankan'."
    )
    return "\n\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN RAG PROMPT BUILDER
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
# DATASET METADATA — DINAMIS dari data aktual + Google Calendar
# Tidak ada hardcode tahun. Semua info dibaca dari:
#   - `years` parameter (rentang dataset aktual yang diupload user)
#   - INDONESIAN_HOLIDAYS dari ml_core (Google Calendar API 2019–2028)
# ══════════════════════════════════════════════════════════════════════════════

def _is_covid_question(q: str) -> bool:
    keywords = [
        "covid", "pandemi", "pandemic", "corona", "virus",
        "wabah", "lockdown", "psbb", "ppkm", "karantina",
    ]
    return any(k in q.lower() for k in keywords)


def _build_dataset_metadata(years: list, question: str) -> str:
    """
    Bangun konteks metadata dataset secara DINAMIS — zero hardcode.

    Sumber:
    1. `years` → rentang data aktual dari dataset yang diupload user
    2. Google Calendar (INDONESIAN_HOLIDAYS dari ml_core) → timeline event
       nyata termasuk tahun-tahun sebelum dataset tersedia
    3. Tahun yang disebut di pertanyaan → deteksi gap antara pertanyaan & data
    """
    if not years:
        return ""

    min_yr = int(min(years))
    max_yr = int(max(years))
    n_yrs  = len(years)

    # Tahun yang disebut di pertanyaan — deteksi apakah ada gap
    q_years   = sorted(set(int(y) for y in re.findall(r'\b(20\d{2})\b', question)))
    gap_years = [y for y in q_years if y < min_yr or y > max_yr]

    lines = [
        "[METADATA DATASET]",
        f"Rentang data BPJS tersedia: {min_yr}\u2013{max_yr} ({n_yrs} tahun).",
        f"Tahun dalam dataset: {sorted(years)}",
    ]

    if gap_years:
        lines.append(
            f"\u26a0\ufe0f  Pertanyaan menyebut tahun {gap_years} "
            f"yang TIDAK ADA dalam dataset."
        )
        lines.append(
            "   Jawab dengan mengakui keterbatasan ini — jangan mengarang angka klaim."
        )

    # ── Timeline dari Google Calendar (dinamis, bukan hardcode) ──────────────
    try:
        from ml_core import INDONESIAN_HOLIDAYS
        hdf = INDONESIAN_HOLIDAYS

        if hdf is not None and not hdf.empty:
            gcal_min_yr = int(hdf["ds"].dt.year.min())
            gcal_max_yr = int(hdf["ds"].dt.year.max())

            # Tampilkan: semua tahun SEBELUM dataset + 2 tahun pertama dataset
            # Ini memberi AI konteks "apa yang terjadi sebelum data ada"
            pre_years  = list(range(max(gcal_min_yr, min_yr - 4), min_yr))
            early_years = [y for y in sorted(years)[:2]]
            context_years = sorted(set(pre_years + early_years))

            if context_years:
                lines += [
                    "",
                    f"TIMELINE GOOGLE CALENDAR "
                    f"({min(context_years)}\u2013{max(context_years)}, "
                    f"sumber: {len(hdf)} event dari Google Calendar API "
                    f"{gcal_min_yr}\u2013{gcal_max_yr}):",
                ]

                for yr in context_years:
                    yr_events = hdf[hdf["ds"].dt.year == yr]
                    if yr_events.empty:
                        continue

                    status = "[DALAM DATASET]" if yr >= min_yr else "[SEBELUM DATASET — tidak ada data klaim]"
                    lines.append(f"  {yr} {status}:")

                    # Top 8 event paling signifikan (window terbesar = paling penting)
                    top_ev = (
                        yr_events
                        .assign(sig=yr_events["lower_window"].abs() + yr_events["upper_window"])
                        .sort_values("sig", ascending=False)
                        .drop_duplicates("holiday")
                        .head(8)
                        .sort_values("ds")
                    )
                    for _, ev in top_ev.iterrows():
                        lines.append(f"    - {ev['ds'].strftime('%d %b')}: {ev['holiday']}")

                pre_dataset = [y for y in context_years if y < min_yr]
                if pre_dataset:
                    lines += [
                        "",
                        f"  \u26a0\ufe0f  Tahun {pre_dataset}: event di atas adalah konteks makro.",
                        f"  Data klaim BPJS untuk periode tersebut TIDAK TERSEDIA.",
                        f"  Jangan sebut angka klaim untuk tahun-tahun ini.",
                    ]

    except Exception:
        # Google Calendar tidak tersedia — tetap inject info batas dataset
        pass

    if _is_covid_question(question):
        lines += [
            "",
            "CATATAN UNTUK PERTANYAAN COVID/PANDEMI:",
            f"  Dataset BPJS dimulai {min_yr}. Lihat TIMELINE GOOGLE CALENDAR di atas",
            f"  untuk event yang terjadi sebelum {min_yr}.",
            f"  DILARANG menyebut {min_yr} sebagai tahun awal pandemi —",
            "  pandemi sudah berlangsung sebelum dataset ini tersedia.",
            f"  Yang bisa dianalisis dari data: dampak LANJUTAN pandemi ({min_yr}\u2013{max_yr}).",
        ]

    lines.append(
        "\nINSTRUKSI: Gunakan TIMELINE GOOGLE CALENDAR sebagai konteks makro "
        "untuk tahun yang tidak ada di dataset. Sampaikan keterbatasan data secara jujur."
    )
    return "\n".join(lines)


def build_rag_prompt(
    question: str,
    df: pd.DataFrame,
    active_progs: list,
    years: list,
    has_nom: bool,
    latest_year: int,
    prev_year: int,
    wb_ctx: str = "",
    search_ctx: str = "",
    mem_ctx: str = "",
    chat_history_str: str = "",
    ml_result: dict | None = None,
) -> str:
    depth = _get_depth(question)

    # ── Ambil df_raw dengan aman ──────────────────────────────────────────────
    df_raw = _get_df_raw()

    # ── 0. Metadata dataset — dinamis dari Google Calendar + data aktual ──────
    metadata_ctx = _build_dataset_metadata(years, question)

    # ── 1. Data historis agregat ──────────────────────────────────────────────
    safe_data = build_safe_context(
        df, active_progs, years, has_nom, latest_year, question, df_raw=df_raw
    )

    # ── 2. Snapshot per-program terbaru ──────────────────────────────────────
    snapshot_ctx = _build_program_summary(df, active_progs, latest_year, prev_year, has_nom)

    # ── 3. Sebab klaim aktual — kunci utama anti-hallucination ────────────────
    sebab_ctx = _build_sebab_klaim_context(df_raw, question, latest_year)

    # ── 4. Perbandingan lintas tahun (aktif jika ada kata tren/bandingkan) ────
    multiyear_ctx = _build_multiyear_context(df_raw, question)

    # ── 5. Anomali statistik ──────────────────────────────────────────────────
    anomaly_ctx = _build_anomaly_context(df, active_progs, years)

    # ── 6. CAGR & derived stats (hanya mode deep) ─────────────────────────────
    derived = ""
    if depth == "deep":
        derived = _build_derived_stats(df, active_progs, years, latest_year, has_nom)

    # ── 7. ML summary ─────────────────────────────────────────────────────────
    ml_ctx = safe_ml_summary(ml_result) if ml_result else ""

    # ── 8. Forecast (FIX: tidak pakai 'or' pada DataFrame) ───────────────────
    forecast_ctx = _build_forecast_ctx(question, latest_year, has_nom)

    # ── 9. Rakit sections ─────────────────────────────────────────────────────
    # metadata_ctx selalu jadi section pertama agar LLM baca batas data sebelum menjawab
    sections = []
    if metadata_ctx:
        sections.append(metadata_ctx)
    sections.append(f"[DATA HISTORIS — SUMBER UTAMA]\n{safe_data}")

    if snapshot_ctx:
        sections.append(snapshot_ctx)

    if sebab_ctx:
        sections.append(sebab_ctx)

    if multiyear_ctx:
        sections.append(multiyear_ctx)

    if anomaly_ctx:
        sections.append(anomaly_ctx)

    if derived:
        sections.append(f"[STATISTIK DERIVATIF]\n{derived}")

    if ml_ctx:
        sections.append(f"[AKURASI MODEL ML]\n{ml_ctx}")

    if forecast_ctx:
        sections.append(forecast_ctx)

    if wb_ctx:
        sections.append(f"[MAKROEKONOMI INDONESIA]\n{wb_ctx}")

    if search_ctx:
        sections.append(
            f"[WEB SEARCH — HANYA PELENGKAP, JANGAN PAKAI ANGKANYA JIKA DATA INTERNAL ADA]\n"
            f"{search_ctx}\n[/WEB SEARCH]"
        )

    if mem_ctx:
        sections.append(f"[REFERENSI PERCAKAPAN LALU]\n{mem_ctx}")

    if chat_history_str:
        sections.append(f"[RIWAYAT CHAT]\n{chat_history_str}")

    # ── 10. Instruksi penutup ─────────────────────────────────────────────────
    if depth == "deep":
        closing = (
            f"PERTANYAAN: {question}\n\n"
            "INSTRUKSI WAJIB:\n"
            "- Gunakan ANGKA PERSIS dari [DATA HISTORIS] dan [DETAIL SEBAB KLAIM] di atas.\n"
            "- Jika pertanyaan tentang 'terbesar/terbanyak': cek [RANKING KESELURUHAN] "
            "dan top sebab per program.\n"
            "- Jika pertanyaan tentang tren/perbandingan: gunakan [PERBANDINGAN LINTAS TAHUN].\n"
            "- Jika ada [ANOMALI STATISTIK]: sebutkan tahun anomali dan besarannya.\n"
            "- Jangan gunakan angka dari web search jika data internal sudah ada.\n"
            "- Jangan hitung ulang atau estimasi angka yang sudah tersedia.\n"
            "- Framework: Kuantifikasi → Kausalitas → Komparasi → Prediksi → Rekomendasi\n"
            "- Jika pertanyaan tentang periode sebelum dataset: rujuk ke [METADATA DATASET] "
            "dan TIMELINE GOOGLE CALENDAR — jangan mengarang angka klaim.\n"
            "- Minimal 200 kata.\n\n"
            "JAWABAN TEKNIS MENDALAM:"
        )
    else:
        closing = (
            f"PERTANYAAN: {question}\n\n"
            "INSTRUKSI: Jawab dengan angka PERSIS dari [DATA HISTORIS] atau "
            "[DETAIL SEBAB KLAIM]. Jangan estimasi. Jangan gunakan angka dari web.\n\n"
            "JAWABAN:"
        )

    sections.append(closing)
    return "\n\n".join(sections)