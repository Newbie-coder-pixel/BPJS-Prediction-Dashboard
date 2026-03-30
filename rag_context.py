"""
rag_context.py — RAG orchestrator: retrieve → mask → build context → inject ke LLM.
Semua data melewati masking sebelum dikirim ke provider eksternal.
"""
from __future__ import annotations
import pandas as pd
import streamlit as st
from data_masking import build_safe_context, safe_ml_summary, build_forecast_context


RAG_SYSTEM_PROMPT = """Kamu adalah AI Analyst ahli BPJS Ketenagakerjaan Indonesia.

PROGRAM BPJS KETENAGAKERJAAN:
- JHT: tabungan hari tua, diklaim saat resign/pensiun/cacat/meninggal
- JKK: kecelakaan kerja atau penyakit akibat kerja
- JKM: santunan meninggal dunia (bukan kecelakaan kerja)
- JKP: jaminan kehilangan pekerjaan (PHK), diluncurkan 2022
- JPN: manfaat pensiun bulanan seumur hidup

ATURAN KERAS — WAJIB DIIKUTI:
1. Jika ada [PREDIKSI ML], WAJIB gunakan angka-angka di sana untuk menjawab.
2. Jawab SPESIFIK dengan angka: sebutkan program, tahun, jumlah kasus, rentang.
3. Data nominal hanya tersedia dalam bentuk rentang (bucket), bukan angka exact.
4. Jangan menyebut nama individu, NIK, atau identitas apapun.
5. Jika [PREDIKSI ML] kosong, jawab dari tren historis dan jelaskan prediksi ML belum dijalankan.
6. Referensi regulasi jika relevan: PP 82/2019, PP 45/2015, UU Cipta Kerja 2020.
7. Integrasikan hasil web search sebagai pendukung, selalu cantumkan URL sumber.
8. Struktur: (a) Angka prediksi per program → (b) Faktor pendorong + URL → (c) Rekomendasi.
9. Maksimal 12 kalimat kecuali diminta lebih panjang.
10. DILARANG jawaban seperti "sulit diprediksi" atau "banyak faktor" tanpa disertai angka konkret.
"""


def _is_forecast_question(question: str) -> bool:
    """Deteksi apakah pertanyaan terkait prediksi/forecast."""
    q = question.lower()
    keywords = [
        "prediksi", "forecast", "tahun depan", "proyeksi", "estimasi",
        "akan", "kedepan", "ke depan", "masa depan", "2026", "2027",
        "perkiraan", "outlook", "trend ke depan", "naik berapa",
    ]
    return any(k in q for k in keywords)


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
    """
    Bangun prompt RAG lengkap.
    Untuk pertanyaan prediksi: inject angka forecast dari session state.
    """

    # 1. Data historis ter-mask
    safe_data = build_safe_context(
        df, active_progs, years, has_nom, latest_year, question
    )

    # 2. ML accuracy summary
    ml_ctx = safe_ml_summary(ml_result) if ml_result else ""

    # 3. ── KUNCI FIX: Ambil angka prediksi dari session state ─────────────────
    forecast_ctx = ""
    if _is_forecast_question(question):
        # Ambil hasil forecast yang sudah dihitung di tab Prediksi / tab ML
        forecast_kasus = st.session_state.get('forecast_Kasus', None)
        forecast_nom   = st.session_state.get('forecast_Nominal', None)

        # Fallback: coba dari last_forecast jika forecast_Kasus tidak ada
        if forecast_kasus is None:
            forecast_kasus = st.session_state.get('last_forecast', None)

        # Fallback kedua: coba bangun dari ml_result per_prog history
        if forecast_kasus is None and ml_result is not None:
            forecast_ctx = _build_forecast_from_ml_result(ml_result, latest_year)
        else:
            ml_target = st.session_state.get('ml_result_target', 'Kasus')
            forecast_ctx = build_forecast_context(
                forecast_df     = forecast_kasus,
                target          = ml_target,
                latest_year     = latest_year,
                has_nom         = has_nom,
                forecast_nom_df = forecast_nom,
            )

    # 4. Rakit semua section
    sections = []
    sections.append(f"[DATA HISTORIS — SUDAH DIAGREGASI]\n{safe_data}")

    if ml_ctx:
        sections.append(f"[AKURASI MODEL ML]\n{ml_ctx}")

    if forecast_ctx:
        sections.append(f"[PREDIKSI ML — GUNAKAN ANGKA INI UNTUK MENJAWAB]\n{forecast_ctx}")
    elif _is_forecast_question(question):
        sections.append(
            "[PREDIKSI ML]\n"
            "Data prediksi belum tersedia. User perlu:\n"
            "1. Buka tab ML → klik 'Jalankan Analisis ML'\n"
            "2. Buka tab Prediksi → klik 'Generate Prediksi'\n"
            "Setelah itu, angka prediksi spesifik akan tersedia.\n"
            "Untuk sementara, jawab berdasarkan tren historis saja."
        )

    if wb_ctx:
        sections.append(f"[MAKROEKONOMI INDONESIA]\n{wb_ctx}")

    if search_ctx:
        sections.append(f"[HASIL WEB SEARCH]\n{search_ctx}\n[/WEB SEARCH]")

    if mem_ctx:
        sections.append(f"[REFERENSI PERCAKAPAN LALU]\n{mem_ctx}")

    if chat_history_str:
        sections.append(f"[RIWAYAT CHAT]\n{chat_history_str}")

    sections.append(f"PERTANYAAN: {question}\n\nJAWABAN:")

    return "\n\n".join(sections)


def _build_forecast_from_ml_result(ml_result: dict, latest_year: int) -> str:
    """
    Fallback: estimasi kasar dari history per_prog di ml_result
    jika forecast DataFrame belum dibuat user.
    """
    try:
        per_prog = ml_result.get('per_prog', {})
        if not per_prog:
            return ""

        lines = [
            f"--- ESTIMASI PREDIKSI (dari tren historis model, base={latest_year}) ---",
            "CATATAN: Ini estimasi kasar. Untuk prediksi akurat, jalankan tab Prediksi.",
            ""
        ]

        for prog, info in per_prog.items():
            history = info.get('history', [])
            if len(history) < 2:
                continue

            last_val  = float(history[-1])
            prev_val  = float(history[-2])
            growth    = (last_val - prev_val) / (prev_val + 1e-9)
            # Cap growth agar tidak absurd
            growth    = max(-0.3, min(0.3, growth))
            pred_next = last_val * (1 + growth)
            mape      = info.get('metrics', {}).get('MAPE (%)', 15.0) or 15.0
            ci        = min(mape / 100, 0.25)

            lines.append(
                f"  {prog}: estimasi {latest_year + 1} = {pred_next:,.0f} kasus "
                f"[rentang: {pred_next*(1-ci):,.0f}–{pred_next*(1+ci):,.0f}] "
                f"(berdasarkan pertumbuhan {growth*100:+.1f}% dari {latest_year})"
            )

        return "\n".join(lines)
    except Exception:
        return ""