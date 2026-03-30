"""
rag_context.py — RAG orchestrator: retrieve → mask → build context → inject ke LLM.
Semua data melewati masking sebelum dikirim ke provider eksternal.
"""
from __future__ import annotations
import pandas as pd
from data_masking import build_safe_context, safe_ml_summary


# Prompt system yang lebih ketat dengan aturan data protection
RAG_SYSTEM_PROMPT = """Kamu adalah AI Analyst ahli BPJS Ketenagakerjaan Indonesia.

PROGRAM BPJS KETENAGAKERJAAN:
- JHT: tabungan hari tua, diklaim saat resign/pensiun/cacat/meninggal
- JKK: kecelakaan kerja atau penyakit akibat kerja
- JKM: santunan meninggal dunia (bukan kecelakaan kerja)
- JKP: jaminan kehilangan pekerjaan (PHK), diluncurkan 2022
- JPN: manfaat pensiun bulanan seumur hidup

ATURAN KERAS — WAJIB DIIKUTI:
1. Jawab HANYA dari data konteks yang diberikan. Jangan karang angka.
2. Data nominal hanya tersedia dalam bentuk rentang (bucket), bukan angka exact.
3. Jangan menyebut nama individu, NIK, atau identitas apapun.
4. Jika data tidak cukup untuk menjawab, katakan "data tidak tersedia".
5. Referensi regulasi jika relevan: PP 82/2019, PP 45/2015, UU Cipta Kerja 2020.
6. Integrasikan hasil web search sebagai pendukung, selalu cantumkan URL sumber.
7. Struktur: (a) Temuan data internal → (b) Konfirmasi web + URL → (c) Kesimpulan.
8. Maksimal 10 kalimat kecuali diminta lebih panjang.
"""


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
    Bangun prompt lengkap dengan RAG pattern.
    LLM hanya menerima: summary aman + context — BUKAN raw dataframe.
    """
    # 1. Retrieve + Mask data
    safe_data = build_safe_context(
        df, active_progs, years, has_nom, latest_year, question
    )

    # 2. ML summary (opsional)
    ml_ctx = safe_ml_summary(ml_result) if ml_result else ""

    # 3. Rakit prompt
    sections = []

    sections.append(f"[DATA INTERNAL — SUDAH DIAGREGASI]\n{safe_data}")

    if ml_ctx:
        sections.append(f"[MODEL ML]\n{ml_ctx}")

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