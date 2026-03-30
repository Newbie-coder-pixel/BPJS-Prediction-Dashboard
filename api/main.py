"""
api/main.py — FastAPI layer untuk /ask, /predict, /analyze.
Jalankan: uvicorn api.main:app --reload --port 8000
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Optional
import pandas as pd

from data_masking import build_safe_context, safe_ml_summary
from rag_context import build_rag_prompt, RAG_SYSTEM_PROMPT
from ai_utils import _get_api_key, _call_ai_groq, _call_ai_gemini, _detect_provider
from data_utils import load_raw, parse_dataset
from ml_core import run_ml, forecast

app = FastAPI(title="BPJS AI API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ── Schemas ───────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str
    latest_year: int
    active_programs: list[str]
    years: list[int]

class AskResponse(BaseModel):
    answer: str
    context_preview: str   # Context yang dikirim ke LLM (untuk audit/transparansi)

class AnalyzeRequest(BaseModel):
    group_by: list[str] = ["Kategori"]
    latest_year: Optional[int] = None

# ── State sederhana (in-memory, replace dengan DB di production) ──────────────
_df_store: Optional[pd.DataFrame] = None

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "data_loaded": _df_store is not None}


@app.post("/ask", response_model=AskResponse)
async def ask_endpoint(req: AskRequest):
    global _df_store
    if _df_store is None:
        raise HTTPException(status_code=400, detail="Data belum diload.")

    safe_ctx = build_safe_context(
        df           = _df_store,
        active_progs = req.active_programs,
        years        = req.years,
        has_nom      = 'Nominal' in _df_store.columns,
        latest_year  = req.latest_year,
        question     = req.question,
    )

    prompt = build_rag_prompt(
        question    = req.question,
        df          = _df_store,
        active_progs= req.active_programs,
        years       = req.years,
        has_nom     = 'Nominal' in _df_store.columns,
        latest_year = req.latest_year,
        prev_year   = req.latest_year - 1,
    )

    provider, key = _get_api_key()
    try:
        if provider == "groq":
            answer = _call_ai_groq(prompt, RAG_SYSTEM_PROMPT, key)
        elif provider == "gemini":
            answer = _call_ai_gemini(prompt, RAG_SYSTEM_PROMPT, key)
        else:
            raise HTTPException(status_code=500, detail="Provider tidak ditemukan.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return AskResponse(answer=answer, context_preview=safe_ctx[:500] + "...")


@app.post("/analyze")
async def analyze_endpoint(req: AnalyzeRequest):
    global _df_store
    if _df_store is None:
        raise HTTPException(status_code=400, detail="Data belum diload.")

    df = _df_store
    latest = req.latest_year or int(df['Tahun'].max())

    result = {}
    for col in req.group_by:
        if col in df.columns:
            agg = df[df['Tahun'] == latest].groupby(col)['Kasus'].sum()
            # Hanya kembalikan jika grup >= 3 (k-anonymity minimal)
            agg = agg[agg >= 3]
            result[col] = agg.to_dict()

    return {"year": latest, "aggregates": result, "total": int(df[df['Tahun'] == latest]['Kasus'].sum())}