"""
ai_utils.py — AI caller (Groq/Gemini/Anthropic), World Bank API,
               web search (Tavily/DDG), chat session manager, QA memory.
"""
import os
import re
import json
import hashlib
import time
import numpy as np
import pandas as pd
import streamlit as st

from auth import _secret
from rag_context import build_rag_prompt, RAG_SYSTEM_PROMPT   # ← tambah ini
from data_masking import build_safe_context                    # ← tambah ini

# ══════════════════════════════════════════════════════════════════════════════
# PROVIDER DETECTION & API KEYS
# ══════════════════════════════════════════════════════════════════════════════

def _detect_provider(key: str) -> str:
    if not key:
        return ""
    k = key.strip()
    if k.startswith("AIza"):    return "gemini"
    if k.startswith("gsk_"):    return "groq"
    if k.startswith("sk-ant-"): return "anthropic"
    return "gemini"


def _get_api_key() -> tuple:
    candidates = []
    for secret_name in ["GEMINI_API_KEY", "gemini_api_key",
                        "GROQ_API_KEY", "groq_api_key",
                        "ANTHROPIC_API_KEY", "anthropic_api_key"]:
        try:
            val = _secret(secret_name, "").strip()
            if val:
                provider = _detect_provider(val)
                if provider:
                    candidates.append((provider, val))
        except Exception:
            pass
    if not candidates:
        return ("", "")
    for pref in ("groq", "gemini", "anthropic"):
        for p, k in candidates:
            if p == pref:
                return (p, k)
    return candidates[0]


# ══════════════════════════════════════════════════════════════════════════════
# AI CALLERS
# ══════════════════════════════════════════════════════════════════════════════

def _call_ai_groq(prompt: str, system: str, key: str, max_tokens: int = 800) -> str:
    import urllib.request, json as _j, urllib.error
    groq_models = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "meta-llama/llama-4-scout-17b-16e-instruct",
    ]
    url  = "https://api.groq.com/openai/v1/chat/completions"
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    last_err = None
    for model_name in groq_models:
        body = _j.dumps({"model": model_name, "messages": msgs,
                         "max_tokens": max_tokens, "temperature": 0.4}).encode("utf-8")
        req = urllib.request.Request(
            url, data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                "Accept": "application/json",
                "Origin": "https://console.groq.com",
                "Referer": "https://console.groq.com/",
            }, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                resp = _j.loads(r.read().decode())
            return resp["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            body_err = e.read().decode("utf-8", errors="ignore")
            last_err = f"HTTP {e.code} ({model_name}) — {body_err[:300]}"
            if e.code in (404, 400): continue
            raise RuntimeError(last_err)
        except Exception as ex:
            last_err = str(ex); continue
    raise RuntimeError(last_err or "Semua model Groq gagal.")


def _call_ai_gemini(prompt: str, system: str, key: str, max_tokens: int = 800) -> str:
    import urllib.request, json as _j, urllib.error
    models_to_try = ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash-002"]
    contents = []
    if system:
        contents.append({"role": "user",  "parts": [{"text": f"[System]: {system}"}]})
        contents.append({"role": "model", "parts": [{"text": "Understood."}]})
    contents.append({"role": "user", "parts": [{"text": prompt}]})
    payload = _j.dumps({"contents": contents,
                         "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.4}}).encode("utf-8")
    last_err = None
    for model in models_to_try:
        url = (f"https://generativelanguage.googleapis.com/v1beta"
               f"/models/{model}:generateContent?key={key}")
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, data=payload,
                    headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(req, timeout=30) as r:
                    resp = _j.loads(r.read().decode())
                return resp["candidates"][0]["content"]["parts"][0]["text"].strip()
            except urllib.error.HTTPError as e:
                last_err = f"HTTP {e.code} ({model})"
                if e.code == 429:
                    time.sleep(2 ** (attempt + 1)); continue
                break
            except Exception as ex:
                last_err = str(ex); break
    raise RuntimeError(last_err or "Semua model Gemini gagal.")


def _call_ai(prompt: str, system: str = "", api_info=None, max_tokens: int = 800):
    if api_info is None:
        api_info = _get_api_key()
    provider, key = api_info if (isinstance(api_info, tuple) and len(api_info) == 2) else ("", "")
    if not key:
        return None
    detected = _detect_provider(key)
    if detected and detected != provider:
        provider = detected
    try:
        if provider == "groq":
            return _call_ai_groq(prompt, system, key, max_tokens)
        elif provider == "gemini":
            return _call_ai_gemini(prompt, system, key, max_tokens)
        elif provider == "anthropic":
            import urllib.request, json as _j
            body = _j.dumps({"model": "claude-haiku-4-5-20251001", "max_tokens": max_tokens,
                              "system": system, "messages": [{"role": "user", "content": prompt}]}).encode("utf-8")
            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages", data=body,
                headers={"Content-Type": "application/json", "x-api-key": key,
                         "anthropic-version": "2023-06-01"}, method="POST")
            with urllib.request.urlopen(req, timeout=30) as r:
                resp = _j.loads(r.read().decode())
            return resp["content"][0]["text"].strip()
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# WORLD BANK API
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=86400, show_spinner=False)
def _fetch_worldbank(years_tuple: tuple) -> dict:
    import urllib.request, json as _j
    indicators = {
        'NY.GDP.MKTP.KD.ZG': 'gdp_pct',
        'FP.CPI.TOTL.ZG':    'inflation_pct',
        'SL.UEM.TOTL.ZS':    'unemployment_pct',
        'NE.EXP.GNFS.KD.ZG': 'export_growth',
    }
    result = {yr: {} for yr in years_tuple}
    for code, key in indicators.items():
        url = (f"https://api.worldbank.org/v2/country/ID/indicator/{code}"
               f"?format=json&date={min(years_tuple)}:{max(years_tuple)}&per_page=30")
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=12) as r:
                data = _j.loads(r.read().decode())
            if isinstance(data, list) and len(data) >= 2:
                for entry in (data[1] or []):
                    try:
                        yr  = int(entry["date"])
                        val = entry["value"]
                        if val is not None and yr in result:
                            result[yr][key] = round(float(val), 2)
                    except Exception:
                        pass
        except Exception:
            pass
    return result


def _ai_analyze_peak_trough(prog_name, peak_yr, peak_val, trough_yr, trough_val, wb_data, api_info):
    import json as _j
    ekon_lines = []
    for yr in sorted(wb_data.keys()):
        d = wb_data[yr]
        parts = []
        if 'gdp_pct'          in d: parts.append(f"PDB {d['gdp_pct']:+.2f}%")
        if 'inflation_pct'    in d: parts.append(f"inflasi {d['inflation_pct']:.1f}%")
        if 'unemployment_pct' in d: parts.append(f"pengangguran {d['unemployment_pct']:.1f}%")
        if 'export_growth'    in d: parts.append(f"ekspor {d['export_growth']:+.1f}%")
        ekon_lines.append(f"  {yr}: {', '.join(parts) if parts else 'data terbatas'}")
    ekon_str = "\n".join(ekon_lines)
    prompt = (
        f"Kamu analis ketenagakerjaan Indonesia. "
        f"Program BPJS: {prog_name} | PEAK: {peak_yr} ({peak_val:,} kasus) | TROUGH: {trough_yr} ({trough_val:,} kasus). "
        f"Data makroekonomi Indonesia (World Bank):\n{ekon_str}\n"
        f"Jelaskan mengapa klaim TINGGI di {peak_yr} dan RENDAH di {trough_yr}. "
        f"Jawab HANYA JSON: "
        + '{"peak_label":"emoji+3kata","peak_desc":"2-3 kalimat+angka","trough_label":"emoji+3kata","trough_desc":"2-3 kalimat+angka"}'
    )
    try:
        raw = _call_ai(prompt, api_info=api_info, max_tokens=500)
        if not raw:
            return None
        if "```" in raw:
            raw = re.sub(r"```[a-z]*\n?", "", raw).replace("```", "").strip()
        j_start, j_end = raw.find("{"), raw.rfind("}") + 1
        if j_start >= 0:
            raw = raw[j_start:j_end]
        return _j.loads(raw)
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# WEB SEARCH
# ══════════════════════════════════════════════════════════════════════════════

def _web_search_tavily(query: str, max_results: int = 4) -> str:
    import urllib.request, json
    tavily_key = _secret("TAVILY_API_KEY", "")
    if not tavily_key:
        return ""
    try:
        body = json.dumps({
            "api_key": tavily_key,
            "query": query + " Indonesia BPJS ketenagakerjaan jaminan sosial",
            "search_depth": "advanced",
            "max_results": max_results,
            "include_answer": True,
            "include_domains": [
                "bpjsketenagakerjaan.go.id", "kemennaker.go.id", "kemenkeu.go.id",
                "ojk.go.id", "bi.go.id", "bps.go.id", "djsn.go.id", "ilo.org",
                "worldbank.org", "databoks.katadata.co.id", "katadata.co.id",
                "bisnis.com", "cnbcindonesia.com", "kompas.com", "tempo.co",
                "antaranews.com", "detik.com", "thejakartapost.com", "kontan.co.id",
            ],
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://api.tavily.com/search", data=body,
            headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
            method="POST")
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode())
        parts = []
        if data.get("answer"):
            parts.append(f"RINGKASAN_TAVILY: {data['answer'][:600]}")
        for i, res in enumerate(data.get("results", [])[:max_results], 1):
            title    = res.get("title", "Tanpa judul")
            snippet  = res.get("content", "")[:400]
            url_full = res.get("url", "")
            score    = res.get("score", 0)
            if snippet or url_full:
                parts.append(f"SUMBER_{i}:\n  Judul: {title}\n  URL: {url_full}\n  Relevansi: {score:.2f}\n  Isi: {snippet}")
        return "\n\n".join(parts)
    except Exception as e:
        return f"TAVILY_ERROR: {str(e)}"


def _web_search_ddg(query: str, max_results: int = 3) -> str:
    import urllib.request, urllib.parse, json, re as _re
    try:
        q = urllib.parse.quote(query + " Indonesia BPJS")
        url = f"https://api.duckduckgo.com/?q={q}&format=json&no_html=1&skip_disambig=1"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as r:
            data = json.loads(r.read().decode())
        snippets = []
        if data.get("AbstractText"):
            snippets.append(f"SUMBER_DDG_1:\n  Isi: {data['AbstractText'][:400]}\n  URL: {data.get('AbstractURL','')}")
        for i, topic in enumerate(data.get("RelatedTopics", [])[:max_results], 2):
            if isinstance(topic, dict) and topic.get("Text"):
                snippets.append(f"SUMBER_DDG_{i}:\n  Isi: {topic['Text'][:200]}\n  URL: {topic.get('FirstURL','')}")
        if snippets:
            return "\n\n".join(snippets)
        q2   = urllib.parse.quote(query + " BPJS ketenagakerjaan Indonesia")
        url2 = f"https://html.duckduckgo.com/html/?q={q2}"
        req2 = urllib.request.Request(url2, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req2, timeout=8) as r2:
            html = r2.read().decode("utf-8", errors="ignore")
        matches = _re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', html, _re.DOTALL)
        clean = [_re.sub(r"<[^>]+>", "", m).strip() for m in matches[:max_results]]
        return "\n".join(f"• {c[:250]}" for c in clean if len(c) > 30)
    except Exception:
        return ""


def _web_search(query: str, max_results: int = 4) -> str:
    result = _web_search_tavily(query, max_results)
    if result:
        return result
    return _web_search_ddg(query, max_results)


# ══════════════════════════════════════════════════════════════════════════════
# CHAT DATA CONTEXT BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def _build_chat_data_ctx(df, active_progs, years, has_nom, latest_year, prev_year) -> str:
    if df is None:
        return "Belum ada data yang diupload."
    rows = ["=" * 60, f"DATA KLAIM BPJS KETENAGAKERJAAN — {latest_year}", "=" * 60]
    rows.append(f"Program aktif: {', '.join(active_progs)}")
    rows.append(f"Periode data: {min(years)}–{max(years)} ({len(years)} tahun)")
    rows.append("")
    for prog in active_progs:
        try:
            prog_df = df[df['Kategori'] == prog].sort_values('Tahun')
            first, last = prog_df.iloc[0], prog_df.iloc[-1]
            kasus_growth = (last['Kasus'] - first['Kasus']) / (first['Kasus'] + 1e-9) * 100
            line = (f"{prog}: {int(first['Kasus']):,}→{int(last['Kasus']):,} kasus "
                    f"[{kasus_growth:+.1f}%] ({int(first['Tahun'])}→{int(last['Tahun'])})")
            if has_nom and 'Nominal' in prog_df.columns:
                nom_growth = (last['Nominal'] - first['Nominal']) / (first['Nominal'] + 1e-9) * 100
                line += (f" | nominal Rp{first['Nominal']/1e9:.1f}M "
                         f"→ Rp{last['Nominal']/1e9:.1f}M [{nom_growth:+.1f}%]")
            rows.append(line)
        except Exception:
            pass
    rows.append("")
    rows.append("PENTING: 'Kasus' = jumlah kejadian/klaim. 'Nominal' = nilai uang (Rp). Keduanya BERBEDA.")
    rows.append("=" * 60)
    return "\n".join(rows)


def _build_chat_wb_ctx(wb_ekon: dict) -> str:
    try:
        if not wb_ekon or not any(wb_ekon.values()):
            return ""
        rows = ["\nData Makroekonomi Indonesia (World Bank):"]
        for yr in sorted(wb_ekon.keys()):
            d = wb_ekon[yr]
            if not d: continue
            parts = []
            if 'gdp_pct'          in d: parts.append(f"PDB {d['gdp_pct']:+.2f}%")
            if 'inflation_pct'    in d: parts.append(f"Inflasi {d['inflation_pct']:.1f}%")
            if 'unemployment_pct' in d: parts.append(f"Pengangguran {d['unemployment_pct']:.1f}%")
            if parts: rows.append(f"  {yr}: {', '.join(parts)}")
        return "\n".join(rows)
    except Exception:
        return ""


# ══════════════════════════════════════════════════════════════════════════════
# CHAT SESSION MANAGER
# ══════════════════════════════════════════════════════════════════════════════

_CHAT_SESSIONS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".bpjs_history", "chat_sessions.json")


def _sessions_load() -> dict:
    try:
        os.makedirs(os.path.dirname(_CHAT_SESSIONS_FILE), exist_ok=True)
        if os.path.exists(_CHAT_SESSIONS_FILE):
            with open(_CHAT_SESSIONS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _sessions_save(sessions: dict):
    try:
        os.makedirs(os.path.dirname(_CHAT_SESSIONS_FILE), exist_ok=True)
        if len(sessions) > 50:
            sorted_keys = sorted(sessions.keys(), key=lambda k: sessions[k].get("ts", ""), reverse=True)
            sessions = {k: sessions[k] for k in sorted_keys[:50]}
        with open(_CHAT_SESSIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(sessions, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _session_generate_title(first_question: str) -> str:
    clean = re.sub(r"[^\w\s\-?.,]", "", first_question).strip()
    if len(clean) > 45:
        clean = clean[:42] + "..."
    return clean if clean else "Percakapan Baru"


def _session_new() -> str:
    sid = f"sess_{int(time.time()*1000)}"
    st.session_state.chat_sessions = _sessions_load()
    st.session_state.chat_sessions[sid] = {
        "title": "Percakapan Baru",
        "messages": [],
        "ts": time.strftime("%Y-%m-%d %H:%M"),
        "ts_raw": time.time(),
    }
    _sessions_save(st.session_state.chat_sessions)
    return sid


def _session_save_current():
    sid = st.session_state.active_session_id
    if not sid:
        return
    sessions = _sessions_load()
    if sid not in sessions:
        sessions[sid] = {"title": "Percakapan Baru", "messages": [], "ts": "", "ts_raw": 0}
    sessions[sid]["messages"] = list(st.session_state.chat_history)
    sessions[sid]["ts"]       = time.strftime("%Y-%m-%d %H:%M")
    first_user = next((m["content"] for m in st.session_state.chat_history if m["role"] == "user"), None)
    if first_user and sessions[sid]["title"] == "Percakapan Baru":
        sessions[sid]["title"] = _session_generate_title(first_user)
    st.session_state.chat_sessions = sessions
    _sessions_save(sessions)


def _session_load(sid: str):
    sessions = _sessions_load()
    if sid in sessions:
        st.session_state.chat_history      = list(sessions[sid].get("messages", []))
        st.session_state.active_session_id = sid
        st.session_state.chat_sessions     = sessions


def _session_delete(sid: str):
    sessions = _sessions_load()
    if sid in sessions:
        del sessions[sid]
    _sessions_save(sessions)
    st.session_state.chat_sessions = sessions
    if st.session_state.active_session_id == sid:
        st.session_state.active_session_id = None
        st.session_state.chat_history = []


# ══════════════════════════════════════════════════════════════════════════════
# QA MEMORY
# ══════════════════════════════════════════════════════════════════════════════

_QA_MEMORY_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".bpjs_history", "qa_memory.json")


def _qa_memory_load() -> list:
    try:
        os.makedirs(os.path.dirname(_QA_MEMORY_FILE), exist_ok=True)
        if os.path.exists(_QA_MEMORY_FILE):
            with open(_QA_MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return []


def _qa_memory_save(question: str, answer: str):
    try:
        os.makedirs(os.path.dirname(_QA_MEMORY_FILE), exist_ok=True)
        mem = _qa_memory_load()
        q_low = question.lower()
        if any(w in q_low for w in ["jht", "hari tua"]):            topic = "JHT"
        elif any(w in q_low for w in ["jkk", "kecelakaan"]):        topic = "JKK"
        elif any(w in q_low for w in ["jkm", "kematian"]):          topic = "JKM"
        elif any(w in q_low for w in ["jkp", "kehilangan", "phk"]): topic = "JKP"
        elif any(w in q_low for w in ["jpn", "pensiun"]):           topic = "JPN"
        elif any(w in q_low for w in ["tren", "trend"]):             topic = "TREND"
        elif any(w in q_low for w in ["prediksi", "forecast"]):      topic = "PREDIKSI"
        elif any(w in q_low for w in ["kenapa", "mengapa", "faktor"]): topic = "ANALISIS"
        else:                                                          topic = "UMUM"
        q_hash = hashlib.md5(question.strip().lower().encode()).hexdigest()
        mem = [m for m in mem if m.get("q_hash") != q_hash]
        mem.append({"q": question.strip(), "a": answer.strip()[:1500],
                    "ts": time.strftime("%Y-%m-%d %H:%M"), "topic": topic, "q_hash": q_hash})
        if len(mem) > 500:
            mem = mem[-500:]
        with open(_QA_MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(mem, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _qa_memory_get_relevant(question: str, n: int = 5) -> str:
    try:
        mem = _qa_memory_load()
        if not mem:
            return ""
        q_words = set(question.lower().split())
        scored = []
        for entry in mem:
            overlap = len(q_words & set(entry["q"].lower().split()))
            q_low = question.lower(); topic = entry.get("topic", "")
            if topic == "JHT" and "jht" in q_low:       overlap += 3
            elif topic == "JKK" and "jkk" in q_low:     overlap += 3
            elif topic == "JKM" and "jkm" in q_low:     overlap += 3
            elif topic == "JKP" and "jkp" in q_low:     overlap += 3
            elif topic == "JPN" and "jpn" in q_low:     overlap += 3
            elif topic == "ANALISIS" and any(w in q_low for w in ["kenapa","mengapa","faktor"]): overlap += 2
            elif topic == "PREDIKSI" and any(w in q_low for w in ["prediksi","forecast"]): overlap += 2
            if overlap > 0:
                scored.append((overlap, entry))
        scored.sort(key=lambda x: -x[0])
        top = scored[:n]
        if not top:
            return ""
        lines = ["[REFERENSI PERCAKAPAN SEBELUMNYA]"]
        for _, entry in top:
            lines.append(f"Pertanyaan: {entry['q']}")
            lines.append(f"Jawaban: {entry['a'][:600]}")
            lines.append("---")
        return "\n".join(lines)
    except Exception:
        return ""


def _qa_memory_stats() -> dict:
    try:
        mem = _qa_memory_load()
        if not mem:
            return {"total": 0, "topics": {}}
        topics = {}
        for m in mem:
            t = m.get("topic", "UMUM")
            topics[t] = topics.get(t, 0) + 1
        return {"total": len(mem), "topics": topics, "latest": mem[-1]["ts"] if mem else "-"}
    except Exception:
        return {"total": 0, "topics": {}}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CHAT ANSWER
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """Kamu adalah AI Analyst ahli BPJS Ketenagakerjaan Indonesia dengan keahlian aktuaria dan ekonomi ketenagakerjaan.

KONTEKS DATASET:
- Data yang diberikan adalah DATA KLAIM BPJS Ketenagakerjaan.
- Kolom "Kasus" = jumlah kejadian klaim. Kolom "Nominal" = nilai manfaat yang dibayarkan dalam Rp.

PROGRAM BPJS KETENAGAKERJAAN:
- JHT: tabungan hari tua, diklaim saat resign/pensiun/cacat/meninggal
- JKK: klaim saat kecelakaan kerja/penyakit akibat kerja
- JKM: santunan meninggal dunia bukan akibat kecelakaan kerja
- JKP: klaim saat PHK, diluncurkan 2022
- JPN: manfaat pensiun bulanan seumur hidup

ATURAN ANALISIS:
1. Analisis HANYA berdasarkan data klaim yang diberikan
2. Jawab spesifik dengan angka kasus DAN nominal dari dataset
3. Sebutkan regulasi spesifik jika relevan: PP 82/2019, PP 45/2015, UU Cipta Kerja 2020
4. Jika ada hasil web search, integrasikan sebagai bukti pendukung
5. Jawab TEKNIS dan SPESIFIK: sebutkan persentase, tahun, angka absolut
6. Maksimal 10 kalimat kecuali diminta lebih panjang
7. WAJIB: Selalu sebut URL lengkap dari sumber yang kamu gunakan. Format: "(Sumber: https://...)"
8. Kamu SUDAH MELAKUKAN WEB SEARCH — jangan bilang kamu tidak bisa search
9. Struktur jawaban: (a) Temuan dari data internal, (b) Konfirmasi dari riset web + URL, (c) Kesimpulan
"""


def _chat_answer(question: str, df, active_progs, years, has_nom, latest_year, prev_year) -> str:
    api_info = _get_api_key()
    provider, key = api_info if isinstance(api_info, tuple) else ("", "")

    if not key:
        return (
            "⚠️ **API Key belum diset.** Pilih salah satu (gratis):\n\n"
            "**🟢 Groq (Rekomendasi):**\n"
            "1. Daftar di https://console.groq.com\n"
            "2. API Keys → Create API Key (dimulai `gsk_...`)\n"
            "3. Tambahkan ke `.streamlit/secrets.toml`: `GROQ_API_KEY = \"gsk_xxxxx\"`\n\n"
            "**🟡 Gemini (alternatif):**\n"
            "1. https://aistudio.google.com → Get API Key\n"
            "2. `GEMINI_API_KEY = \"AIza...\"`"
        )

    detected = _detect_provider(key)
    if detected:
        provider = detected

    # ── Cache check ───────────────────────────────────────────────────────────
    hist_str = ""
    for h in st.session_state.chat_history[-6:]:
        role = "User" if h["role"] == "user" else "AI"
        hist_str += f"{role}: {h['content']}\n"

    # Cache key menyertakan apakah forecast sudah ada, agar tidak
    # serve cached jawaban lama yang tidak punya angka prediksi
    _has_forecast = str(st.session_state.get('forecast_Kasus') is not None)
    _cache_id = hashlib.md5(
        f"{question}|{hist_str[-200:]}|{_has_forecast}".encode()
    ).hexdigest()

    if '_ai_resp_cache' not in st.session_state:
        st.session_state._ai_resp_cache = {}
    if _cache_id in st.session_state._ai_resp_cache:
        return st.session_state._ai_resp_cache[_cache_id]

    # ── Web search ────────────────────────────────────────────────────────────
    _search_ctx = ""
    _skip_keywords = ["halo", "hai", "terima kasih", "makasih", "oke", "ok",
                      "siapa kamu", "kamu apa", "test", "coba", "hitung saja",
                      "berapa total", "tampilkan data"]
    _is_trivial = (len(question.strip()) < 15 or
                   any(question.lower().strip().startswith(w) for w in _skip_keywords))

    if not _is_trivial:
        _q_low = question.lower()
        _prog_hint = ""
        if "jht" in _q_low:   _prog_hint = "JHT jaminan hari tua"
        elif "jkk" in _q_low: _prog_hint = "JKK jaminan kecelakaan kerja"
        elif "jkm" in _q_low: _prog_hint = "JKM jaminan kematian"
        elif "jkp" in _q_low: _prog_hint = "JKP jaminan kehilangan pekerjaan"
        elif "jpn" in _q_low: _prog_hint = "JPN jaminan pensiun"
        _search_query = f"{question} {_prog_hint} BPJS Ketenagakerjaan Indonesia".strip()
        with st.spinner("🔍 Mencari referensi dari sumber terpercaya..."):
            _search_ctx = _web_search(_search_query, max_results=5)
            if _search_ctx:
                st.session_state['_last_web_search_query']  = _search_query
                st.session_state['_last_web_search_result'] = _search_ctx

    # ── Build context ─────────────────────────────────────────────────────────
    wb_ctx    = _build_chat_wb_ctx(st.session_state.get('_wb_ekon_cache', {}))
    mem_ctx   = _qa_memory_get_relevant(question, n=4)
    ml_result = st.session_state.get('ml_result', None)

    # ── RAG prompt (rag_context.py sekarang otomatis ambil forecast
    #    dari st.session_state['forecast_Kasus'] / ['forecast_Nominal']) ───────
    prompt = build_rag_prompt(
        question         = question,
        df               = df,
        active_progs     = active_progs,
        years            = years,
        has_nom          = has_nom,
        latest_year      = latest_year,
        prev_year        = prev_year,
        wb_ctx           = wb_ctx,
        search_ctx       = _search_ctx,
        mem_ctx          = mem_ctx,
        chat_history_str = hist_str,
        ml_result        = ml_result,
    )

    system = RAG_SYSTEM_PROMPT

    # ── Call AI ───────────────────────────────────────────────────────────────
    try:
        result = None
        if provider == "groq":
            result = _call_ai_groq(prompt, system, key, max_tokens=1100)
        elif provider == "gemini":
            result = _call_ai_gemini(prompt, system, key, max_tokens=1100)
        elif provider == "anthropic":
            import urllib.request, json as _j
            body = _j.dumps({
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 1100,
                "system": system,
                "messages": [{"role": "user", "content": prompt}]
            }).encode("utf-8")
            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages", data=body,
                headers={"Content-Type": "application/json", "x-api-key": key,
                         "anthropic-version": "2023-06-01"}, method="POST")
            with urllib.request.urlopen(req, timeout=30) as r:
                resp = _j.loads(r.read().decode())
            result = resp["content"][0]["text"].strip()
        else:
            return f"⚠️ Provider '{provider}' tidak dikenali."

        if result:
            if len(st.session_state._ai_resp_cache) >= 50:
                oldest = next(iter(st.session_state._ai_resp_cache))
                del st.session_state._ai_resp_cache[oldest]
            st.session_state._ai_resp_cache[_cache_id] = result
            _qa_memory_save(question, result)
        return result

    except RuntimeError as e:
        err = str(e)
        if "HTTP 401" in err or "HTTP 403" in err or "invalid_api_key" in err.lower():
            if provider == "groq" and "HTTP 403" in err:
                gemini_key = _secret("GEMINI_API_KEY", "")
                if gemini_key:
                    try:
                        return _call_ai_gemini(prompt, system, gemini_key, max_tokens=1100)
                    except Exception:
                        pass
            return f"❌ **API Key ditolak.** Provider: **{provider}**\nDetail: `{err[:200]}`"
        elif "429" in err or "Semua model" in err:
            return "⏳ **Rate limit.** Tunggu 30–60 detik lalu coba lagi."
        elif "timeout" in err.lower():
            return "⏱️ **Request timeout.** Coba lagi dalam beberapa detik."
        else:
            return f"⚠️ **Error:** `{err[:600]}`"
    except Exception as e:
        return f"⚠️ **Error tidak terduga:** `{str(e)[:300]}`"