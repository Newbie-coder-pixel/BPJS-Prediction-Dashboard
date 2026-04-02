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


# ── Domain knowledge per program — logika kausal yang benar ──────────────────
_PROGRAM_DOMAIN = {
    "JHT": {
        "nama": "Jaminan Hari Tua",
        "mekanisme": (
            "JHT hanya bisa diklaim saat: berhenti bekerja/PHK (1 bulan setelah berhenti), "
            "pensiun, cacat tetap, atau meninggal. PP 2/2022 menambahkan: bisa cair parsial "
            "10% untuk persiapan pensiun dan 20% untuk KPR meski masih aktif bekerja. "
            "LOGIKA KAUSAL WAJIB: PHK naik → klaim JHT NAIK. "
            "Pengangguran naik = lebih banyak yang di-PHK = klaim JHT NAIK, bukan turun. "
            "Relaksasi aturan pencairan → klaim JHT NAIK meski peserta masih bekerja."
        ),
        "driver_naik": [
            "Gelombang PHK massal (krisis, otomasi, resesi pasca-pandemi)",
            "Relaksasi aturan pencairan via PP 2/2022 (cair parsial tanpa harus berhenti kerja)",
            "Banyak pekerja resign untuk pindah karir atau wirausaha",
            "Peserta mencapai usia pensiun 56 tahun dalam jumlah besar",
        ],
        "driver_turun": [
            "Pasar kerja stabil — PHK rendah, turnover karyawan rendah",
            "Pandemi awal 2020: pekerja takut keluar meski mau resign",
            "Aturan pencairan masih ketat (sebelum PP 2/2022)",
        ],
        "korelasi": {
            "Pengangguran NAIK": "JHT NAIK — pengangguran meningkat berarti lebih banyak yang di-PHK dan mengklaim JHT",
            "PDB TURUN (resesi)": "JHT NAIK — resesi memicu PHK massal yang langsung dorong klaim JHT",
            "PDB NAIK (stabil)": "JHT cenderung TURUN — pasar kerja membaik, PHK berkurang",
        }
    },
    "JKK": {
        "nama": "Jaminan Kecelakaan Kerja",
        "mekanisme": (
            "JKK HANYA bisa diklaim oleh peserta yang SEDANG AKTIF BEKERJA saat kecelakaan. "
            "Pengangguran TIDAK BISA klaim JKK sama sekali — ini aturan fundamental. "
            "LOGIKA KAUSAL WAJIB: pengangguran naik = pekerja aktif berkurang = "
            "potensi kecelakaan berkurang = klaim JKK TURUN. "
            "Klaim JKK berkorelasi langsung dengan aktivitas industri berisiko tinggi: "
            "konstruksi, manufaktur, pertambangan, transportasi."
        ),
        "driver_naik": [
            "Ekspansi sektor konstruksi, manufaktur, pertambangan (industri padat risiko)",
            "Boom infrastruktur pemerintah — proyek tol, bandara, gedung meningkat",
            "Normalisasi aktivitas industri pasca-pandemi (pabrik dan proyek kembali berjalan)",
            "Pertumbuhan jumlah peserta aktif terdaftar BPJS Ketenagakerjaan",
            "Peningkatan kesadaran pelaporan kecelakaan kerja",
        ],
        "driver_turun": [
            "Pandemi COVID-19: pabrik tutup, proyek konstruksi berhenti, pekerja WFH",
            "PSBB/lockdown: aktivitas kerja fisik di lapangan berhenti total",
            "Resesi: industri melambat, banyak pekerja dirumahkan",
            "Pengangguran tinggi: lebih sedikit pekerja aktif = lebih sedikit kecelakaan",
        ],
        "korelasi": {
            "Pengangguran NAIK": "JKK TURUN — lebih sedikit pekerja aktif berarti lebih sedikit yang bisa kecelakaan kerja",
            "PDB NAIK": "JKK NAIK — aktivitas industri intensif, lebih banyak pekerja di lapangan",
            "PDB TURUN (resesi/pandemi)": "JKK TURUN — aktivitas industri melambat drastis",
        }
    },
    "JKM": {
        "nama": "Jaminan Kematian",
        "mekanisme": (
            "JKM memberi santunan Rp42 juta + biaya pemakaman Rp10 juta kepada ahli waris "
            "peserta aktif yang meninggal bukan karena kecelakaan kerja. "
            "LOGIKA KAUSAL: pandemi COVID-19 secara langsung membunuh peserta aktif "
            "→ klaim JKM melonjak. Normalisasi pasca-pandemi → klaim kembali ke baseline. "
            "Pertumbuhan peserta aktif → pool risiko lebih besar → klaim perlahan naik."
        ),
        "driver_naik": [
            "Pandemi/wabah: kematian peserta aktif melonjak drastis di luar kecelakaan kerja",
            "Pertumbuhan jumlah peserta aktif (pool risiko lebih besar)",
            "Penyakit kardiovaskular, kanker, dan penyakit tidak menular di usia produktif",
        ],
        "driver_turun": [
            "Normalisasi pasca-pandemi: angka kematian kembali ke baseline alami",
            "Penurunan jumlah peserta aktif",
        ],
        "korelasi": {
            "Pandemi aktif": "JKM NAIK drastis karena kematian COVID-19 langsung mengenai peserta aktif",
            "Pasca pandemi": "JKM TURUN ke baseline — angka kematian kembali normal",
        }
    },
    "JKP": {
        "nama": "Jaminan Kehilangan Pekerjaan",
        "mekanisme": (
            "JKP diluncurkan Februari 2022 (PP 37/2021), memberi manfaat cash 45%→25%→15% upah "
            "selama max 6 bulan + pelatihan + akses loker. HANYA untuk yang di-PHK, bukan resign. "
            "LOGIKA KAUSAL: PHK massal → klaim JKP langsung naik. "
            "Di 2022 awal, klaim masih sangat rendah karena program baru dan sosialisasi belum merata. "
            "Pertumbuhan JKP mencerminkan: kenaikan PHK formal + meningkatnya kesadaran peserta."
        ),
        "driver_naik": [
            "Gelombang PHK massal — efisiensi perusahaan, otomasi, perlambatan ekonomi",
            "Meningkatnya kesadaran peserta tentang hak klaim JKP",
            "PHK di sektor manufaktur, tekstil, garmen yang padat karya",
            "Penegakan kepesertaan BPJS (lebih banyak pekerja terdaftar = lebih banyak yang bisa klaim)",
        ],
        "driver_turun": [
            "Program masih baru (2022): sosialisasi belum merata, banyak yang tidak tahu haknya",
            "PHK rendah di periode pasar kerja stabil",
        ],
        "korelasi": {
            "Pengangguran NAIK": "JKP NAIK — PHK formal meningkat langsung mendorong klaim JKP",
            "PDB TURUN": "JKP NAIK — perlambatan ekonomi mendorong PHK massal",
        }
    },
    "JPN": {
        "nama": "Jaminan Pensiun",
        "mekanisme": (
            "JPN memberi manfaat pensiun bulanan seumur hidup, syarat: masa iur minimal 15 tahun "
            "dan usia pensiun 56 tahun. Program baru mulai 2015. "
            "Klaim masif baru terjadi sekitar 2030 ketika peserta pertama (2015) genap 15 tahun masa iur. "
            "Pertumbuhan klaim saat ini dari: peserta awal yang cukup masa iur, "
            "dan klaim cacat/janda/duda."
        ),
        "driver_naik": [
            "Peserta awal 2015 mulai memenuhi syarat 15 tahun masa iur",
            "Klaim manfaat cacat total tetap dari peserta aktif",
            "Klaim janda/duda dari peserta yang meninggal sebelum pensiun",
        ],
        "driver_turun": [
            "Masih fase akumulasi — mayoritas peserta belum memenuhi 15 tahun masa iur",
        ],
        "korelasi": {
            "Catatan": "JPN tidak sensitif terhadap fluktuasi makro jangka pendek. Driver utama: demografi peserta dan masa iur."
        }
    },
}

_DEFAULT_DOMAIN = {
    "nama": "Program BPJS Ketenagakerjaan",
    "mekanisme": "Program jaminan sosial ketenagakerjaan Indonesia.",
    "driver_naik": ["Pertumbuhan peserta aktif", "Perubahan regulasi yang meringankan syarat klaim"],
    "driver_turun": ["Penurunan peserta aktif", "Aturan klaim diperketat"],
    "korelasi": {}
}


def _get_prog_domain(prog_name: str) -> dict:
    pn = prog_name.upper().strip()
    for key, domain in _PROGRAM_DOMAIN.items():
        if key in pn or pn.startswith(key):
            return domain
    return _DEFAULT_DOMAIN


def _ai_analyze_peak_trough(prog_name, peak_yr, peak_val, trough_yr, trough_val, wb_data, api_info):
    import json as _j

    domain    = _get_prog_domain(prog_name)
    nama      = domain["nama"]
    mekanisme = domain["mekanisme"]
    drv_naik  = "\n".join(f"  - {d}" for d in domain.get("driver_naik", []))
    drv_turun = "\n".join(f"  - {d}" for d in domain.get("driver_turun", []))
    korelasi  = "\n".join(f"  - {k}: {v}" for k, v in domain.get("korelasi", {}).items())

    # Data makro lengkap per tahun
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

    # Konteks ekonomi spesifik tahun peak & trough
    def _fmt_ekon(yr):
        d = wb_data.get(yr, {})
        parts = []
        if 'gdp_pct'          in d: parts.append(f"PDB {d['gdp_pct']:+.1f}%")
        if 'unemployment_pct' in d: parts.append(f"pengangguran {d['unemployment_pct']:.1f}%")
        if 'inflation_pct'    in d: parts.append(f"inflasi {d['inflation_pct']:.1f}%")
        return ", ".join(parts) if parts else "data terbatas"

    prompt = f"""Kamu adalah Senior Actuary BPJS Ketenagakerjaan Indonesia dengan 15 tahun pengalaman.

PROGRAM YANG DIANALISIS: {prog_name} ({nama})

MEKANISME KLAIM PROGRAM INI — WAJIB DIPAHAMI DAN DIPATUHI:
{mekanisme}

FAKTOR YANG MENAIKKAN KLAIM {prog_name}:
{drv_naik}

FAKTOR YANG MENURUNKAN KLAIM {prog_name}:
{drv_turun}

KORELASI KAUSAL DENGAN MAKROEKONOMI — GUNAKAN LOGIKA INI:
{korelasi if korelasi else "Gunakan logika sesuai mekanisme program di atas."}

DATA KLAIM:
- PEAK  : tahun {peak_yr} = {peak_val:,} kasus | Ekonomi {peak_yr}: {_fmt_ekon(peak_yr)}
- TROUGH: tahun {trough_yr} = {trough_val:,} kasus | Ekonomi {trough_yr}: {_fmt_ekon(trough_yr)}

TREN MAKROEKONOMI INDONESIA (World Bank):
{ekon_str}

INSTRUKSI WAJIB:
1. Reasoning HARUS konsisten dengan mekanisme program — contoh untuk JKK: jika pengangguran {trough_yr} tinggi, jelaskan bahwa pengangguran tinggi = pekerja aktif berkurang = kecelakaan kerja berkurang = JKK turun (BUKAN pengangguran tinggi karena pekerja takut klaim)
2. Gunakan angka ekonomi aktual dari data di atas dalam penjelasan
3. DILARANG KERAS menggunakan: "mungkin", "kemungkinan", "diperkirakan", "sepertinya", "tampaknya", "bisa jadi", "diduga"
4. Gunakan pernyataan kausal langsung: "karena", "sehingga", "akibat", "mendorong", "mengakibatkan"
5. Bahasa Indonesia formal, 2-3 kalimat per kondisi

Jawab HANYA JSON ini tanpa teks lain:
{{"peak_label":"emoji+3kata","peak_desc":"2-3 kalimat kausal dengan angka ekonomi","trough_label":"emoji+3kata","trough_desc":"2-3 kalimat kausal dengan angka ekonomi"}}"""

    try:
        raw = _call_ai(prompt, api_info=api_info, max_tokens=700)
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

# SYSTEM_PROMPT di sini tidak dipakai — yang aktif adalah RAG_SYSTEM_PROMPT
# dari rag_context.py yang sudah jauh lebih lengkap dan konsisten.
# Variabel ini dipertahankan hanya untuk backward compatibility jika ada import.
SYSTEM_PROMPT = ""  # deprecated — gunakan RAG_SYSTEM_PROMPT dari rag_context.py


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
        # Skip entry chart — tidak punya key 'content', hanya punya 'chart_data'
        if h.get("role") == "chart" or "content" not in h:
            continue
        role = "User" if h["role"] == "user" else "AI"
        hist_str += f"{role}: {h['content']}\n"

    # Cache key: sertakan fingerprint ML + forecast agar otomatis invalid
    # setelah user jalankan model baru di tab ML / Prediksi
    _has_forecast = str(st.session_state.get('forecast_Kasus') is not None)
    _ml_res       = st.session_state.get('ml_result')
    _ml_sig       = str(id(_ml_res)) if _ml_res is not None else "none"
    _fc            = st.session_state.get('forecast_Kasus')
    _fc_sig        = "fc_none"
    if isinstance(_fc, pd.DataFrame) and not _fc.empty:
        try:    _fc_sig = str(_fc.iloc[:3].values.tolist())
        except: _fc_sig = "fc_exists"
    _cache_id = hashlib.md5(
        f"{question}|{hist_str[-200:]}|{_has_forecast}|{_ml_sig}|{_fc_sig}".encode()
    ).hexdigest()

    if '_ai_resp_cache' not in st.session_state:
        st.session_state._ai_resp_cache = {}
    if _cache_id in st.session_state._ai_resp_cache:
        return st.session_state._ai_resp_cache[_cache_id]

    # ── Web search — hanya untuk pertanyaan yang BENAR butuh sumber eksternal ──
    # Pertanyaan data internal (kasus, nominal, sebab klaim, tren historis)
    # TIDAK perlu web search → hemat token, kurangi noise.
    # Web search hanya aktif untuk: regulasi baru, makroekonomi terkini, berita PHK, dll.
    _search_ctx = ""
    _q_low      = question.lower()

    # Pertanyaan yang jelas cukup dari data internal — skip web search
    _data_internal_kw = [
        "berapa kasus", "berapa nominal", "terbesar", "terbanyak", "tertinggi",
        "terendah", "tren klaim", "data klaim", "sebab klaim", "penyebab klaim",
        "program apa", "ranking", "bandingkan program", "jht", "jkk", "jkm", "jkp", "jpn",
        "prediksi", "forecast", "hitung", "berapa total", "jumlah kasus",
        "tampilkan", "tabel", "grafik", "chart", "distribusi",
    ]
    # Pertanyaan yang butuh konteks eksternal — aktifkan web search
    _external_kw = [
        "regulasi", "peraturan", "undang-undang", "pp ", "uu ", "kebijakan baru",
        "berita", "terkini", "2025", "2026", "phk massal", "pemutusan massal",
        "makroekonomi", "inflasi", "suku bunga", "resesi", "pdb indonesia",
        "omnibus", "cipta kerja", "reforma", "bpjs terbaru",
        "mengapa naik", "kenapa naik", "mengapa turun", "kenapa turun",
        "covid", "pandemi", "dampak ekonomi",
    ]

    _skip_keywords = ["halo", "hai", "terima kasih", "makasih", "oke", "ok",
                      "siapa kamu", "kamu apa", "test", "coba"]
    _is_trivial         = (len(question.strip()) < 15 or
                           any(_q_low.strip().startswith(w) for w in _skip_keywords))
    _needs_internal_only = any(k in _q_low for k in _data_internal_kw)
    _needs_external      = any(k in _q_low for k in _external_kw)

    # Search hanya jika: bukan trivial, bukan pure internal, dan ada sinyal external
    _do_search = (not _is_trivial) and (not _needs_internal_only or _needs_external)

    if _do_search:
        _prog_hint = ""
        if "jht" in _q_low:   _prog_hint = "JHT jaminan hari tua"
        elif "jkk" in _q_low: _prog_hint = "JKK jaminan kecelakaan kerja"
        elif "jkm" in _q_low: _prog_hint = "JKM jaminan kematian"
        elif "jkp" in _q_low: _prog_hint = "JKP jaminan kehilangan pekerjaan"
        elif "jpn" in _q_low: _prog_hint = "JPN jaminan pensiun"
        _search_query = f"{question} {_prog_hint} BPJS Ketenagakerjaan Indonesia".strip()
        with st.spinner("🔍 Mencari referensi dari sumber terpercaya..."):
            _search_ctx = _web_search(_search_query, max_results=4)
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

    # ── Trim prompt cerdas — pertahankan data internal, potong web search ──────
    # Groq: ~30k token context. Gemini Flash: 1M token. Kita set 28k char ≈ 7k token.
    # Strategi: potong [WEB SEARCH] dulu, baru [RIWAYAT CHAT], baru potong kasar.
    _MAX_PROMPT_CHARS = 28_000
    if len(prompt) > _MAX_PROMPT_CHARS:
        # Tahap 1: hapus blok web search jika ada (paling boros, paling tidak kritis)
        import re as _re
        prompt_no_web = _re.sub(
            r'\[WEB SEARCH.*?\[/WEB SEARCH\]', '[WEB SEARCH: dihapus karena prompt terlalu panjang]',
            prompt, flags=_re.DOTALL
        )
        if len(prompt_no_web) <= _MAX_PROMPT_CHARS:
            prompt = prompt_no_web
        else:
            # Tahap 2: hapus riwayat chat jika masih terlalu panjang
            prompt_no_hist = _re.sub(
                r'\[RIWAYAT CHAT\].*?(?=\nPERTANYAAN:)',
                '[RIWAYAT CHAT: dihapus]\n',
                prompt_no_web, flags=_re.DOTALL
            )
            if len(prompt_no_hist) <= _MAX_PROMPT_CHARS:
                prompt = prompt_no_hist
            else:
                # Tahap 3: potong kasar — tapi pertahankan 6k awal (metadata+data)
                # dan 4k akhir (instruksi+pertanyaan)
                _keep_start = 6_000
                _keep_end   = 4_000
                prompt = (
                    prompt_no_hist[:_keep_start]
                    + f"\n\n[... sebagian konteks diperpendek ({len(prompt):,} char → {_MAX_PROMPT_CHARS:,}) ...]\n\n"
                    + prompt_no_hist[-_keep_end:]
                )

    system = RAG_SYSTEM_PROMPT

    # ── Call AI ───────────────────────────────────────────────────────────────
    try:
        result = None
        if provider == "groq":
            result = _call_ai_groq(prompt, system, key, max_tokens=2000)
        elif provider == "gemini":
            result = _call_ai_gemini(prompt, system, key, max_tokens=2000)
        elif provider == "anthropic":
            import urllib.request, json as _j
            body = _j.dumps({
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 2000,
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
            # Retry sekali dengan delay jika rate limit — sebelum menyerah
            import time as _time
            _time.sleep(8)
            try:
                if provider == "groq":
                    result2 = _call_ai_groq(prompt, system, key, max_tokens=1200)
                elif provider == "gemini":
                    result2 = _call_ai_gemini(prompt, system, key, max_tokens=1200)
                else:
                    result2 = None
                if result2:
                    return result2
            except Exception:
                pass
            # Fallback ke Gemini jika Groq rate limit dan Gemini key tersedia
            if provider == "groq":
                _gem_key = _secret("GEMINI_API_KEY", "")
                if _gem_key:
                    try:
                        return _call_ai_gemini(prompt, system, _gem_key, max_tokens=1200)
                    except Exception:
                        pass
            return "⏳ **Rate limit.** API sedang sibuk — tunggu 30 detik lalu coba lagi.\n\nTips: Pertanyaan pendek menggunakan lebih sedikit token dan lebih jarang terkena limit."
        elif "timeout" in err.lower():
            return "⏱️ **Request timeout.** Coba lagi dalam beberapa detik."
        else:
            return f"⚠️ **Error:** `{err[:600]}`"
    except Exception as e:
        return f"⚠️ **Error tidak terduga:** `{str(e)[:300]}`"