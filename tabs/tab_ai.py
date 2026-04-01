"""tabs/tab_ai.py — Tab 5: AI Analyst BPJS (clean, tanpa history/new chat)."""
import streamlit as st
from ai_chart_utils import detect_chart_intent, build_chart_data, render_ai_chart, ChartIntent


def render_tab_ai(df, active_progs, years, latest_year, prev_year, has_nom,
                  COLORS, DARK,
                  _get_api_key, _session_new, _session_save_current, _session_load,
                  _session_delete, _sessions_load, _chat_answer, _qa_memory_stats,
                  _fetch_worldbank):

    # ── Init session state ────────────────────────────────────────────────────
    for k, default in [
        ('chat_history', []),
        ('_chat_pending', None),
        ('_last_web_search_result', ''),
        ('_last_web_search_query', ''),
        ('ai_chart_history', []),   # list chart, tidak dihapus saat pertanyaan baru
    ]:
        if k not in st.session_state:
            st.session_state[k] = default

    # ── Process pending question ──────────────────────────────────────────────
    if st.session_state._chat_pending:
        _q = st.session_state._chat_pending
        st.session_state._chat_pending = None
        _api_provider = _get_api_key()[0] or "AI"
        _spinner_lbl = "🤖 Gemini AI menganalisis..." if _api_provider == "gemini" else "🤖 AI menganalisis..."
        with st.spinner(_spinner_lbl):
            _ans = _chat_answer(_q, df, active_progs, years, has_nom, latest_year, prev_year)

        # ── Deteksi intent chart — append ke history, tidak overwrite ──────────
        _intent = detect_chart_intent(_q)
        if _intent != ChartIntent.NO_CHART:
            _chart_data = build_chart_data(
                question     = _q,
                df           = df,
                intent       = _intent,
                active_progs = list(active_progs),
                years        = list(years),
                has_nom      = has_nom,
                latest_year  = latest_year,
                colors       = COLORS,
                dark         = DARK,
            )
            if _chart_data is not None:
                # Simpan referensi ke pesan chat yang bersangkutan
                _chart_data['chat_index'] = len(st.session_state.chat_history)
                if 'ai_chart_history' not in st.session_state:
                    st.session_state['ai_chart_history'] = []
                st.session_state['ai_chart_history'].append(_chart_data)

        st.session_state.chat_history.append({"role": "assistant", "content": _ans})
        st.rerun()

    # ── AI badge ──────────────────────────────────────────────────────────────
    try:
        _ai_prov, _ai_key = _get_api_key()
        if _ai_prov == "gemini" and _ai_key:   _ai_badge = "🟢 Gemini 2.0 Flash"
        elif _ai_prov == "groq" and _ai_key:   _ai_badge = "🟡 Groq LLaMA 3.3 70B"
        else:                                   _ai_badge = "🔴 API Belum Diset"
    except Exception:
        _ai_badge = "🔴 API Belum Diset"

    # ── Header bar ────────────────────────────────────────────────────────────
    st.markdown(
        '<div style="background:linear-gradient(135deg,#1b5e20,#2e7d32,#43a047);'
        'border-radius:14px;padding:12px 20px;display:flex;align-items:center;'
        'justify-content:space-between;margin-bottom:14px;">'
        '<div style="display:flex;align-items:center;gap:12px;">'
        '<div style="width:38px;height:38px;background:rgba(255,255,255,.2);border-radius:50%;'
        'display:flex;align-items:center;justify-content:center;font-size:1.15rem;">🤖</div>'
        '<div>'
        '<div style="font-size:.95rem;font-weight:800;color:white;">AI Analyst BPJS</div>'
        '<div style="font-size:.68rem;color:#a5d6a7;display:flex;align-items:center;gap:5px;margin-top:2px;">'
        '<span style="width:7px;height:7px;background:#69f0ae;border-radius:50%;display:inline-block;"></span>'
        ' Online — tanya tentang data klaim, tren, prediksi, atau insight makroekonomi'
        '</div></div></div>'
        f'<div style="font-size:.72rem;font-weight:600;padding:5px 14px;border-radius:20px;'
        f'background:rgba(255,255,255,.15);color:white;border:1px solid rgba(255,255,255,.3);">'
        f'{_ai_badge}</div></div>',
        unsafe_allow_html=True)

    # ── Web search expander ───────────────────────────────────────────────────
    _last_search = st.session_state.get("_last_web_search_result", "")
    _last_query  = st.session_state.get("_last_web_search_query", "")
    if _last_search:
        with st.expander("🌐 Referensi web yang digunakan AI", expanded=False):
            st.caption(f"Query: **{_last_query}**")
            import re as _re
            for _src in _re.split(r'SUMBER_\d+:', _last_search):
                _src = _src.strip()
                if not _src:
                    continue
                _ls    = {k.strip(): v.strip() for k, v in
                          [l.split(':', 1) for l in _src.split('\n') if ':' in l]}
                _url   = _ls.get('URL', '')
                _judul = _ls.get('Judul', _ls.get('Isi', '')[:60])
                _isi   = _ls.get('Isi', _src[:200])
                if _url:
                    st.markdown(f"**[{_judul}]({_url})**\n\n{_isi[:300]}", unsafe_allow_html=True)
                    st.divider()

    # ── CSS chat bubble ───────────────────────────────────────────────────────
    st.markdown("""
    <style>
    .chat-area {
        padding: 16px 10px;
        background: #f8fafc;
        border-radius: 14px;
        border: 1px solid #e2e8f0;
        min-height: 400px;
        max-height: 500px;
        overflow-y: auto;
        margin-bottom: 10px;
    }
    .chat-area::-webkit-scrollbar { width: 4px; }
    .chat-area::-webkit-scrollbar-thumb { background: #c8e6c9; border-radius: 2px; }
    .msg-row { display: flex; margin: 10px 6px; align-items: flex-end; gap: 8px; }
    .msg-row.user { justify-content: flex-end; }
    .msg-row.bot  { justify-content: flex-start; }
    .bubble {
        max-width: 80%;
        font-size: .88rem;
        line-height: 1.75;
        padding: 11px 16px;
        word-break: break-word;
    }
    .user-bubble {
        background: linear-gradient(135deg, #1b5e20, #2e7d32);
        color: white;
        border-radius: 18px 18px 4px 18px;
        box-shadow: 0 2px 10px rgba(27,94,32,.22);
    }
    .bot-bubble {
        background: white;
        color: #1a202c;
        border: 1px solid #e8f5e9;
        border-radius: 4px 18px 18px 18px;
        box-shadow: 0 1px 5px rgba(0,0,0,.06);
    }
    .bot-avatar {
        width: 30px; height: 30px;
        background: linear-gradient(135deg, #1b5e20, #43a047);
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: .7rem; font-weight: 800; color: white; flex-shrink: 0;
    }
    </style>""", unsafe_allow_html=True)

    # ── Build chat HTML ───────────────────────────────────────────────────────
    _msgs_html_parts = []
    for _msg in st.session_state.chat_history[-40:]:
        _c = (_msg["content"]
              .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
              .replace("\n", "<br>"))
        if _msg["role"] == "user":
            _msgs_html_parts.append(
                '<div class="msg-row user">'
                '<div class="bubble user-bubble">' + _c + '</div>'
                '</div>')
        else:
            _msgs_html_parts.append(
                '<div class="msg-row bot">'
                '<div class="bot-avatar">AI</div>'
                '<div class="bubble bot-bubble">' + _c + '</div>'
                '</div>')
    _msgs_html = "\n".join(_msgs_html_parts)

    # ── Chat area ─────────────────────────────────────────────────────────────
    if not st.session_state.chat_history:
        st.markdown(
            '<div class="chat-area">'
            '<div style="text-align:center;padding:60px 20px 24px;">'
            '<div style="font-size:3rem;margin-bottom:14px;">🤖</div>'
            '<div style="font-size:1.15rem;font-weight:800;color:#1b5e20;margin-bottom:8px;">'
            'Halo! Saya AI Analyst BPJS.</div>'
            '<div style="font-size:.88rem;color:#388e3c;line-height:1.9;">'
            'Tanya saya tentang data klaim, tren, analisis program,<br>'
            'perbandingan antar tahun, atau insight makroekonomi.</div>'
            f'<div style="display:flex;flex-wrap:wrap;gap:8px;justify-content:center;margin-top:22px;">'
            f'<span style="background:white;border:1px solid #c8e6c9;border-radius:20px;'
            f'padding:7px 14px;font-size:.78rem;color:#2e7d32;font-weight:500;">📊 Program terbesar {latest_year}?</span>'
            f'<span style="background:white;border:1px solid #c8e6c9;border-radius:20px;'
            f'padding:7px 14px;font-size:.78rem;color:#2e7d32;font-weight:500;">📈 Tren klaim 5 tahun</span>'
            f'<span style="background:white;border:1px solid #c8e6c9;border-radius:20px;'
            f'padding:7px 14px;font-size:.78rem;color:#2e7d32;font-weight:500;">🦠 Dampak COVID</span>'
            f'<span style="background:white;border:1px solid #c8e6c9;border-radius:20px;'
            f'padding:7px 14px;font-size:.78rem;color:#2e7d32;font-weight:500;">🔮 Prediksi tahun depan</span>'
            f'</div></div></div>',
            unsafe_allow_html=True)
    else:
        st.markdown('<div class="chat-area">' + _msgs_html + '</div>', unsafe_allow_html=True)

    # ── Chart AI — render per jawaban, tidak ada yang hilang ────────────────
    _chart_history = st.session_state.get('ai_chart_history', [])
    if _chart_history and st.session_state.chat_history:
        # Grouping chart per chat_index agar tiap pertanyaan punya chart sendiri
        for _cd in _chart_history:
            render_ai_chart(_cd)

    # ── Quick chips (muncul setelah ada chat) ─────────────────────────────────
    if st.session_state.chat_history:
        _quick_list = [
            f"📊 Program terbesar {latest_year}?",
            "📈 Tren klaim 5 tahun terakhir?",
            "🦠 Dampak COVID ke klaim?",
            "⚠️ Program paling volatil?",
            "📉 Kenapa ada penurunan klaim?",
            "🔮 Prediksi klaim tahun depan?",
        ]
        _qcols = st.columns(3)
        for _qi, _qt in enumerate(_quick_list):
            with _qcols[_qi % 3]:
                if st.button(_qt, key=f"quick_{_qi}", use_container_width=True):
                    _cq = _qt.split(" ", 1)[1] if " " in _qt else _qt
                    st.session_state.chat_history.append({"role": "user", "content": _cq})
                    st.session_state._chat_pending = _cq
                    st.rerun()

    # ── Input area ────────────────────────────────────────────────────────────
    _ic, _bc = st.columns([9, 1])
    with _ic:
        _user_input = st.text_input(
            "msg", key="ai_tab_input",
            placeholder="Ketik pertanyaan Anda tentang data BPJS...",
            label_visibility="collapsed")
    with _bc:
        _send_btn = st.button("➤", key="ai_tab_send", type="primary", use_container_width=True)

    if _send_btn and _user_input and _user_input.strip():
        st.session_state.chat_history.append({"role": "user", "content": _user_input.strip()})
        st.session_state._chat_pending = _user_input.strip()
        st.rerun()

    # ── Tombol clear chat (kecil, di bawah) ──────────────────────────────────
    if st.session_state.chat_history:
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        col_clear, _ = st.columns([1, 5])
        with col_clear:
            if st.button("🗑 Hapus chat", key="clear_chat_btn", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state["_last_web_search_result"] = ""
                st.session_state["ai_chart_history"] = []
                st.rerun()