"""tabs/tab_ai.py — Tab 5: AI Analyst BPJS."""
import streamlit as st


def render_tab_ai(df, active_progs, years, latest_year, prev_year, has_nom,
                  COLORS, DARK,
                  _get_api_key, _session_new, _session_save_current, _session_load,
                  _session_delete, _sessions_load, _chat_answer, _qa_memory_stats,
                  _fetch_worldbank):

    # ── Init session state ────────────────────────────────────────────────────
    for k, default in [
        ('chat_history', []), ('_chat_pending', None),
        ('chat_sessions', {}), ('active_session_id', None),
        ('_show_history', False), ('_last_web_search_result', ''),
        ('_last_web_search_query', ''),
    ]:
        if k not in st.session_state:
            st.session_state[k] = default

    # ── Handle query_params ───────────────────────────────────────────────────
    _qp = st.query_params.to_dict()
    if "hist_open" in _qp:
        _target = _qp["hist_open"]
        _all_s_tmp = _sessions_load()
        if _target in _all_s_tmp:
            _session_load(_target)
            st.session_state["_last_web_search_result"] = ""
            st.session_state["_show_history"] = False
        st.query_params.clear(); st.rerun()
    if "hist_del" in _qp:
        _session_delete(_qp["hist_del"])
        if not st.session_state.active_session_id:
            st.session_state.active_session_id = _session_new()
        st.query_params.clear(); st.rerun()
    if "hist_new" in _qp:
        _ns = _session_new()
        st.session_state.active_session_id = _ns
        st.session_state.chat_history = []
        st.session_state["_last_web_search_result"] = ""
        st.session_state["_show_history"] = False
        st.query_params.clear(); st.rerun()

    # ── Init session ──────────────────────────────────────────────────────────
    _all_sessions = _sessions_load()
    st.session_state.chat_sessions = _all_sessions
    if not st.session_state.active_session_id:
        st.session_state.active_session_id = _session_new()

    # ── AI badge ──────────────────────────────────────────────────────────────
    try:
        _ai_prov, _ai_key = _get_api_key()
        if _ai_prov == "gemini" and _ai_key:   _ai_badge = "🟢 Gemini 2.0 Flash"
        elif _ai_prov == "groq" and _ai_key:   _ai_badge = "🟡 Groq LLaMA 3.3 70B"
        else:                                   _ai_badge = "🔴 API Belum Diset"
    except Exception:
        _ai_badge = "🔴 API Belum Diset"; _ai_key = None

    # ── Process pending question ──────────────────────────────────────────────
    if st.session_state._chat_pending:
        _q = st.session_state._chat_pending
        st.session_state._chat_pending = None
        _api_provider = _get_api_key()[0] or "AI"
        _spinner_lbl = "🤖 Gemini AI menganalisis..." if _api_provider == "gemini" else "🤖 AI menganalisis..."
        with st.spinner(_spinner_lbl):
            _ans = _chat_answer(_q, df, active_progs, years, has_nom, latest_year, prev_year)
        st.session_state.chat_history.append({"role": "assistant", "content": _ans})
        _session_save_current()
        st.rerun()

    # ── Sessions list ─────────────────────────────────────────────────────────
    _sessions_sorted = sorted(
        st.session_state.chat_sessions.items(),
        key=lambda x: x[1].get("ts_raw", 0), reverse=True)
    _sessions_with_msgs = [(sid, sdata) for sid, sdata in _sessions_sorted
                           if len(sdata.get("messages", [])) > 0]
    _hist_items = []
    for _sid, _sdata in _sessions_with_msgs:
        _hist_items.append({
            "id": _sid, "title": _sdata.get("title", "Percakapan"),
            "ts": _sdata.get("ts", ""),
            "count": len([m for m in _sdata.get("messages", []) if m["role"] == "user"]),
            "active": _sid == st.session_state.active_session_id,
        })

    # ── Build chat HTML ───────────────────────────────────────────────────────
    _msgs_html_parts = []
    for _msg in st.session_state.chat_history[-30:]:
        _c = (_msg["content"]
              .replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
              .replace("\n","<br>"))
        if _msg["role"] == "user":
            _msgs_html_parts.append(
                '<div class="msg-row user"><div class="bubble user-bubble">' + _c + '</div></div>')
        else:
            _msgs_html_parts.append(
                '<div class="msg-row bot">'
                '<div class="bot-avatar">AI</div>'
                '<div class="bubble bot-bubble">' + _c + '</div></div>')
    _msgs_html = "\n".join(_msgs_html_parts)

    # ── Topbar ────────────────────────────────────────────────────────────────
    _tb1, _tb2, _tb3 = st.columns([1, 6, 2])
    with _tb1:
        if st.button("☰", key="hist_toggle", use_container_width=True):
            st.session_state._show_history = not st.session_state._show_history
            st.rerun()
    with _tb2:
        st.markdown(
            '<div style="background:linear-gradient(135deg,#1b5e20,#2e7d32,#43a047);'
            'border-radius:12px;padding:10px 18px;display:flex;align-items:center;justify-content:space-between;margin-top:2px;">'
            '<div style="display:flex;align-items:center;gap:12px;">'
            '<div style="width:36px;height:36px;background:rgba(255,255,255,.2);border-radius:50%;'
            'display:flex;align-items:center;justify-content:center;font-size:1.1rem;">🤖</div>'
            '<div><div style="font-size:.92rem;font-weight:800;color:white;">AI Analyst BPJS</div>'
            '<div style="font-size:.68rem;color:#a5d6a7;display:flex;align-items:center;gap:4px;margin-top:1px;">'
            '<span style="width:6px;height:6px;background:#69f0ae;border-radius:50%;display:inline-block;"></span>'
            ' Online</div></div></div>'
            f'<div style="font-size:.7rem;font-weight:600;padding:4px 12px;border-radius:16px;'
            f'background:rgba(255,255,255,.15);color:white;border:1px solid rgba(255,255,255,.25);">'
            + _ai_badge + '</div></div>', unsafe_allow_html=True)
    with _tb3:
        if st.button("✏️ New Chat", key="new_chat_btn", type="primary", use_container_width=True):
            _ns2 = _session_new()
            st.session_state.active_session_id = _ns2
            st.session_state.chat_history = []
            st.session_state["_last_web_search_result"] = ""
            st.session_state._show_history = False
            st.rerun()

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # ── Layout ────────────────────────────────────────────────────────────────
    if st.session_state._show_history:
        _left_col, _right_col = st.columns([1, 2])
    else:
        _left_col = None
        _right_col = st.container()

    # ── History panel ─────────────────────────────────────────────────────────
    if _left_col is not None:
        with _left_col:
            st.markdown("""
            <style>
            .hist-wrap button[kind="secondary"] {
                background:transparent !important; border:none !important;
                box-shadow:none !important; color:#d1d5db !important;
                text-align:left !important; justify-content:flex-start !important;
                font-size:.82rem !important; padding:8px 4px 4px 12px !important;
                border-radius:0 !important; min-height:0 !important;
                width:100% !important; margin:0 !important; line-height:1.3 !important;
            }
            .hist-wrap button[kind="secondary"]:hover { background:rgba(255,255,255,.08) !important; }
            .hist-wrap .del-btn button { padding:6px 8px !important; color:#4b5563 !important; }
            .hist-wrap .del-btn button:hover { color:#ef4444 !important; }
            .hist-wrap [data-testid="stHorizontalBlock"] { gap:0 !important; }
            </style>
            """, unsafe_allow_html=True)

            st.markdown(
                '<div style="background:#111827;border-radius:14px 14px 0 0;'
                'padding:13px 14px 10px;border-bottom:1px solid rgba(255,255,255,.07);">'
                '<span style="font-size:.84rem;font-weight:700;color:#e5e7eb;">🕐 Riwayat Chat</span>'
                '</div>', unsafe_allow_html=True)

            st.markdown('<div class="hist-wrap" style="background:#111827;border-radius:0 0 14px 14px;padding:4px 0 8px;">', unsafe_allow_html=True)

            if not _hist_items:
                st.markdown('<div style="padding:14px;font-size:.78rem;color:#4b5563;">Belum ada riwayat.</div>', unsafe_allow_html=True)
            else:
                for _h in _hist_items:
                    _border_c = "#4ade80" if _h["active"] else "transparent"
                    _bg_c     = "rgba(74,222,128,.08)" if _h["active"] else "transparent"
                    _short    = (_h["title"][:26] + "…") if len(_h["title"]) > 26 else _h["title"]
                    st.markdown(f'<div style="border-left:3px solid {_border_c};background:{_bg_c};">', unsafe_allow_html=True)
                    _rc1, _rc2 = st.columns([11, 1])
                    with _rc1:
                        if st.button(_short, key=f"hs_{_h['id']}", use_container_width=True):
                            _session_load(_h["id"])
                            st.session_state["_last_web_search_result"] = ""
                            st.session_state["_show_history"] = False
                            st.rerun()
                        st.markdown(
                            f'<p style="font-size:.61rem;color:#4b5563;margin:-4px 0 5px 12px;">'
                            + _h["ts"] + f' · {_h["count"]} pesan</p>', unsafe_allow_html=True)
                    with _rc2:
                        st.markdown('<div class="del-btn">', unsafe_allow_html=True)
                        if st.button("🗑", key=f"hd_{_h['id']}", use_container_width=True, help="Hapus"):
                            _session_delete(_h["id"])
                            if not st.session_state.active_session_id:
                                st.session_state.active_session_id = _session_new()
                            st.rerun()
                        st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown('<div style="border-top:1px solid rgba(255,255,255,.04);margin:0 12px;"></div></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ── Chat panel ────────────────────────────────────────────────────────────
    with _right_col:
        _last_search = st.session_state.get("_last_web_search_result", "")
        _last_query  = st.session_state.get("_last_web_search_query", "")
        if _last_search:
            with st.expander("🌐 Web research yang digunakan AI", expanded=False):
                st.caption(f"Query: **{_last_query}**")
                import re as _re
                for _src in _re.split(r'SUMBER_\d+:', _last_search):
                    _src = _src.strip()
                    if not _src: continue
                    _ls = {k.strip(): v.strip() for k, v in
                           [l.split(':', 1) for l in _src.split('\n') if ':' in l]}
                    _url   = _ls.get('URL', '')
                    _judul = _ls.get('Judul', _ls.get('Isi', '')[:60])
                    _isi   = _ls.get('Isi', _src[:200])
                    if _url:
                        st.markdown(f"**[{_judul}]({_url})**\n\n{_isi[:300]}", unsafe_allow_html=True)
                        st.divider()

        # Chat CSS
        st.markdown("""
        <style>
        .chat-area{padding:14px 8px;background:#f8fafc;border-radius:12px;
          border:1px solid #e2e8f0;min-height:380px;max-height:450px;
          overflow-y:auto;margin-bottom:8px;}
        .chat-area::-webkit-scrollbar{width:4px;}
        .chat-area::-webkit-scrollbar-thumb{background:#c8e6c9;border-radius:2px;}
        .msg-row{display:flex;margin:8px 4px;align-items:flex-end;gap:8px;}
        .msg-row.user{justify-content:flex-end;}
        .msg-row.bot{justify-content:flex-start;}
        .bubble{max-width:78%;font-size:.87rem;line-height:1.7;padding:10px 14px;word-break:break-word;}
        .user-bubble{background:linear-gradient(135deg,#1b5e20,#2e7d32);color:white;
          border-radius:16px 16px 3px 16px;box-shadow:0 2px 8px rgba(27,94,32,.25);}
        .bot-bubble{background:white;color:#1a202c;border:1px solid #e8f5e9;
          border-radius:3px 16px 16px 16px;box-shadow:0 1px 4px rgba(0,0,0,.06);}
        .bot-avatar{width:28px;height:28px;background:linear-gradient(135deg,#1b5e20,#43a047);
          border-radius:50%;display:flex;align-items:center;justify-content:center;
          font-size:.68rem;font-weight:800;color:white;flex-shrink:0;}
        </style>""", unsafe_allow_html=True)

        if not st.session_state.chat_history:
            st.markdown(
                '<div class="chat-area">'
                '<div style="text-align:center;padding:50px 20px 20px;">'
                '<div style="font-size:2.8rem;margin-bottom:12px;">🤖</div>'
                '<div style="font-size:1.1rem;font-weight:800;color:#1b5e20;margin-bottom:8px;">'
                'Halo! Saya AI Analyst BPJS.</div>'
                '<div style="font-size:.87rem;color:#388e3c;line-height:1.8;">'
                'Tanya saya tentang data klaim, tren, analisis program,<br>'
                'perbandingan antar tahun, atau insight makroekonomi.</div>'
                f'<div style="display:flex;flex-wrap:wrap;gap:8px;justify-content:center;margin-top:20px;">'
                f'<span style="background:white;border:1px solid #c8e6c9;border-radius:20px;'
                f'padding:7px 13px;font-size:.77rem;color:#2e7d32;font-weight:500;">📊 Program terbesar {latest_year}?</span>'
                f'<span style="background:white;border:1px solid #c8e6c9;border-radius:20px;'
                f'padding:7px 13px;font-size:.77rem;color:#2e7d32;font-weight:500;">📈 Tren klaim 5 tahun</span>'
                f'<span style="background:white;border:1px solid #c8e6c9;border-radius:20px;'
                f'padding:7px 13px;font-size:.77rem;color:#2e7d32;font-weight:500;">🦠 Dampak COVID</span>'
                f'<span style="background:white;border:1px solid #c8e6c9;border-radius:20px;'
                f'padding:7px 13px;font-size:.77rem;color:#2e7d32;font-weight:500;">🔮 Prediksi tahun depan</span>'
                f'</div></div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="chat-area">' + _msgs_html + '</div>', unsafe_allow_html=True)

        # Quick chips
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

        # Input
        _ic, _bc = st.columns([8, 1])
        with _ic:
            _user_input = st.text_input("msg", key="ai_tab_input",
                placeholder="Ketik pesan...", label_visibility="collapsed")
        with _bc:
            _send_btn = st.button("➤", key="ai_tab_send", type="primary", use_container_width=True)

        if _send_btn and _user_input and _user_input.strip():
            st.session_state.chat_history.append({"role": "user", "content": _user_input.strip()})
            st.session_state._chat_pending = _user_input.strip()
            st.rerun()