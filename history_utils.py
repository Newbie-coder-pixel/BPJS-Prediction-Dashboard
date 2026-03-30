"""
history_utils.py — Manajemen riwayat analisis (load/save/delete).
"""
import os
import json
import pickle
from datetime import datetime
import streamlit as st

HISTORY_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.bpjs_history')
HISTORY_META = os.path.join(HISTORY_DIR, 'history_meta.json')
os.makedirs(HISTORY_DIR, exist_ok=True)


def _hpath(eid: str) -> str:
    return os.path.join(HISTORY_DIR, f'{eid}.pkl')


def load_history_meta() -> list:
    if not os.path.exists(HISTORY_META):
        return []
    try:
        with open(HISTORY_META, 'r') as f:
            return json.load(f)
    except Exception:
        return []


def save_history_meta(meta_list: list):
    with open(HISTORY_META, 'w') as f:
        json.dump(meta_list, f, ensure_ascii=False, indent=2)


def save_history_entry(eid: str, df, results: dict, extra: dict = None) -> bool:
    try:
        with open(_hpath(eid), 'wb') as f:
            pickle.dump({'df': df, 'results': results, 'extra': extra or {}}, f)
        return True
    except Exception:
        return False


def load_history_entry(eid: str):
    p = _hpath(eid)
    if not os.path.exists(p):
        return None, {}, {}
    try:
        with open(p, 'rb') as f:
            d = pickle.load(f)
        return d.get('df'), d.get('results', {}), d.get('extra', {})
    except Exception:
        return None, {}, {}


def delete_history_entry(eid: str):
    p = _hpath(eid)
    if os.path.exists(p):
        os.remove(p)


def add_to_history(label: str, eid: str, df, results: dict, extra: dict = None):
    meta = load_history_meta()
    meta = [m for m in meta if m['id'] != eid]
    meta.append({'id': eid, 'label': label, 'timestamp': datetime.now().isoformat()})
    if len(meta) > 20:
        for m in meta[:-20]:
            delete_history_entry(m['id'])
        meta = meta[-20:]
    save_history_meta(meta)
    save_history_entry(eid, df, results, extra)


def init_session_defaults():
    for k, v in [('active_data', None), ('active_results', {}),
                 ('active_entry_id', None), ('history_loaded', False)]:
        if k not in st.session_state:
            st.session_state[k] = v


def render_history_sidebar():
    """Render riwayat analisis di sidebar."""
    with st.sidebar:
        st.markdown("---")
        st.markdown("**🕑 Riwayat Analisis**")
        st.caption("Riwayat tersimpan permanen — tidak hilang saat restart.")
        history_meta = load_history_meta()
        if history_meta:
            for h in reversed(history_meta):
                col_h, col_del = st.columns([5, 1])
                with col_h:
                    if st.button(h['label'], key=f"hbtn_{h['id']}", width='stretch'):
                        df_h, res_h, extra_h = load_history_entry(h['id'])
                        if df_h is not None:
                            st.session_state.active_data     = df_h
                            st.session_state.active_results  = res_h
                            st.session_state.active_entry_id = h['id']
                            for k, v in extra_h.items():
                                st.session_state[k] = v
                            st.rerun()
                        else:
                            st.warning("Data riwayat tidak ditemukan.")
                with col_del:
                    if st.button("🗑", key=f"hdel_{h['id']}", help="Hapus riwayat ini"):
                        delete_history_entry(h['id'])
                        meta = load_history_meta()
                        meta = [m for m in meta if m['id'] != h['id']]
                        save_history_meta(meta)
                        st.rerun()
            if st.button("🗑 Hapus Semua Riwayat", width='stretch'):
                for h in history_meta:
                    delete_history_entry(h['id'])
                save_history_meta([])
                st.rerun()
        else:
            st.caption("Belum ada riwayat tersimpan.")