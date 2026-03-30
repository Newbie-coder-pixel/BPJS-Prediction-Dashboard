"""
auth.py — Autentikasi, login, session management.
"""
import hashlib
import time
import os
import streamlit as st

MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_SECONDS    = 300
SESSION_TIMEOUT    = 28800


# ── Secret helpers ────────────────────────────────────────────────────────────
def _secret(key: str, default=""):
    try:
        val = st.secrets[key]
        return val if val is not None else default
    except Exception:
        return os.environ.get(key, default)


def _secret_section(section: str) -> dict:
    try:
        return dict(st.secrets[section])
    except Exception:
        return {}


# ── Password helpers ──────────────────────────────────────────────────────────
def _hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.strip().encode()).hexdigest()


def _load_users() -> dict:
    users = {}
    try:
        users_section = _secret_section("users")
        for uname, hashed in users_section.items():
            users[uname.lower()] = str(hashed)
    except Exception:
        pass
    if not users:
        raw_pw = _secret("DASHBOARD_PASSWORD", "bpjs2026")
        users["admin"] = _hash_pw(raw_pw)
    return users


# ── Session state helpers ─────────────────────────────────────────────────────
def _init_security_state():
    defaults = {
        "authenticated": False,
        "auth_user": None,
        "auth_time": None,
        "login_attempts": 0,
        "lockout_until": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _is_locked_out() -> tuple:
    if st.session_state.lockout_until > time.time():
        remaining = int(st.session_state.lockout_until - time.time())
        return True, remaining
    return False, 0


def _is_session_expired() -> bool:
    if not st.session_state.auth_time:
        return True
    return (time.time() - st.session_state.auth_time) > SESSION_TIMEOUT


def _do_logout():
    st.session_state.authenticated = False
    st.session_state.auth_user     = None
    st.session_state.auth_time     = None


# ── Main auth function ────────────────────────────────────────────────────────
def check_password():
    _init_security_state()

    if st.session_state.authenticated:
        if _is_session_expired():
            _do_logout()
            st.warning("⏰ Sesi Anda telah berakhir. Silakan login kembali.")
            st.rerun()
        else:
            return

    st.markdown("""
    <style>
    #MainMenu, footer, header, [data-testid="stToolbar"],
    [data-testid="stSidebarNav"], section[data-testid="stSidebar"] {display:none!important;}
    .stApp {background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 50%, #eff6ff 100%) !important;}
    </style>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        st.markdown("""
        <div style="background:white;border-radius:20px;padding:40px 36px 32px;
        box-shadow:0 25px 60px rgba(0,0,0,.4);margin-top:60px;">
          <div style="text-align:center;margin-bottom:28px;">
            <div style="font-size:2.8rem;">📊</div>
            <div style="font-size:1.35rem;font-weight:800;color:#0f172a;margin:8px 0 4px;">
              BPJS ML Dashboard
            </div>
            <div style="font-size:.82rem;color:#64748b;">
              Internal Tool — BPJS Ketenagakerjaan
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        locked, remaining = _is_locked_out()
        if locked:
            st.error(f"🔒 Terlalu banyak percobaan gagal. Coba lagi dalam **{remaining//60}m {remaining%60}s**.")
            st.stop()

        users     = _load_users()
        multi_user = len(users) > 1

        with st.form("login_form", clear_on_submit=True):
            if multi_user:
                username = st.text_input("👤 Username", placeholder="username").strip().lower()
            else:
                username = "admin"
            password  = st.text_input("🔑 Password", type="password", placeholder="••••••••")
            submitted = st.form_submit_button("🚀 Masuk", use_container_width=True, type="primary")

        if submitted:
            if not password or (multi_user and not username):
                st.warning("Lengkapi semua field.")
                st.stop()

            hashed_input = _hash_pw(password)
            stored_hash  = users.get(username) if multi_user else list(users.values())[0]

            if stored_hash and hashed_input == stored_hash:
                st.session_state.authenticated  = True
                st.session_state.auth_user      = username
                st.session_state.auth_time      = time.time()
                st.session_state.login_attempts = 0
                st.session_state.lockout_until  = 0
                st.rerun()
            else:
                st.session_state.login_attempts += 1
                remaining_attempts = MAX_LOGIN_ATTEMPTS - st.session_state.login_attempts
                if st.session_state.login_attempts >= MAX_LOGIN_ATTEMPTS:
                    st.session_state.lockout_until  = time.time() + LOCKOUT_SECONDS
                    st.session_state.login_attempts = 0
                    st.error(f"🔒 Akun dikunci selama {LOCKOUT_SECONDS//60} menit.")
                else:
                    st.error(f"❌ Username atau password salah. Sisa percobaan: **{remaining_attempts}**")
                st.stop()

    st.stop()


def render_user_badge():
    """Tampilkan info user & tombol logout di sidebar."""
    with st.sidebar:
        st.markdown("---")
        _user       = st.session_state.get("auth_user", "admin")
        _login_time = st.session_state.get("auth_time")
        _elapsed    = int(time.time() - _login_time) if _login_time else 0
        _sisa_menit = max(0, (SESSION_TIMEOUT - _elapsed) // 60)
        st.markdown(f"""
        <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;
        padding:10px 14px;font-size:.78rem;color:#475569;line-height:1.9;">
        👤 <b>{_user}</b><br>
        ⏰ Sesi berakhir: ~{_sisa_menit} menit
        </div>""", unsafe_allow_html=True)
        st.markdown("")
        if st.button("🚪 Logout", width='stretch'):
            _do_logout()
            st.rerun()