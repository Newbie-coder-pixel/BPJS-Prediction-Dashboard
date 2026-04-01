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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Hide semua chrome Streamlit ────────────────────────────────────── */
    #MainMenu, footer, header, [data-testid="stToolbar"],
    [data-testid="stSidebarNav"], section[data-testid="stSidebar"],
    [data-testid="stDecoration"], [data-testid="stStatusWidget"] {
        display: none !important;
    }

    /* ── Fullscreen reset ────────────────────────────────────────────────── */
    html, body { height: 100% !important; margin: 0 !important; }

    .stApp {
        min-height: 100vh !important;
        background: #ffffff !important;
        font-family: 'Inter', sans-serif;
    }

    /* Hapus semua padding bawaan Streamlit */
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewBlockContainer"],
    section.main, .main > div, .block-container {
        padding: 0 !important;
        max-width: 100% !important;
        min-height: 100vh !important;
        background: transparent !important;
    }

    /* ── Panel kiri BPJS — fixed, full height ───────────────────────────── */
    .lo-brand {
        position: fixed;
        top: 0; left: 0;
        width: 44%;
        height: 100vh;
        background: linear-gradient(155deg, #001f4d 0%, #003F8A 55%, #0055b8 100%);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        padding: 56px 52px;
        overflow: hidden;
        z-index: 10;
    }
    /* Triangle lime kanan atas */
    .lo-brand::before {
        content: '';
        position: absolute; top: 0; right: 0;
        width: 0; height: 0;
        border-style: solid;
        border-width: 0 220px 220px 0;
        border-color: transparent rgba(151,193,31,0.24) transparent transparent;
    }
    /* Triangle hijau kanan bawah */
    .lo-brand::after {
        content: '';
        position: absolute; bottom: 0; right: 0;
        width: 0; height: 0;
        border-style: solid;
        border-width: 0 0 280px 280px;
        border-color: transparent transparent rgba(0,138,75,0.22) transparent;
    }
    .lo-glow {
        position: absolute;
        width: 420px; height: 420px; border-radius: 50%;
        background: radial-gradient(circle, rgba(0,138,75,0.20) 0%, transparent 65%);
        bottom: -120px; left: -100px;
        pointer-events: none;
    }
    .lo-logo-box {
        width: 58px; height: 58px; border-radius: 15px;
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.22);
        display: flex; align-items: center; justify-content: center;
        font-size: 1.65rem; margin-bottom: 32px;
        position: relative; z-index: 1;
    }
    .lo-badge {
        display: inline-block;
        font-size: .64rem; font-weight: 700;
        color: #003F8A; background: #97C11F;
        border-radius: 5px; padding: 4px 12px;
        text-transform: uppercase; letter-spacing: .9px;
        margin-bottom: 18px;
        position: relative; z-index: 1;
    }
    .lo-title {
        font-family: 'Inter', sans-serif;
        font-size: 2.4rem; font-weight: 700;
        color: #fff; line-height: 1.15;
        letter-spacing: -.3px; margin-bottom: 18px;
        position: relative; z-index: 1;
    }
    .lo-desc {
        font-size: .86rem; color: rgba(255,255,255,.54);
        line-height: 1.85; font-weight: 300; max-width: 280px;
        position: relative; z-index: 1;
    }
    .lo-chips {
        display: flex; gap: 12px;
        position: relative; z-index: 1;
    }
    .lo-chip {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.14);
        border-radius: 10px; padding: 13px 20px;
    }
    .lo-chip-num {
        font-family: 'Inter', sans-serif;
        font-size: 1.15rem; font-weight: 700;
        color: #97C11F; line-height: 1;
    }
    .lo-chip-lbl {
        font-size: .61rem; color: rgba(255,255,255,.42);
        margin-top: 4px; text-transform: uppercase; letter-spacing: .6px;
    }

    /* ── Panel kanan — Streamlit columns, posisi di sebelah kanan brand ── */
    /* Semua elemen Streamlit di sisi kanan harus z-index > 10 */
    [data-testid="stHorizontalBlock"] {
        min-height: 100vh !important;
        background: #ffffff !important;
        align-items: stretch !important;
    }

    /* Column pertama (gap/spacer) transparan */
    [data-testid="stHorizontalBlock"] > div:first-child {
        background: transparent !important;
        pointer-events: none !important;
    }

    /* Column kedua (form area) white, full height, vertically centered */
    [data-testid="stHorizontalBlock"] > div:last-child {
        background: #ffffff !important;
        border-left: 4px solid #97C11F !important;
        min-height: 100vh !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        position: relative !important;
        z-index: 20 !important;
    }

    /* Inner columns (fl, fm, fr) juga full height centered */
    [data-testid="stHorizontalBlock"] > div:last-child
    [data-testid="stHorizontalBlock"] {
        min-height: unset !important;
        background: transparent !important;
        border-left: none !important;
        flex: 1 !important;
    }

    /* Form container */
    .form-right-wrap {
        width: 100%;
        max-width: 400px;
        margin: 0 auto;
        padding: 0 12px;
    }
    .fr-eyebrow {
        font-size: .68rem; font-weight: 600;
        color: #008A4B; text-transform: uppercase; letter-spacing: 1.3px;
        margin-bottom: 10px;
    }
    .fr-heading {
        font-family: 'Inter', sans-serif;
        font-size: 2rem; font-weight: 800;
        color: #0a1628; letter-spacing: -.5px; margin-bottom: 6px;
    }
    .fr-sub {
        font-size: .82rem; color: #8a9ab0; margin-bottom: 0;
        line-height: 1.6;
    }
    .fr-divider {
        height: 2px;
        background: linear-gradient(90deg, #003F8A, #008A4B, #97C11F);
        border-radius: 2px; margin: 22px 0 26px; opacity: .22;
    }
    .fr-sec {
        display: flex; align-items: center; gap: 8px;
        font-size: .72rem; color: #94a3b8; margin-top: 18px;
    }
    .fr-dot {
        width: 7px; height: 7px; border-radius: 50%;
        background: #008A4B;
        box-shadow: 0 0 7px rgba(0,138,75,.5);
        flex-shrink: 0;
    }

    /* ── Streamlit widget overrides ─────────────────────────────────────── */
    div[data-testid="stForm"] {
        background: transparent !important;
        border: none !important; padding: 0 !important;
    }
    div[data-testid="stForm"] .stTextInput > label {
        font-size: .7rem !important; font-weight: 600 !important;
        color: #5a6a80 !important; text-transform: uppercase !important;
        letter-spacing: .9px !important;
    }
    div[data-testid="stForm"] .stTextInput input {
        background: #f8fafc !important;
        border: 1.5px solid #e2e8f0 !important;
        border-radius: 10px !important; color: #0a1628 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: .9rem !important; padding: 13px 16px !important;
        height: 48px !important;
        transition: border-color .2s, box-shadow .2s !important;
    }
    div[data-testid="stForm"] .stTextInput input:focus {
        border-color: #003F8A !important;
        box-shadow: 0 0 0 3px rgba(0,63,138,.10) !important;
        outline: none !important;
    }
    div[data-testid="stForm"] .stTextInput input::placeholder {
        color: #b8c4d0 !important;
    }
    div[data-testid="stForm"] .stButton > button {
        background: linear-gradient(135deg, #003F8A 0%, #0056c7 100%) !important;
        color: #fff !important; border: none !important;
        border-radius: 10px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important; font-size: .9rem !important;
        letter-spacing: .5px !important;
        height: 50px !important; width: 100% !important;
        margin-top: 8px !important;
        box-shadow: 0 4px 20px rgba(0,63,138,.28) !important;
        transition: all .2s !important;
    }
    div[data-testid="stForm"] .stButton > button:hover {
        background: linear-gradient(135deg, #0056c7 0%, #003F8A 100%) !important;
        box-shadow: 0 6px 28px rgba(0,63,138,.42) !important;
        transform: translateY(-1px) !important;
    }
    div[data-testid="stAlert"] {
        border-radius: 10px !important; font-size: .82rem !important;
        border: none !important; margin-bottom: 10px !important;
    }

    /* Animasi */
    @keyframes fadeInLeft {
        from { opacity: 0; transform: translateX(-18px); }
        to   { opacity: 1; transform: translateX(0); }
    }
    @keyframes fadeInRight {
        from { opacity: 0; transform: translateX(18px); }
        to   { opacity: 1; transform: translateX(0); }
    }
    .lo-brand { animation: fadeInLeft .55s cubic-bezier(.22,.68,0,1.1) both; }
    .form-right-wrap { animation: fadeInRight .55s cubic-bezier(.22,.68,0,1.1) .1s both; }
    </style>
    """, unsafe_allow_html=True)

    # ── Panel kiri: brand BPJS (fixed HTML) ──────────────────────────────────
    st.markdown("""
    <div class="lo-brand">
        <div class="lo-glow"></div>
        <div>
            <div class="lo-logo-box">📊</div>
            <div class="lo-badge">Internal Tool</div>
            <div class="lo-title">BPJS ML<br>Dashboard</div>
            <div class="lo-desc">
                Platform analitik klaim &amp; prediksi berbasis
                Machine Learning untuk BPJS Ketenagakerjaan.
            </div>
        </div>
        <div class="lo-chips">
            <div class="lo-chip">
                <div class="lo-chip-num">5</div>
                <div class="lo-chip-lbl">Program</div>
            </div>
            <div class="lo-chip">
                <div class="lo-chip-num">ML</div>
                <div class="lo-chip-lbl">Prediksi</div>
            </div>
            <div class="lo-chip">
                <div class="lo-chip-num">AI</div>
                <div class="lo-chip-lbl">Analyst</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Panel kanan: spacer 44% | form 56% ───────────────────────────────────
    _spacer, _right = st.columns([44, 56])
    with _right:
        _fl, _fm, _fr = st.columns([1, 5, 1])
        with _fm:
            users      = _load_users()
            multi_user = len(users) > 1

            locked, remaining = _is_locked_out()
            if locked:
                st.error(f"🔒 Dikunci. Coba dalam **{remaining//60}m {remaining%60}s**.")
                st.stop()

            # Heading langsung di atas form — tidak ada wrapper min-height
            st.markdown("""
            <div style="padding-top:32px;">
                <div style="font-size:.68rem;font-weight:600;color:#008A4B;
                text-transform:uppercase;letter-spacing:1.3px;margin-bottom:10px;">
                    Selamat datang kembali
                </div>
                <div style="font-family:'Inter',sans-serif;font-size:1.75rem;
                font-weight:600;color:#0a1628;letter-spacing:-.2px;margin-bottom:6px;">
                    Silakan masuk
                </div>
                <div style="font-size:.82rem;color:#8a9ab0;line-height:1.6;margin-bottom:0;">
                    Akses terbatas untuk pegawai BPJS Ketenagakerjaan
                </div>
                <div style="height:2px;background:linear-gradient(90deg,#003F8A,#008A4B,#97C11F);
                border-radius:2px;margin:20px 0 24px;opacity:.22;"></div>
            </div>
            """, unsafe_allow_html=True)

            with st.form("login_form", clear_on_submit=True):
                if multi_user:
                    username = st.text_input("Username", placeholder="username").strip().lower()
                else:
                    username = "admin"
                password  = st.text_input("Password", type="password",
                                          placeholder="Masukkan password Anda")
                submitted = st.form_submit_button("Masuk →",
                                                  use_container_width=True, type="primary")

            st.markdown("""
            <div style="display:flex;align-items:center;gap:8px;
            font-size:.72rem;color:#94a3b8;margin-top:14px;padding-bottom:32px;">
                <div style="width:7px;height:7px;border-radius:50%;background:#008A4B;
                box-shadow:0 0 7px rgba(0,138,75,.5);flex-shrink:0;"></div>
                Koneksi terenkripsi &nbsp;·&nbsp; Internal use only
            </div>
            """, unsafe_allow_html=True)

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
                    st.error(f"❌ Password salah. Sisa percobaan: **{remaining_attempts}**")
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