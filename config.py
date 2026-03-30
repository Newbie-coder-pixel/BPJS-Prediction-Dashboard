"""
config.py — Konstanta global, CSS, warna, dan helper chart.
"""
import streamlit as st

# ── WARNA ─────────────────────────────────────────────────────────────────────
COLORS = [
    '#2563eb', '#16a34a', '#ea580c', '#7c3aed', '#dc2626',
    '#ca8a04', '#0891b2', '#db2777', '#65a30d', '#9333ea'
]

DARK = dict(
    template='plotly_white',
    paper_bgcolor='rgba(255,255,255,0)',
    plot_bgcolor='rgba(248,250,252,0.8)',
    font_color='#334155',
    font_family='Inter',
)

# ── HELPER ────────────────────────────────────────────────────────────────────
def hex_to_rgba(hex_c: str, alpha: float = 1.0) -> str:
    h = hex_c.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'


def styled_chart(fig, height=400, legend_bottom=True, margin_b=80):
    fig.update_layout(
        **DARK, height=height,
        hovermode='x unified',
        hoverlabel=dict(bgcolor='#ffffff', font_size=12, bordercolor='#e2e8f0',
                        font_color='#334155'),
        legend=dict(orientation='h', y=-0.22, font=dict(size=10.5, color='#475569'),
                    groupclick='toggleitem') if legend_bottom else {},
        margin=dict(b=margin_b if legend_bottom else 40, t=20, l=60, r=20),
        xaxis=dict(showgrid=True, gridcolor='rgba(226,232,240,0.8)', gridwidth=1,
                   zeroline=False, linecolor='#cbd5e1', tickfont=dict(color='#64748b')),
        yaxis=dict(showgrid=True, gridcolor='rgba(226,232,240,0.8)', gridwidth=1,
                   zeroline=False, linecolor='#cbd5e1', tickfont=dict(color='#64748b')),
    )
    return fig


# ── CSS GLOBAL ────────────────────────────────────────────────────────────────
GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
*{box-sizing:border-box;}
html,body,[class*="css"]{font-family:'Inter',sans-serif;}

.stApp{background:#f1f5f9 !important;color:#334155 !important;}
.main .block-container{background:#f1f5f9;padding-top:1.5rem;}
section[data-testid="stSidebar"]{background:#ffffff !important;border-right:1px solid #e2e8f0 !important;}
section[data-testid="stSidebar"] > div{background:#ffffff !important;}

::-webkit-scrollbar{width:5px;height:5px;}
::-webkit-scrollbar-track{background:#f1f5f9;}
::-webkit-scrollbar-thumb{background:#cbd5e1;border-radius:4px;}
::-webkit-scrollbar-thumb:hover{background:#94a3b8;}

.kpi{background:#ffffff;border:1px solid #e2e8f0;border-radius:14px;
  padding:20px 14px 16px;text-align:center;
  box-shadow:0 1px 4px rgba(0,0,0,.06);
  transition:transform .2s,box-shadow .2s;cursor:default;}
.kpi:hover{transform:translateY(-2px);box-shadow:0 4px 16px rgba(0,0,0,.10);}
.kpi::before{content:'';position:absolute;display:none;}
.kpi .val{font-size:1.7rem;font-weight:800;color:#0f172a;font-family:'JetBrains Mono',monospace;line-height:1.2;}
.kpi .lbl{font-size:.68rem;color:#64748b;text-transform:uppercase;letter-spacing:1.2px;margin-top:6px;font-weight:600;}
.kpi .delta{font-size:.75rem;margin-top:5px;font-weight:500;}
.delta-pos{color:#16a34a;} .delta-neg{color:#dc2626;} .delta-neu{color:#64748b;}

.sec{font-size:.7rem;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:1.5px;
  margin:24px 0 10px;padding-bottom:6px;border-bottom:2px solid #e2e8f0;}

.badge{background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;
  padding:12px 16px;font-size:.84rem;line-height:1.8;border-left:3px solid #3b82f6;color:#1e40af;}
.badge b{color:#1d4ed8;}
.warn{background:#fffbeb;border:1px solid #fde68a;border-radius:10px;
  padding:12px 16px;color:#92400e;font-size:.84rem;margin:8px 0;
  border-left:3px solid #f59e0b;line-height:1.7;}
.info-box{background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;
  padding:12px 16px;font-size:.84rem;color:#166534;margin:8px 0;line-height:1.8;
  border-left:3px solid #22c55e;}
.success-box{background:#f0fdf4;border:1px solid #86efac;border-radius:10px;
  padding:12px 16px;font-size:.84rem;color:#166534;margin:8px 0;
  border-left:3px solid #16a34a;line-height:1.8;}
.insight-note{background:#fafafa;border:1px solid #e2e8f0;border-radius:8px;
  padding:10px 14px;font-size:.82rem;color:#475569;line-height:1.7;margin:6px 0;}

.stTabs [data-baseweb="tab-list"]{gap:2px;background:#e2e8f0;border-radius:10px;padding:3px;border:none;}
.stTabs [data-baseweb="tab"]{border-radius:8px;padding:8px 18px;font-size:.84rem;font-weight:500;color:#475569;}
.stTabs [aria-selected="true"]{background:#ffffff !important;color:#0f172a !important;
  box-shadow:0 1px 4px rgba(0,0,0,.12) !important;}

.stButton>button{border-radius:8px;font-weight:600;font-size:.85rem;transition:all .2s;}
.stButton>button[kind="primary"]{background:#2563eb !important;border:none !important;
  color:#fff !important;box-shadow:0 2px 8px rgba(37,99,235,.35);}
.stButton>button[kind="primary"]:hover{background:#1d4ed8 !important;}

.stSelectbox [data-baseweb="select"] > div{background:#ffffff !important;border-color:#e2e8f0 !important;color:#334155 !important;}
.stTextInput input{background:#ffffff !important;border:1px solid #e2e8f0 !important;
  border-radius:8px !important;color:#334155 !important;}
.stDataFrame{border-radius:10px;overflow:hidden;border:1px solid #e2e8f0;}
[data-testid="stExpander"]{border:1px solid #e2e8f0;border-radius:10px;background:#ffffff;}
[data-testid="stExpander"] summary{background:#ffffff;color:#475569;font-size:.85rem;}

section[data-testid="stSidebar"] label{color:#475569 !important;font-size:.82rem !important;}
section[data-testid="stSidebar"] .stMarkdown{color:#475569;}
section[data-testid="stSidebar"] .stButton>button{background:#f8fafc;border:1px solid #e2e8f0;
  color:#475569;border-radius:8px;font-size:.82rem;transition:all .2s;}
section[data-testid="stSidebar"] .stButton>button:hover{border-color:#2563eb;color:#2563eb;}

.hero-wrap{padding:20px 24px;background:#ffffff;border:1px solid #e2e8f0;
  border-radius:14px;border-left:4px solid #2563eb;margin-bottom:8px;
  box-shadow:0 1px 4px rgba(0,0,0,.05);}
.hero-logo{font-size:1.35rem;font-weight:800;color:#0f172a;line-height:1.2;margin-bottom:2px;}
.hero-sub{font-size:.78rem;color:#64748b;letter-spacing:.3px;}

.mpill{display:inline-block;padding:2px 10px;border-radius:20px;font-size:.75rem;font-weight:600;margin:2px;}
.mpill-green{background:#dcfce7;color:#15803d;border:1px solid #86efac;}
.mpill-blue{background:#dbeafe;color:#1d4ed8;border:1px solid #93c5fd;}
.mpill-yellow{background:#fef9c3;color:#854d0e;border:1px solid #fde047;}
.mpill-red{background:#fee2e2;color:#b91c1c;border:1px solid #fca5a5;}

.concl-card{background:#ffffff;border:1px solid #e2e8f0;border-radius:10px;
  padding:16px 20px;margin:8px 0;border-left:3px solid #2563eb;}
.concl-card .ct{font-size:.7rem;color:#64748b;text-transform:uppercase;letter-spacing:1px;font-weight:600;margin-bottom:5px;}
.concl-card .cv{color:#334155;font-size:.88rem;line-height:1.8;}

.prog-card{background:#ffffff;border:1px solid #e2e8f0;border-radius:10px;padding:10px 14px;min-width:120px;display:inline-block;}
.prog-card .pc-name{font-size:.68rem;color:#64748b;text-transform:uppercase;letter-spacing:.8px;font-weight:600;margin-bottom:3px;}
.prog-card .pc-model{font-size:.88rem;font-weight:700;color:#0f172a;margin-bottom:4px;}

[data-testid="stDownloadButton"] button{background:#059669 !important;
  border:1px solid #047857 !important;color:#fff !important;border-radius:8px;font-weight:600;}

.tag-add{display:inline-block;background:#dcfce7;color:#15803d;border:1px solid #86efac;
  border-radius:5px;padding:2px 8px;font-size:.76rem;margin:2px;font-weight:600;}
.tag-rem{display:inline-block;background:#fee2e2;color:#b91c1c;border:1px solid #fca5a5;
  border-radius:5px;padding:2px 8px;font-size:.76rem;margin:2px;font-weight:600;}
.tag-stable{display:inline-block;background:#f1f5f9;color:#475569;border:1px solid #cbd5e1;
  border-radius:5px;padding:2px 8px;font-size:.76rem;margin:2px;font-weight:600;}

.insight-card{background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;padding:18px 20px;
  box-shadow:0 1px 4px rgba(0,0,0,.05);}
.insight-card .ic-title{font-size:.7rem;color:#64748b;text-transform:uppercase;letter-spacing:1px;font-weight:600;margin-bottom:6px;}
.insight-card .ic-val{font-size:1.3rem;font-weight:700;color:#0f172a;}
.insight-card .ic-sub{font-size:.8rem;color:#64748b;margin-top:3px;}

[data-testid="stAlert"]{border-radius:10px;}

.feat-card{background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;padding:22px 18px;
  text-align:center;height:180px;border-top:3px solid #2563eb;
  box-shadow:0 1px 4px rgba(0,0,0,.05);transition:transform .2s;}
.feat-card:hover{transform:translateY(-3px);}
.feat-icon{font-size:1.8rem;margin-bottom:8px;}
.feat-title{font-weight:700;color:#0f172a;margin-bottom:6px;font-size:.92rem;}
.feat-desc{color:#64748b;font-size:.8rem;line-height:1.6;}

.export-box{background:#ffffff;border:1px solid #e2e8f0;border-radius:10px;padding:16px 18px;line-height:2;font-size:.84rem;}
.export-box .ok{color:#16a34a;} .export-box .nok{color:#94a3b8;}

.empty-state{text-align:center;padding:70px 0 50px;}
.empty-icon{font-size:3.5rem;margin-bottom:16px;}
.empty-title{font-size:1.8rem;font-weight:800;color:#0f172a;margin-bottom:12px;}
.empty-sub{color:#64748b;max-width:500px;margin:auto;font-size:.9rem;line-height:1.8;}
</style>
"""


def inject_global_css():
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


KEEPALIVE_JS = """
<script>
(function keepAlive() {
  setInterval(function() {
    fetch(window.location.href, {method: 'GET', cache: 'no-cache'}).catch(function() {});
  }, 600000);
})();
</script>
"""