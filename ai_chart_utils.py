"""
ai_chart_utils.py — Chart engine untuk AI Analyst BPJS.

Alur:
  1. detect_chart_intent(question)  → ChartIntent (enum)
  2. build_chart_data(df, intent, years, active_progs, has_nom, latest_year)
     → dict {type, fig, title, subtitle}  atau  None
  3. render_ai_chart(chart_data, COLORS, DARK)  → dipanggil dari tab_ai.py

Dipanggil juga dari api/main.py untuk endpoint /ask (kembalikan JSON fig).
"""
from __future__ import annotations
from enum import Enum
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st


# ══════════════════════════════════════════════════════════════════════════════
# INTENT ENUM
# ══════════════════════════════════════════════════════════════════════════════

class ChartIntent(str, Enum):
    TREND_LINE      = "trend_line"       # tren kasus/nominal historis per program
    FORECAST_BAR    = "forecast_bar"     # prediksi bar chart + CI
    COMPARISON_BAR  = "comparison_bar"   # perbandingan antar program satu tahun
    PIE_SHARE       = "pie_share"        # distribusi kasus per program
    YOY_DELTA       = "yoy_delta"        # delta YoY per program (waterfall-like)
    SEBAB_KLAIM_BAR = "sebab_klaim_bar"  # top sebab klaim per program
    NO_CHART        = "no_chart"         # tidak perlu chart


# ══════════════════════════════════════════════════════════════════════════════
# INTENT DETECTION
# ══════════════════════════════════════════════════════════════════════════════

# ── Keyword ekslusif per intent — tidak overlap ──────────────────────────────

# Pertanyaan yang TIDAK perlu chart meski mengandung kata analitik
_NO_CHART_PATTERNS = [
    # Pertanyaan "kenapa/mengapa" tentang perubahan = analisis naratif, bukan chart sebab
    ("kenapa ada", ),
    ("mengapa ada", ),
    ("kenapa terjadi", ),
    ("mengapa terjadi", ),
    ("apa penyebab", ),
    ("apa alasan", ),
    # Pertanyaan definisi/konsep
    ("apa itu", "apa yang dimaksud", "jelaskan apa", "pengertian"),
    # Pertanyaan rekomendasi/saran
    ("apa rekomendasi", "apa saran", "apa yang harus", "bagaimana cara"),
]

# Kata yang WAJIB ADA agar SEBAB_KLAIM_BAR aktif
# (harus spesifik minta data sebab, bukan hanya bertanya "kenapa")
_SEBAB_REQUIRED = [
    "sebab klaim", "penyebab klaim", "data sebab", "top sebab",
    "list sebab", "tabel sebab", "alasan klaim", "driver klaim",
    "sebab terbesar", "penyebab terbesar",
]

# Chart hanya muncul jika pertanyaan EKSPLISIT minta visual
_EXPLICIT_CHART_TRIGGERS = [
    "grafik", "chart", "diagram", "visualisasi", "tampilkan", "tunjukkan",
    "lihat grafik", "buat chart", "plot",
]


def detect_chart_intent(question: str) -> ChartIntent:
    """
    Deteksi apakah pertanyaan memerlukan chart dan chart apa.

    Filosofi:
    - Chart hanya muncul jika pertanyaan BENAR-BENAR memerlukan visualisasi.
    - Pertanyaan analitik naratif ("kenapa ada penurunan?") → NO_CHART.
    - Pertanyaan yang eksplisit minta data visual → chart sesuai tipe.
    - Jika ragu → NO_CHART (lebih baik tidak ada chart daripada chart salah).
    """
    q = question.lower().strip()

    # ── Pertanyaan terlalu pendek atau trivial ────────────────────────────────
    trivial = [
        "halo", "hai", "apa itu", "siapa", "terima kasih", "ok", "oke",
        "bagus", "mantap", "tolong", "bantu", "cara", "gimana",
    ]
    if len(q) < 12 or any(q.startswith(t) for t in trivial):
        return ChartIntent.NO_CHART

    # ── Pola yang TIDAK perlu chart (cek lebih dulu sebelum rules lain) ───────
    for patterns in _NO_CHART_PATTERNS:
        if any(p in q for p in patterns):
            return ChartIntent.NO_CHART

    # ── FORECAST — kata kunci prediksi/tahun masa depan ──────────────────────
    _forecast_kw = [
        "prediksi", "forecast", "proyeksi", "tahun depan", "estimasi",
        "perkiraan", "2026", "2027", "2028", "outlook",
    ]
    if any(k in q for k in _forecast_kw):
        return ChartIntent.FORECAST_BAR

    # ── TREND LINE — kata kunci tren historis ────────────────────────────────
    _trend_kw = [
        "tren", "trend", "historis", "dari tahun ke tahun",
        "yoy", "5 tahun", "fluktuasi", "grafik tren",
        "perkembangan dari", "naik dari tahun", "turun dari tahun",
    ]
    if any(k in q for k in _trend_kw):
        return ChartIntent.TREND_LINE

    # ── TREND LINE juga jika ada trigger eksplisit visual + kata pertumbuhan ──
    _growth_kw = ["pertumbuhan", "perkembangan", "perubahan", "naik turun"]
    if any(t in q for t in _EXPLICIT_CHART_TRIGGERS) and any(k in q for k in _growth_kw):
        return ChartIntent.TREND_LINE

    # ── PIE SHARE — distribusi/porsi ─────────────────────────────────────────
    _pie_kw = [
        "porsi", "komposisi", "distribusi", "proporsi", "share",
        "berapa persen", "pie chart", "diagram lingkaran",
    ]
    if any(k in q for k in _pie_kw):
        return ChartIntent.PIE_SHARE

    # ── COMPARISON BAR — perbandingan antar program ───────────────────────────
    _comp_kw = [
        "bandingkan", "perbandingan", " vs ", "versus", "dibanding",
        "program mana yang lebih", "ranking program", "urutan program",
    ]
    if any(k in q for k in _comp_kw):
        return ChartIntent.COMPARISON_BAR

    # ── YOY DELTA — selisih tahun ini vs tahun lalu ───────────────────────────
    _yoy_kw = [
        "selisih", "delta klaim", "berapa naik", "berapa turun",
        "berubah berapa", "naik berapa persen", "turun berapa persen",
    ]
    if any(k in q for k in _yoy_kw):
        return ChartIntent.YOY_DELTA

    # ── SEBAB KLAIM BAR — hanya jika eksplisit minta data sebab klaim ────────
    # TIDAK aktif hanya karena ada "kenapa" atau "mengapa"
    if any(k in q for k in _SEBAB_REQUIRED):
        return ChartIntent.SEBAB_KLAIM_BAR

    # ── Jika ada trigger visual eksplisit tapi tidak cocok di atas → trend ────
    if any(t in q for t in _EXPLICIT_CHART_TRIGGERS):
        return ChartIntent.TREND_LINE

    return ChartIntent.NO_CHART


# ══════════════════════════════════════════════════════════════════════════════
# CHART DATA BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def _safe_get_df(*keys: str) -> pd.DataFrame | None:
    """Ambil DataFrame dari session_state dengan aman (tanpa 'or' pada DataFrame)."""
    for key in keys:
        val = st.session_state.get(key)
        if val is None:
            continue
        if not isinstance(val, pd.DataFrame):
            continue
        if val.empty:
            continue
        return val
    return None


def _build_trend_line(
    df: pd.DataFrame,
    active_progs: list,
    years: list,
    has_nom: bool,
    target: str,
    colors: list,
    dark: dict,
) -> go.Figure | None:
    """Line chart tren historis per program."""
    if df is None or df.empty:
        return None

    col = "Nominal" if (target == "Nominal" and has_nom and "Nominal" in df.columns) else "Kasus"
    df_f = df[df["Kategori"].isin(active_progs)].copy()
    if df_f.empty:
        return None

    fig = go.Figure()
    for i, prog in enumerate(sorted(active_progs)):
        sub = df_f[df_f["Kategori"] == prog].sort_values("Tahun")
        if sub.empty:
            continue
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=sub["Tahun"], y=sub[col],
            name=prog, mode="lines+markers",
            line=dict(color=color, width=2.5),
            marker=dict(size=8),
        ))

    unit = "Rp" if col == "Nominal" else "kasus"
    fig.update_layout(
        **dark,
        height=340,
        margin=dict(t=10, b=40, l=60, r=20),
        xaxis=dict(dtick=1, title="Tahun"),
        yaxis=dict(title=f"{col} ({unit})"),
        legend=dict(orientation="h", y=-0.25, font=dict(size=11)),
        hovermode="x unified",
    )
    return fig


def _build_forecast_bar(
    df: pd.DataFrame,
    active_progs: list,
    has_nom: bool,
    target: str,
    colors: list,
    dark: dict,
) -> go.Figure | None:
    """Bar chart prediksi per program dengan CI band."""
    # Coba ambil forecast dari session_state
    fc = _safe_get_df(f"forecast_{target}", "forecast_Kasus", "last_forecast")
    if fc is None:
        return None

    col = target if target in fc.columns else "Kasus"
    upper_col = f"{col}_upper"
    lower_col = f"{col}_lower"
    has_ci = upper_col in fc.columns and lower_col in fc.columns

    fig = go.Figure()
    for i, prog in enumerate(sorted(fc["Kategori"].unique())):
        if prog not in active_progs:
            continue
        sub = fc[fc["Kategori"] == prog].sort_values("Tahun")
        color = colors[i % len(colors)]

        if has_ci:
            fig.add_trace(go.Scatter(
                x=list(sub["Tahun"]) + list(sub["Tahun"][::-1]),
                y=list(sub[upper_col]) + list(sub[lower_col][::-1]),
                fill="toself",
                fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.12)",
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
                hoverinfo="skip",
            ))

        fig.add_trace(go.Bar(
            x=sub["Tahun"], y=sub[col],
            name=prog,
            marker_color=color,
            text=sub[col].apply(lambda v: f"{int(v):,}"),
            textposition="outside",
        ))

    fig.update_layout(
        **dark,
        height=340,
        barmode="group",
        margin=dict(t=10, b=40, l=60, r=20),
        xaxis=dict(dtick=1, title="Tahun"),
        yaxis=dict(title=col),
        legend=dict(orientation="h", y=-0.25, font=dict(size=11)),
    )
    return fig


def _build_comparison_bar(
    df: pd.DataFrame,
    active_progs: list,
    latest_year: int,
    has_nom: bool,
    target: str,
    colors: list,
    dark: dict,
) -> go.Figure | None:
    """Bar chart perbandingan per program untuk tahun terbaru."""
    col = "Nominal" if (target == "Nominal" and has_nom) else "Kasus"
    df_yr = df[df["Tahun"] == latest_year].copy()
    if df_yr.empty:
        return None

    agg = df_yr.groupby("Kategori")[col].sum().reset_index()
    agg = agg[agg["Kategori"].isin(active_progs)].sort_values(col, ascending=False)

    fig = px.bar(
        agg, x="Kategori", y=col,
        color="Kategori",
        color_discrete_sequence=colors,
        text=agg[col].apply(lambda v: f"{int(v):,}"),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        **dark,
        height=320,
        margin=dict(t=10, b=40, l=60, r=20),
        showlegend=False,
        xaxis_title="Program",
        yaxis_title=col,
    )
    return fig


def _build_pie_share(
    df: pd.DataFrame,
    active_progs: list,
    latest_year: int,
    colors: list,
    dark: dict,
) -> go.Figure | None:
    """Pie chart distribusi kasus per program."""
    df_yr = df[(df["Tahun"] == latest_year) & (df["Kategori"].isin(active_progs))].copy()
    if df_yr.empty:
        return None

    agg = df_yr.groupby("Kategori")["Kasus"].sum().reset_index()
    fig = go.Figure(go.Pie(
        labels=agg["Kategori"],
        values=agg["Kasus"],
        hole=0.38,
        marker=dict(colors=colors[:len(agg)]),
        textinfo="label+percent",
        hovertemplate="%{label}: %{value:,} kasus<extra></extra>",
    ))
    fig.update_layout(
        **dark,
        height=320,
        margin=dict(t=10, b=10, l=10, r=10),
        showlegend=True,
        legend=dict(orientation="h", y=-0.12),
    )
    return fig


def _build_yoy_delta(
    df: pd.DataFrame,
    active_progs: list,
    latest_year: int,
    colors: list,
    dark: dict,
) -> go.Figure | None:
    """Bar chart delta YoY per program."""
    prev_year = latest_year - 1
    df_cur  = df[df["Tahun"] == latest_year].groupby("Kategori")["Kasus"].sum()
    df_prev = df[df["Tahun"] == prev_year].groupby("Kategori")["Kasus"].sum()

    rows = []
    for prog in active_progs:
        cur  = df_cur.get(prog, 0)
        prev = df_prev.get(prog, 0)
        if prev > 0:
            pct = (cur - prev) / prev * 100
            rows.append({"Program": prog, "Delta (%)": round(pct, 1), "pos": pct >= 0})

    if not rows:
        return None

    agg = pd.DataFrame(rows).sort_values("Delta (%)", ascending=False)
    bar_colors = ["#22c55e" if p else "#ef4444" for p in agg["pos"]]

    fig = go.Figure(go.Bar(
        x=agg["Program"], y=agg["Delta (%)"],
        marker_color=bar_colors,
        text=agg["Delta (%)"].apply(lambda v: f"{v:+.1f}%"),
        textposition="outside",
    ))
    fig.add_hline(y=0, line_color="var(--t)", line_width=1, line_dash="dot")
    fig.update_layout(
        **dark,
        height=300,
        margin=dict(t=10, b=40, l=60, r=20),
        xaxis_title="Program",
        yaxis_title=f"Delta YoY {prev_year}→{latest_year} (%)",
        showlegend=False,
    )
    return fig


def _build_sebab_klaim_bar(
    question: str,
    latest_year: int,
    colors: list,
    dark: dict,
) -> go.Figure | None:
    """Top-10 sebab klaim dari df_raw_for_ai."""
    import re
    df_raw = _safe_get_df("df_raw_for_ai", "df_sebab_klaim")
    if df_raw is None:
        return None

    # Normalisasi kolom
    col_map = {}
    for c in df_raw.columns:
        cl = c.lower().strip()
        if "periode" in cl:     col_map[c] = "Periode"
        elif "program" in cl:   col_map[c] = "Program"
        elif "sebab" in cl:     col_map[c] = "Sebab Klaim"
        elif "kasus" in cl:     col_map[c] = "Total Kasus"
        elif "nominal" in cl:   col_map[c] = "Total Nominal"
    df = df_raw.rename(columns=col_map)

    required = {"Periode", "Program", "Sebab Klaim", "Total Kasus"}
    if not required.issubset(df.columns):
        return None

    df["Total Kasus"] = pd.to_numeric(df["Total Kasus"], errors="coerce").fillna(0)
    df["Periode"] = df["Periode"].astype(str)
    df_yr = df[df["Periode"].str.startswith(str(latest_year))]
    if df_yr.empty:
        return None

    # Filter program jika disebut di pertanyaan
    q = question.lower()
    prog_filter = next(
        (p for p in ("jkp", "jkk", "jkm", "jpn", "jht") if p in q), None
    )
    if prog_filter:
        df_yr = df_yr[df_yr["Program"].str.upper().str.contains(prog_filter.upper())]

    if df_yr.empty:
        return None

    top = (
        df_yr.groupby("Sebab Klaim")["Total Kasus"]
        .sum()
        .nlargest(10)
        .reset_index()
        .sort_values("Total Kasus")
    )

    prog_label = prog_filter.upper() if prog_filter else "Semua Program"
    fig = go.Figure(go.Bar(
        x=top["Total Kasus"],
        y=top["Sebab Klaim"],
        orientation="h",
        marker_color=colors[0],
        text=top["Total Kasus"].apply(lambda v: f"{int(v):,}"),
        textposition="outside",
    ))
    fig.update_layout(
        **dark,
        height=max(280, len(top) * 34 + 60),
        margin=dict(t=10, b=40, l=200, r=80),
        xaxis_title="Total Kasus",
        yaxis_title="",
        showlegend=False,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# MAIN BUILD FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def build_chart_data(
    question: str,
    df: pd.DataFrame,
    intent: ChartIntent,
    active_progs: list,
    years: list,
    has_nom: bool,
    latest_year: int,
    colors: list,
    dark: dict,
) -> dict | None:
    """
    Bangun chart sesuai intent. Kembalikan dict:
      {
        "type":     ChartIntent value (str),
        "fig":      go.Figure,
        "fig_json": fig.to_json(),   # untuk FastAPI
        "title":    str,
        "subtitle": str,
        "question": str,
      }
    Kembalikan None jika chart tidak bisa dibangun.
    """
    # Tentukan target (Kasus atau Nominal) dari pertanyaan
    q_low   = question.lower()
    target  = "Nominal" if ("nominal" in q_low or "rupiah" in q_low or " rp" in q_low) else "Kasus"

    fig = None
    title = ""
    subtitle = ""

    if intent == ChartIntent.TREND_LINE:
        fig = _build_trend_line(df, active_progs, years, has_nom, target, colors, dark)
        title = f"Tren {target} per Program ({min(years)}–{max(years)})"
        subtitle = "Data historis aktual"

    elif intent == ChartIntent.FORECAST_BAR:
        fig = _build_forecast_bar(df, active_progs, has_nom, target, colors, dark)
        if fig is None:
            # Fallback ke trend jika forecast belum dijalankan
            fig = _build_trend_line(df, active_progs, years, has_nom, target, colors, dark)
            title = f"Tren {target} Historis (prediksi belum dijalankan)"
            subtitle = "Jalankan model di tab Prediksi untuk chart prediksi"
        else:
            title = f"Prediksi {target} per Program"
            subtitle = "Hasil model ML — termasuk confidence interval"

    elif intent == ChartIntent.COMPARISON_BAR:
        fig = _build_comparison_bar(df, active_progs, latest_year, has_nom, target, colors, dark)
        title = f"Perbandingan {target} per Program — {latest_year}"
        subtitle = "Diurutkan dari terbesar"

    elif intent == ChartIntent.PIE_SHARE:
        fig = _build_pie_share(df, active_progs, latest_year, colors, dark)
        title = f"Distribusi Kasus per Program — {latest_year}"
        subtitle = "Proporsi kontribusi masing-masing program"

    elif intent == ChartIntent.YOY_DELTA:
        fig = _build_yoy_delta(df, active_progs, latest_year, colors, dark)
        title = f"Perubahan YoY ({latest_year-1}→{latest_year})"
        subtitle = "Hijau = naik, Merah = turun"

    elif intent == ChartIntent.SEBAB_KLAIM_BAR:
        fig = _build_sebab_klaim_bar(question, latest_year, colors, dark)
        title = f"Top Sebab Klaim — {latest_year}"
        subtitle = "Diurutkan berdasarkan jumlah kasus"

    if fig is None:
        return None

    return {
        "type":     intent.value,
        "fig":      fig,
        "fig_json": fig.to_json(),
        "title":    title,
        "subtitle": subtitle,
        "question": question,
    }



# ══════════════════════════════════════════════════════════════════════════════
# NARASI NON-TEKNIS PER CHART TYPE
# ══════════════════════════════════════════════════════════════════════════════

_CHART_NARASI = {
    "trend_line": (
        "Grafik garis ini menunjukkan perubahan jumlah klaim dari tahun ke tahun "
        "untuk setiap program BPJS. Garis naik = klaim meningkat, garis turun = klaim berkurang. "
        "Semakin curam kemiringannya, semakin besar laju perubahannya."
    ),
    "forecast_bar": (
        "Grafik batang ini menampilkan hasil prediksi model Machine Learning untuk tahun-tahun ke depan. "
        "Setiap batang = estimasi jumlah klaim per program. "
        "Area transparan di sekitar batang (jika ada) adalah rentang ketidakpastian — "
        "semakin jauh prediksinya, semakin lebar rentangnya."
    ),
    "comparison_bar": (
        "Grafik batang ini membandingkan jumlah klaim antar program BPJS untuk tahun terpilih. "
        "Batang lebih panjang = program dengan lebih banyak klaim. "
        "Gunakan ini untuk melihat program mana yang paling besar kontribusinya."
    ),
    "pie_share": (
        "Diagram lingkaran ini memperlihatkan porsi kontribusi masing-masing program "
        "terhadap total klaim. Irisan lebih besar = program lebih dominan. "
        "Persentase ditampilkan langsung pada setiap irisan."
    ),
    "yoy_delta": (
        "Grafik ini menunjukkan perubahan klaim tahun ini dibandingkan tahun lalu dalam persentase. "
        "Batang hijau = klaim naik, batang merah = klaim turun. "
        "Angka menunjukkan besar perubahannya."
    ),
    "sebab_klaim_bar": (
        "Grafik horizontal ini meranking penyebab klaim dari yang terbanyak hingga paling sedikit. "
        "Batang lebih panjang = lebih banyak peserta mengklaim dengan alasan tersebut. "
        "Gunakan ini untuk mengidentifikasi masalah yang paling sering dialami peserta."
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT RENDERER
# ══════════════════════════════════════════════════════════════════════════════

def render_ai_chart(chart_data: dict) -> None:
    """
    Render satu chart beserta narasi penjelasan non-teknis.
    Dipanggil dari tab_ai.py untuk setiap chart dalam ai_chart_history.
    """
    if chart_data is None:
        return

    fig      = chart_data.get("fig")
    title    = chart_data.get("title", "")
    subtitle = chart_data.get("subtitle", "")
    question = chart_data.get("question", "")
    ctype    = chart_data.get("type", "")

    if fig is None:
        return

    # ── Header ───────────────────────────────────────────────────────────────
    q_preview = f'<div style="font-size:.68rem;color:#86efac;margin-top:2px;">Untuk: <i>{question[:80]}{"..." if len(question)>80 else ""}</i></div>' if question else ""
    st.markdown(
        f'<div style="background:#f0fdf4;border:1px solid #bbf7d0;'
        f'border-radius:12px 12px 0 0;padding:10px 16px 8px;margin-top:14px;">'
        f'<div style="font-size:.78rem;font-weight:600;color:#166534;">📊 {title}</div>'
        f'<div style="font-size:.7rem;color:#16a34a;margin-top:2px;">{subtitle}</div>'
        f'{q_preview}</div>',
        unsafe_allow_html=True,
    )

    # ── Plot ─────────────────────────────────────────────────────────────────
    st.plotly_chart(fig, use_container_width=True)

    # ── Narasi non-teknis ─────────────────────────────────────────────────────
    narasi = _CHART_NARASI.get(ctype)
    if narasi:
        st.markdown(
            f'<div style="background:#f8fafc;border:1px solid #e2e8f0;'
            f'border-radius:0 0 12px 12px;padding:10px 16px;margin-top:-8px;">'
            f'<div style="font-size:.75rem;color:#475569;line-height:1.7;">'
            f'<b>Cara membaca grafik ini:</b> {narasi}'
            f'</div></div>',
            unsafe_allow_html=True,
        )
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)