"""
data_utils.py — Parsing, cleaning, dan merging dataset CSV/Excel.
"""
import io
import re
import numpy as np
import pandas as pd
import streamlit as st


def clean_num(val):
    s = re.sub(r'[^\d\.\-]', '', str(val).replace(',', ''))
    try:
        return float(s)
    except Exception:
        return np.nan


@st.cache_data(show_spinner=False)
def load_raw(fb: bytes, fname: str):
    try:
        df = pd.read_csv(io.BytesIO(fb)) if fname.lower().endswith('.csv') \
             else pd.read_excel(io.BytesIO(fb))
        df.columns = df.columns.str.strip().str.upper()
        return df
    except Exception:
        return None


def parse_dataset(df: pd.DataFrame, year_hint: int):
    cols = df.columns.tolist()

    # ── Program/Kategori column ───────────────────────────────────────────────
    prog_col = None
    for c in cols:
        if c in ('PROGRAM', 'KATEGORI', 'CATEGORY', 'JENIS', 'JAMINAN'):
            prog_col = c
            break
    if prog_col is None:
        for c in cols:
            if df[c].dtype == object and df[c].nunique() < 50:
                if not any(k in c for k in ('DATE', 'TANGGAL', 'KODE', 'ID', 'PERIODE',
                                             'SEBAB', 'KLAIM', 'ALASAN', 'REASON')):
                    prog_col = c
                    break
    if prog_col is None:
        return None, f"Kolom PROGRAM tidak ditemukan. Kolom: {cols}"

    # ── Kasus column — tambah TOTAL KASUS ────────────────────────────────────
    kasus_col = None
    for candidate in ('TOTAL KASUS', 'AKTUAL_KASUS', 'KASUS', 'CASE',
                      'KLAIM', 'COUNT', 'TOTAL_KASUS', 'JML KASUS',
                      'JUMLAH KASUS', 'JML_KASUS', 'JUMLAH_KASUS'):
        if candidate in cols:
            kasus_col = candidate
            break
    if kasus_col is None:
        # Fuzzy: cari kolom yang mengandung kata KASUS atau COUNT
        for c in cols:
            if 'KASUS' in c or 'COUNT' in c or 'JUMLAH' in c:
                kasus_col = c
                break
    if kasus_col is None:
        return None, f"Kolom KASUS tidak ditemukan. Kolom: {cols}"

    # ── Nominal column — tambah TOTAL NOMINAL ────────────────────────────────
    nominal_col = None
    for candidate in ('TOTAL NOMINAL', 'AKTUAL_NOMINAL', 'NOMINAL', 'AMOUNT',
                      'NILAI', 'MANFAAT', 'BEBAN', 'TOTAL_NOMINAL',
                      'TOTAL AMOUNT', 'TOTAL_AMOUNT', 'BIAYA', 'TOTAL BIAYA'):
        if candidate in cols:
            nominal_col = candidate
            break
    if nominal_col is None:
        # Fuzzy: cari kolom yang mengandung kata NOMINAL atau AMOUNT
        for c in cols:
            if 'NOMINAL' in c or 'AMOUNT' in c or 'BIAYA' in c or 'NILAI' in c:
                nominal_col = c
                break

    # ── Date column ───────────────────────────────────────────────────────────
    date_col = None
    for candidate in ('PERIODE', 'DATE', 'TANGGAL', 'BULAN', 'MONTH', 'PERIOD'):
        if candidate in cols:
            date_col = candidate
            break

    df = df.copy()
    df['Tahun'] = year_hint
    df['Bulan'] = 12

    if date_col:
        raw_date = df[date_col].astype(str).str.strip()
        # Handle format YYYYMMDD (misal: 20211231)
        yyyymmdd = raw_date.str.match(r'^\d{8}$')
        if yyyymmdd.mean() > 0.5:
            df.loc[yyyymmdd, 'Tahun'] = raw_date[yyyymmdd].str[:4].astype(int)
            df.loc[yyyymmdd, 'Bulan'] = raw_date[yyyymmdd].str[4:6].astype(int)
        else:
            ym = raw_date.str.extract(r'(\d{4})[-/]?(\d{2})', expand=True)
            has_ym = ym[0].notna() & ym[1].notna()
            if has_ym.mean() > 0.5:
                df.loc[has_ym, 'Tahun'] = ym[0][has_ym].astype(int)
                df.loc[has_ym, 'Bulan'] = ym[1][has_ym].astype(int)
            else:
                try:
                    dt = pd.to_datetime(raw_date, errors='coerce')
                    ok = dt.notna()
                    if ok.mean() > 0.5:
                        df.loc[ok, 'Tahun'] = dt[ok].dt.year.astype(int)
                        df.loc[ok, 'Bulan'] = dt[ok].dt.month.astype(int)
                except Exception:
                    pass

    tmp = pd.DataFrame()
    tmp['Kategori'] = df[prog_col].astype(str).str.strip().str.upper()
    tmp['Kasus']    = df[kasus_col].apply(clean_num)
    if nominal_col:
        tmp['Nominal'] = df[nominal_col].apply(clean_num)
    tmp['Tahun'] = df['Tahun'].astype(int)
    tmp['Bulan'] = df['Bulan'].astype(int)

    tmp = tmp[~tmp['Kategori'].str.lower().isin(['nan', 'none', ''])]
    tmp = tmp.dropna(subset=['Kasus'])
    tmp = tmp[tmp['Kasus'] >= 0]

    rows_per_group = tmp.groupby(['Tahun', 'Kategori']).size()
    is_monthly = rows_per_group.max() > 1

    if is_monthly:
        agg_dict = {'Kasus': 'sum'}
        if 'Nominal' in tmp.columns:
            agg_dict['Nominal'] = 'sum'
        yearly = tmp.groupby(['Tahun', 'Kategori'], as_index=False).agg(agg_dict)
    else:
        yearly = tmp.drop(columns=['Bulan'])

    return yearly, None


def _detect_cols_quick(df: pd.DataFrame):
    cols = df.columns.tolist()
    prog_col = kasus_col = nominal_col = date_col = None

    for c in cols:
        if c in ('PROGRAM', 'KATEGORI', 'CATEGORY', 'JENIS', 'JAMINAN') and not prog_col:
            prog_col = c
        # Kasus: cek exact dulu, lalu fuzzy
        if not kasus_col:
            if c in ('TOTAL KASUS', 'AKTUAL_KASUS', 'KASUS', 'CASE', 'KLAIM',
                     'TOTAL_KASUS', 'JUMLAH KASUS', 'JML KASUS'):
                kasus_col = c
            elif 'KASUS' in c or 'COUNT' in c:
                kasus_col = c
        # Nominal: cek exact dulu, lalu fuzzy
        if not nominal_col:
            if c in ('TOTAL NOMINAL', 'AKTUAL_NOMINAL', 'NOMINAL', 'AMOUNT',
                     'NILAI', 'MANFAAT', 'BEBAN', 'TOTAL_NOMINAL'):
                nominal_col = c
            elif 'NOMINAL' in c or 'AMOUNT' in c or 'BIAYA' in c:
                nominal_col = c
        if c in ('DATE', 'PERIODE', 'TANGGAL', 'BULAN', 'MONTH') and not date_col:
            date_col = c

    if prog_col and kasus_col:
        return {'prog': prog_col, 'kasus': kasus_col, 'nominal': nominal_col, 'date': date_col}
    return None


def _build_raw_monthly(df: pd.DataFrame, year_hint: int, m: dict):
    try:
        tmp = pd.DataFrame()
        tmp['Kategori'] = df[m['prog']].astype(str).str.strip().str.upper()
        tmp['Kasus']    = df[m['kasus']].apply(clean_num)
        if m['nominal']:
            tmp['Nominal'] = df[m['nominal']].apply(clean_num)
        tmp['Tahun'] = year_hint
        tmp['Bulan'] = 12
        if m['date']:
            raw_d = df[m['date']].astype(str).str.strip()
            # Handle YYYYMMDD format
            yyyymmdd = raw_d.str.match(r'^\d{8}$')
            if yyyymmdd.mean() > 0.5:
                tmp.loc[yyyymmdd, 'Tahun'] = raw_d[yyyymmdd].str[:4].astype(int)
                tmp.loc[yyyymmdd, 'Bulan'] = raw_d[yyyymmdd].str[4:6].astype(int)
            else:
                ym = raw_d.str.extract(r'(\d{4})[-/]?(\d{2})', expand=True)
                ok = ym[0].notna() & ym[1].notna()
                if ok.mean() > 0.5:
                    tmp.loc[ok, 'Tahun'] = ym[0][ok].astype(int)
                    tmp.loc[ok, 'Bulan'] = ym[1][ok].astype(int)
        tmp = tmp[~tmp['Kategori'].str.lower().isin(['nan', 'none', ''])]
        tmp = tmp.dropna(subset=['Kasus'])
        tmp = tmp[tmp['Kasus'] >= 0]
        tmp['Tahun'] = tmp['Tahun'].astype(int)
        tmp['Bulan'] = tmp['Bulan'].astype(int)
        return tmp
    except Exception:
        return None


def merge_all(files_info: list):
    all_dfs, errors = [], []
    for year_hint, raw, fname in files_info:
        parsed, err = parse_dataset(raw, year_hint)
        if err:
            errors.append(f"⚠️ {fname}: {err}")
        elif parsed is not None and len(parsed) > 0:
            all_dfs.append(parsed)
    if not all_dfs:
        return None, errors
    combined = pd.concat(all_dfs, ignore_index=True)
    agg = {'Kasus': 'last'}
    if 'Nominal' in combined.columns:
        agg['Nominal'] = 'last'
    combined = combined.groupby(['Tahun', 'Kategori'], as_index=False).agg(agg)
    return combined, errors


def analyze_program_changes(df: pd.DataFrame) -> dict:
    years = sorted(df['Tahun'].unique())
    changes = {}
    for i in range(len(years) - 1):
        y0 = years[i]
        y1 = years[i + 1]
        p0 = set(df[df['Tahun'] == y0]['Kategori'].unique())
        p1 = set(df[df['Tahun'] == y1]['Kategori'].unique())
        changes[(y0, y1)] = {
            'added':   sorted(p1 - p0),
            'removed': sorted(p0 - p1),
            'stable':  sorted(p0 & p1),
        }
    return changes


def get_active_programs(df: pd.DataFrame) -> list:
    latest = df['Tahun'].max()
    return sorted(df[df['Tahun'] == latest]['Kategori'].unique())