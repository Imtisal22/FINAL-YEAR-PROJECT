"""
Dataset cleaning module for IMTISAL maize price prediction pipeline.

For each raw file:
  1. Load (CSV or Excel)
  2. Standardise column names
  3. Parse / normalise date column → YYYY-MM-DD
  4. Remove duplicates
  5. Handle missing values (numeric median, categorical mode)
  6. Detect & cap outliers (IQR method)
  7. Standardise units (price → NGN/100 kg bag, yield → kg/ha, rain → mm)
  8. Save cleaned file to CLEAN_DIR
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

from config import (
    RAW_DIR, CLEAN_DIR,
    DATE_COLUMN_VARIANTS, DATASET_KEYWORDS, OUTLIER_IQR_MULTIPLIER,
)

warnings.filterwarnings("ignore")


# ── helpers ───────────────────────────────────────────────────────────────────

def _snake(col: str) -> str:
    """'Total Rainfall (mm)' → 'total_rainfall_mm'"""
    col = str(col).strip()
    col = re.sub(r"[^\w\s]", "", col)   # strip punctuation
    col = re.sub(r"\s+", "_", col)      # spaces → _
    col = re.sub(r"_+", "_", col)       # collapse __
    return col.lower().strip("_")


def _detect_dataset_type(filepath: str) -> str:
    name = os.path.basename(filepath).lower()
    for kw, dtype in DATASET_KEYWORDS.items():
        if kw in name:
            return dtype
    return "generic"


def _load(filepath: str) -> pd.DataFrame:
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".csv":
        # try utf-8, fall back to latin-1
        try:
            df = pd.read_csv(filepath, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(filepath, encoding="latin-1")
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(filepath, engine="openpyxl")
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return df


# ── date parsing ──────────────────────────────────────────────────────────────

COMMON_DATE_FORMATS = [
    "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y",
    "%d-%m-%Y", "%m-%d-%Y", "%Y/%m/%d",
    "%B %Y",    "%b %Y",    "%B-%Y",
    "%Y",       "%b-%y",    "%d %B %Y",
]


def _parse_dates(series: pd.Series) -> pd.Series:
    """
    Try multiple date formats; fall back to pandas inference.
    Year-only values like 2015 → 2015-01-01.
    Month-Year like 'Jan 2015' → 2015-01-01.
    """
    s = series.astype(str).str.strip()

    # Year only (e.g. "2015", "2015.0")
    year_mask = s.str.match(r"^\d{4}(\.0)?$")
    if year_mask.all():
        return pd.to_datetime(s.str[:4] + "-01-01", format="%Y-%m-%d", errors="coerce")

    parsed = None
    for fmt in COMMON_DATE_FORMATS:
        try:
            parsed = pd.to_datetime(s, format=fmt, errors="coerce")
            if parsed.notna().mean() > 0.7:   # >70 % parsed → good enough
                break
        except Exception:
            pass

    if parsed is None or parsed.notna().mean() < 0.5:
        parsed = pd.to_datetime(s, infer_datetime_format=True, errors="coerce")

    return parsed


def _find_date_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if col in DATE_COLUMN_VARIANTS:
            return col
    return None


# ── outlier handling ──────────────────────────────────────────────────────────

def _cap_outliers(df: pd.DataFrame, multiplier: float = OUTLIER_IQR_MULTIPLIER) -> tuple[pd.DataFrame, dict]:
    report = {}
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # skip year / id-like columns
    skip_if = {"year", "month", "day", "id"}
    num_cols = [c for c in num_cols if c not in skip_if]

    for col in num_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lo  = q1 - multiplier * iqr
        hi  = q3 + multiplier * iqr
        n_out = ((df[col] < lo) | (df[col] > hi)).sum()
        if n_out > 0:
            df[col] = df[col].clip(lower=lo, upper=hi)
            report[col] = {"capped": int(n_out), "lower": round(lo, 4), "upper": round(hi, 4)}
    return df, report


# ── unit standardisation ──────────────────────────────────────────────────────

def _standardise_units(df: pd.DataFrame, dtype: str) -> pd.DataFrame:
    """
    Best-effort unit normalisation based on column name heuristics.
    All prices expected in NGN; convert per-tonne → per 100-kg bag (/10).
    Rainfall: no conversion needed if already in mm.
    Temperature: no conversion (assume Celsius).
    """
    for col in df.select_dtypes(include=[np.number]).columns:
        # price per tonne → per 100 kg bag  (÷ 10)
        if "price" in col and "tonne" in col:
            df[col] = df[col] / 10
            df.rename(columns={col: col.replace("tonne", "100kg_bag")}, inplace=True)
        # yield in tonnes/ha → kg/ha  (× 1000)
        if "yield" in col and "tonne" in col:
            df[col] = df[col] * 1000
            df.rename(columns={col: col.replace("tonne", "kg")}, inplace=True)
    return df


# ── main cleaning function ────────────────────────────────────────────────────

def clean_file(filepath: str) -> tuple[pd.DataFrame, dict]:
    """
    Clean a single file. Returns (cleaned_df, cleaning_report).
    Saves cleaned file to CLEAN_DIR.
    """
    fname  = os.path.basename(filepath)
    dtype  = _detect_dataset_type(filepath)
    report = {"file": fname, "type": dtype, "steps": {}}

    print(f"\n[clean] ── {fname} ({dtype}) ──")

    # 1. Load
    df = _load(filepath)
    report["steps"]["raw_shape"] = list(df.shape)
    print(f"        loaded: {df.shape[0]} rows × {df.shape[1]} cols")

    # 2. Drop fully-empty rows/cols
    df.dropna(how="all", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)

    # 3. Standardise column names
    df.columns = [_snake(c) for c in df.columns]
    report["steps"]["columns"] = list(df.columns)

    # 4. Find & parse date column
    date_col = _find_date_column(df)
    if date_col:
        if date_col != "date":
            df.rename(columns={date_col: "date"}, inplace=True)
        df["date"] = _parse_dates(df["date"])
        n_bad = df["date"].isna().sum()
        report["steps"]["date_parse_errors"] = int(n_bad)
        if n_bad > 0:
            print(f"        [warn] {n_bad} dates could not be parsed → dropped")
            df.dropna(subset=["date"], inplace=True)
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        print(f"        date column parsed: {df['date'].iloc[0]} … {df['date'].iloc[-1]}")
    else:
        print("        [warn] no date column detected — skipping date parsing")

    # 5. Remove duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    n_dup = before - len(df)
    report["steps"]["duplicates_removed"] = n_dup
    if n_dup:
        print(f"        duplicates removed: {n_dup}")

    # 6. Fill missing values
    mv_report = {}
    for col in df.columns:
        n_miss = df[col].isna().sum()
        if n_miss == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            fill_val = df[col].median()
            df[col].fillna(fill_val, inplace=True)
            mv_report[col] = {"missing": int(n_miss), "fill": "median", "value": round(float(fill_val), 4)}
        else:
            fill_val = df[col].mode().iloc[0] if not df[col].mode().empty else "unknown"
            df[col].fillna(fill_val, inplace=True)
            mv_report[col] = {"missing": int(n_miss), "fill": "mode", "value": str(fill_val)}
    report["steps"]["missing_values"] = mv_report
    total_mv = sum(v["missing"] for v in mv_report.values())
    if total_mv:
        print(f"        missing values filled: {total_mv} cells across {len(mv_report)} columns")

    # 7. Outlier capping
    df, out_report = _cap_outliers(df)
    report["steps"]["outliers"] = out_report
    if out_report:
        print(f"        outliers capped in: {list(out_report.keys())}")

    # 8. Unit standardisation
    df = _standardise_units(df, dtype)

    # 9. Add dataset-type tag (helps during merge)
    df["dataset_source"] = dtype

    # 10. Save
    clean_path = os.path.join(CLEAN_DIR, fname.rsplit(".", 1)[0] + "_clean.csv")
    df.to_csv(clean_path, index=False)
    report["steps"]["clean_shape"] = list(df.shape)
    report["clean_path"] = clean_path
    print(f"        saved → {clean_path}  ({df.shape[0]} rows × {df.shape[1]} cols)")

    return df, report


def clean_all(filepaths: list[str]) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    """Clean all downloaded files. Returns {filename: df} and list of reports."""
    dfs, reports = {}, []
    for fp in filepaths:
        try:
            df, rpt = clean_file(fp)
            dfs[os.path.basename(fp)] = df
            reports.append(rpt)
        except Exception as exc:
            print(f"[ERROR] cleaning {fp}: {exc}")
    return dfs, reports
