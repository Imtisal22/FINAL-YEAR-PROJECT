"""
IMTISAL Maize Price Prediction – Local Pipeline (no Google Drive needed)
=========================================================================
Reads all datasets from Desktop/imtisal_project, cleans, merges, and
produces:
  • data/output/maize_master.csv
  • reports/eda_report.html

Runs with:  ~/imtisal_venv/Scripts/python run_pipeline.py
"""

# ── stdlib ─────────────────────────────────────────────────────────────────
import csv, io, os, re, sys, json, math, statistics, datetime, warnings, base64
from collections import defaultdict

# ── third-party (venv) ─────────────────────────────────────────────────────
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
import openpyxl

# ── paths ──────────────────────────────────────────────────────────────────
SRC_DIR   = "C:/Users/USER/Desktop/imtisal_project"
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
RAW_DIR   = os.path.join(BASE_DIR, "data", "raw")
CLEAN_DIR = os.path.join(BASE_DIR, "data", "cleaned")
OUT_DIR   = os.path.join(BASE_DIR, "data", "output")
RPT_DIR   = os.path.join(BASE_DIR, "reports")
for d in (RAW_DIR, CLEAN_DIR, OUT_DIR, RPT_DIR):
    os.makedirs(d, exist_ok=True)

MASTER_CSV = os.path.join(OUT_DIR,  "maize_master.csv")
EDA_HTML   = os.path.join(RPT_DIR,  "eda_report.html")

plt.style.use("seaborn-v0_8-whitegrid")
PALETTE = ["#4878cf","#6acc65","#d65f5f","#b47cc7","#c4ad66","#77bedb",
           "#e78ac3","#a6d854","#ffd92f","#e5c494","#b3b3b3","#fc8d62"]

warnings.filterwarnings("ignore")

# ===============================================================================
# HELPERS
# ===============================================================================

def snake(s):
    s = str(s).strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.lower().strip("_")


def to_float(v):
    try:
        f = float(str(v).replace(",", "").strip())
        return None if math.isnan(f) or f == -999 else f
    except Exception:
        return None


def median_val(lst):
    nums = [x for x in lst if x is not None]
    return statistics.median(nums) if nums else None


def mode_val(lst):
    clean = [x for x in lst if x and str(x).strip() not in ("", "nan", "None")]
    if not clean:
        return ""
    counts = defaultdict(int)
    for v in clean:
        counts[v] += 1
    return max(counts, key=counts.get)


# ── date parsers ──────────────────────────────────────────────────────────────
_MONTH_MAP = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
              "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12}

def parse_date(s):
    """Return (yyyy, mm) tuple or None."""
    s = str(s).strip()
    if not s or s.lower() in ("nan", "none", ""):
        return None
    # YYYY-MM-DD or YYYY/MM/DD
    m = re.match(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})$", s)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    # DD/MM/YYYY or DD-MM-YYYY
    m = re.match(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{4})$", s)
    if m:
        return (int(m.group(3)), int(m.group(2)))
    # YYYY only
    if re.match(r"^\d{4}$", s):
        return (int(s), 1)
    # Jan-17 or Jan 2017
    m = re.match(r"^([A-Za-z]{3})[-\s](\d{2,4})$", s)
    if m:
        mo = _MONTH_MAP.get(m.group(1).lower())
        yr = int(m.group(2))
        if yr < 100:
            yr += 2000 if yr < 50 else 1900
        if mo:
            return (yr, mo)
    return None


def ym_to_date(yr, mo):
    return f"{yr:04d}-{mo:02d}-01"


# ── outlier capping ───────────────────────────────────────────────────────────
def cap_outliers(values, mult=2.5):
    nums = sorted(x for x in values if x is not None)
    if len(nums) < 4:
        return values
    q1 = np.percentile(nums, 25)
    q3 = np.percentile(nums, 75)
    iqr = q3 - q1
    lo, hi = q1 - mult * iqr, q3 + mult * iqr
    return [max(lo, min(hi, v)) if v is not None else None for v in values]


# ── EDA helpers ───────────────────────────────────────────────────────────────
def fig_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def img(b64, cap=""):
    return f'<figure><img src="data:image/png;base64,{b64}" style="max-width:100%"/>' \
           f'<figcaption>{cap}</figcaption></figure>'


def section(title, body):
    return f'<section><h2>{title}</h2>{body}</section>\n'


# ===============================================================================
# DATASET 1 – FEWS NET Price Data
#   Columns: country,market,admin_1,…,period_date,price_type,product,unit,value
#   Filter: admin_1 == Adamawa, product contains maize
#   Aggregate: monthly mean price per kg, NGN
# ===============================================================================

def load_fews():
    path = os.path.join(SRC_DIR, "FEWS_NET_Staple_Food_Price_Data.csv.xls")
    print("[FEWS] loading …")
    monthly = defaultdict(list)   # (yr,mo) -> list of prices
    markets = defaultdict(set)

    with open(path, encoding="utf-8-sig", errors="replace") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("country", "").strip().lower() != "nigeria":
                continue
            adm = row.get("admin_1", "").strip().lower()
            if "adamawa" not in adm:
                continue
            prod = row.get("product", "").strip().lower()
            if "maize" not in prod:
                continue
            ym = parse_date(row.get("period_date", ""))
            if not ym:
                continue
            price = to_float(row.get("value"))
            if price is None or price <= 0:
                continue
            unit = row.get("unit", "").strip().lower()
            # Normalise to NGN/kg
            if unit in ("100kg", "100 kg", "bag"):
                price /= 100
            elif unit in ("tonne", "mt"):
                price /= 1000
            monthly[ym].append(price)
            markets[ym].add(row.get("market", "").strip())

    rows = []
    for ym, prices in sorted(monthly.items()):
        rows.append({
            "date": ym_to_date(*ym),
            "year": ym[0], "month": ym[1],
            "fews_maize_price_ngn_kg": round(statistics.mean(prices), 4),
            "fews_n_markets": len(markets[ym]),
        })

    # Cap outliers
    prices_col = [r["fews_maize_price_ngn_kg"] for r in rows]
    prices_col = cap_outliers(prices_col)
    for r, p in zip(rows, prices_col):
        r["fews_maize_price_ngn_kg"] = round(p, 4)

    print(f"       {len(rows)} monthly records  "
          f"({rows[0]['date'] if rows else '?'} -> {rows[-1]['date'] if rows else '?'})")
    return rows


# ===============================================================================
# DATASET 2 – NFPT Complete Dataset
#   Columns: Date,State,LGA,Outlet Type,Country,Sector,Food Item,Price Category,UPRICE
#   Filter: State == ADAMAWA, Food Item contains maize
#   Aggregate: monthly mean price, NGN (assume per kg)
# ===============================================================================

def load_nfpt():
    path = os.path.join(SRC_DIR, "NFPT Complete Dataset (4).csv.xls")
    print("[NFPT] loading …")
    monthly = defaultdict(list)
    sectors = defaultdict(set)

    with open(path, encoding="utf-8-sig", errors="replace") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            state = row.get("State", "").strip().upper()
            if "ADAMAWA" not in state:
                continue
            item = row.get("Food Item", "").strip().lower()
            if "maize" not in item:
                continue
            ym = parse_date(row.get("Date", ""))
            if not ym:
                continue
            price = to_float(row.get("UPRICE"))
            if price is None or price <= 0:
                continue
            monthly[ym].append(price)
            sectors[ym].add(row.get("Sector", "").strip())

    rows = []
    for ym, prices in sorted(monthly.items()):
        rows.append({
            "date": ym_to_date(*ym),
            "year": ym[0], "month": ym[1],
            "nfpt_maize_price_ngn": round(statistics.mean(prices), 4),
            "nfpt_n_observations": len(prices),
            "nfpt_price_std": round(statistics.stdev(prices) if len(prices) > 1 else 0, 4),
        })

    prices_col = [r["nfpt_maize_price_ngn"] for r in rows]
    prices_col = cap_outliers(prices_col)
    for r, p in zip(rows, prices_col):
        r["nfpt_maize_price_ngn"] = round(p, 4)

    print(f"       {len(rows)} monthly records  "
          f"({rows[0]['date'] if rows else '?'} -> {rows[-1]['date'] if rows else '?'})")
    return rows


# ===============================================================================
# DATASET 3 – NASA POWER Monthly (Cloud Amount)
#   Format: PARAMETER, YEAR, JAN…DEC, ANN  (after -END HEADER- line)
# ===============================================================================

def load_power_monthly():
    path = os.path.join(SRC_DIR,
        "POWER_Point_Monthly_20150101_20251231_009d34N_012d47E_UTC.csv.xls")
    print("[POWER Monthly] loading …")

    with open(path, encoding="utf-8-sig", errors="replace") as fh:
        lines = fh.readlines()

    # Find data section (after -END HEADER-)
    start = 0
    for i, l in enumerate(lines):
        if "-END HEADER-" in l:
            start = i + 1
            break

    data_lines = [l.strip() for l in lines[start:] if l.strip()]
    if not data_lines:
        print("       WARNING: no data found")
        return []

    header = [c.strip() for c in data_lines[0].split(",")]
    month_names = ["JAN","FEB","MAR","APR","MAY","JUN",
                   "JUL","AUG","SEP","OCT","NOV","DEC"]

    rows = []
    for line in data_lines[1:]:
        parts = [c.strip() for c in line.split(",")]
        if len(parts) < len(header):
            continue
        row_dict = dict(zip(header, parts))
        param = snake(row_dict.get("PARAMETER", ""))
        year  = to_float(row_dict.get("YEAR"))
        if year is None:
            continue
        yr = int(year)
        for mo_idx, mo_name in enumerate(month_names, 1):
            val = to_float(row_dict.get(mo_name))
            rows.append({
                "date": ym_to_date(yr, mo_idx),
                "year": yr, "month": mo_idx,
                f"climate_{param}": val,
            })

    print(f"       {len(rows)} monthly records")
    return rows


# ===============================================================================
# DATASET 4 – NASA POWER Climatic Design Conditions (xlsx)
#   Single row of design statistics; extract monthly averages for:
#   avg_temp, precip_avg, wind_speed.  Broadcast across all months.
# ===============================================================================

def load_power_design():
    path = os.path.join(SRC_DIR,
        "POWER_Climatic_Design_Conditions_20160101_20241231_009d34N_012d47E.xlsx")
    print("[POWER Design] loading …")

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb["Main"]
    all_rows = [list(r) for r in ws.iter_rows(values_only=True)]

    # Row 0: group headers (multi-level), Row 2: sub-headers (Jan,Feb…), Row 3: data
    # We need the month-indexed columns for: Avg Daily Temp, Avg Precipitation, Avg Wind Speed
    # These appear as repeated "Annual,Jan,Feb…Dec" blocks in row 2
    sub_header = all_rows[2] if len(all_rows) > 2 else []
    data_row   = all_rows[3] if len(all_rows) > 3 else []
    grp_header = all_rows[0] if len(all_rows) > 0 else []

    # Build a flat column index: (group_label_idx, sub_label) -> col_idx
    # We look for the "Average Daily Temperature" and "Average Precipitation" groups
    month_abbr = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    # Identify column indices for each target metric (by scanning group header for keywords)
    metrics = {
        "design_avg_temp_c":  "Average Daily Dry Bulb Temperature",
        "design_avg_precip_mm": "Average Precipitation",
        "design_avg_wind_ms":   "Average Wind Speed",
    }

    monthly_climate = {}   # metric -> {month: value}
    for metric_key, search_str in metrics.items():
        # Find start column of this group
        start_col = None
        for ci, cell in enumerate(grp_header):
            if cell and search_str.lower()[:20] in str(cell).lower():
                start_col = ci
                break
        if start_col is None:
            continue
        # Columns start_col+1 through start_col+12 are Jan…Dec
        monthly_climate[metric_key] = {}
        for mo_idx in range(1, 13):
            ci = start_col + mo_idx  # +1 for Annual, then Jan=+1, Feb=+2…
            if ci < len(data_row):
                val = to_float(data_row[ci])
                monthly_climate[metric_key][mo_idx] = val

    # Expand into one row per month (static — same value regardless of year)
    rows = []
    for mo_idx in range(1, 13):
        row = {"month": mo_idx}
        for key, mo_dict in monthly_climate.items():
            row[key] = mo_dict.get(mo_idx)
        rows.append(row)

    print(f"       {len(rows)} monthly design-condition records extracted")
    return rows   # keyed by month (1-12), broadcast in merge


# ===============================================================================
# DATASET 5 – Temperatures, Degree-Days Table (CSV)
#   Rows: DBAvg, DBStd, CDD18.3 etc.; Columns: Annual, Jan…Dec
#   Extract DBAvg (avg dry bulb), CDD18.3, CDH26.7 by month -> broadcast
# ===============================================================================

def load_temperature_table():
    path = os.path.join(SRC_DIR,
        "Temperatures, Degree-Days and Degree-Hours Table.csv.xls")
    print("[Temp Table] loading …")

    with open(path, encoding="utf-8-sig", errors="replace") as fh:
        rows = list(csv.reader(fh))

    if not rows:
        return []

    header = rows[0]  # ['', 'Annual', 'Jan', 'Feb', ...]
    month_abbr = ["Jan","Feb","Mar","Apr","May","Jun",
                  "Jul","Aug","Sep","Oct","Nov","Dec"]
    mo_cols = {m: header.index(m) for m in month_abbr if m in header}

    metrics = {}
    for row in rows[1:]:
        if not row:
            continue
        label = row[0].strip()
        # Extract label: "DBAvg (C)" -> "temp_db_avg_c"
        key = snake(label).replace("_c_d_", "_cd_")
        metrics[key] = {}
        for m_abbr, ci in mo_cols.items():
            metrics[key][_MONTH_MAP[m_abbr.lower()]] = to_float(row[ci]) if ci < len(row) else None

    # Keep only the most useful columns
    keep = [k for k in metrics if any(x in k for x in
            ["dbavg", "dbstd", "cdd18", "cdd10", "cdh23", "cdh26"])]

    out_rows = []
    for mo in range(1, 13):
        row = {"month": mo}
        for k in keep:
            row[f"temp_{k}"] = metrics[k].get(mo)
        out_rows.append(row)

    print(f"       {len(out_rows)} monthly static records  ({len(keep)} metrics)")
    return out_rows


# ===============================================================================
# MERGE
# ===============================================================================

def merge_all(fews, nfpt, power_monthly, design_monthly, temp_monthly):
    """
    Merge all datasets into one monthly time-series DataFrame-equivalent.
    Returns list-of-dicts sorted by date.
    """
    print("\n[MERGE] building monthly spine …")

    # 1. Build date spine from price data date range
    all_dates = set()
    for r in fews:  all_dates.add((r["year"], r["month"]))
    for r in nfpt:  all_dates.add((r["year"], r["month"]))
    for r in power_monthly: all_dates.add((r["year"], r["month"]))

    spine = sorted(all_dates)
    print(f"        date spine: {ym_to_date(*spine[0])} -> {ym_to_date(*spine[-1])}  ({len(spine)} months)")

    # 2. Index each dataset by (year, month)
    def idx_by_ym(rows, key_fields=("year","month")):
        d = {}
        for r in rows:
            k = (r.get("year"), r.get("month"))
            d[k] = r
        return d

    fews_idx   = idx_by_ym(fews)
    nfpt_idx   = idx_by_ym(nfpt)
    power_idx  = idx_by_ym(power_monthly)
    design_idx = {r["month"]: r for r in design_monthly}  # keyed by month only
    temp_idx   = {r["month"]: r for r in temp_monthly}

    # 3. Merge
    master = []
    for ym in spine:
        yr, mo = ym
        row = {"date": ym_to_date(yr, mo), "year": yr, "month": mo,
               "quarter": (mo - 1) // 3 + 1}

        # price data
        for key, val in fews_idx.get(ym, {}).items():
            if key not in ("date","year","month"):
                row[key] = val
        for key, val in nfpt_idx.get(ym, {}).items():
            if key not in ("date","year","month"):
                row[key] = val

        # NASA POWER monthly climate
        for key, val in power_idx.get(ym, {}).items():
            if key not in ("date","year","month"):
                row[key] = val

        # Static monthly climate (by month number)
        for key, val in design_idx.get(mo, {}).items():
            if key != "month":
                row[key] = val
        for key, val in temp_idx.get(mo, {}).items():
            if key != "month":
                row[key] = val

        master.append(row)

    # 4. Fill missing numerics with column median
    all_keys = list(dict.fromkeys(k for r in master for k in r))
    numeric_keys = []
    for k in all_keys:
        vals = [r.get(k) for r in master if r.get(k) is not None]
        if vals and all(isinstance(v, (int, float)) for v in vals[:5]):
            numeric_keys.append(k)

    for k in numeric_keys:
        col = [r.get(k) for r in master]
        med = median_val(col)
        if med is not None:
            for r in master:
                if r.get(k) is None:
                    r[k] = med

    print(f"        master shape: {len(master)} rows x {len(all_keys)} columns")
    missing_total = sum(1 for r in master for v in r.values() if v is None)
    print(f"        remaining missing cells: {missing_total}")
    return master


# ===============================================================================
# WRITE MASTER CSV
# ===============================================================================

def write_master(master):
    all_keys = list(dict.fromkeys(k for r in master for k in r))
    with open(MASTER_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(master)
    print(f"\n[OUTPUT] master CSV -> {MASTER_CSV}  ({len(master)} rows x {len(all_keys)} cols)")
    return all_keys


# ===============================================================================
# SAVE CLEANED INDIVIDUAL DATASETS
# ===============================================================================

def save_cleaned(name, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    path = os.path.join(CLEAN_DIR, f"{name}_clean.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  saved clean/{name}_clean.csv  ({len(rows)} rows)")


# ===============================================================================
# EDA REPORT
# ===============================================================================

def eda_report(master, col_keys):
    print("\n[EDA] generating report …")
    dates  = [r["date"] for r in master]
    parsed = [datetime.datetime.strptime(d, "%Y-%m-%d") for d in dates]
    num_keys = [k for k in col_keys if k not in ("date",)
                and all(isinstance(r.get(k), (int, float, type(None))) for r in master[:10])
                and any(r.get(k) is not None for r in master)]
    skip_for_plots = {"year","month","quarter","fews_n_markets","nfpt_n_observations"}
    plot_keys = [k for k in num_keys if k not in skip_for_plots]

    # ── numeric array ──
    def col_arr(k):
        return np.array([r.get(k) if r.get(k) is not None else np.nan for r in master],
                        dtype=float)

    # ── 1. Overview table ──
    n_miss = sum(1 for r in master for v in r.values() if v is None)
    n_dup  = len(master) - len({r["date"] for r in master})
    overview = {
        "Total rows": len(master),
        "Total columns": len(col_keys),
        "Numeric columns": len(num_keys),
        "Date range": f"{dates[0]}  ->  {dates[-1]}",
        "Missing cells": n_miss,
        "Duplicate dates": n_dup,
        "Data sources": "FEWS NET, NFPT, NASA POWER, Design Conditions, Temp Table",
    }
    ov_html = "<table class='tbl'>" + "".join(
        f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in overview.items()
    ) + "</table>"

    # Descriptive stats
    desc_rows = []
    for k in plot_keys:
        arr = col_arr(k)
        arr_v = arr[~np.isnan(arr)]
        if len(arr_v) == 0:
            continue
        desc_rows.append({
            "Column": k, "Count": len(arr_v),
            "Mean": f"{arr_v.mean():.2f}", "Std": f"{arr_v.std():.2f}",
            "Min": f"{arr_v.min():.2f}", "25%": f"{np.percentile(arr_v,25):.2f}",
            "50%": f"{np.percentile(arr_v,50):.2f}", "75%": f"{np.percentile(arr_v,75):.2f}",
            "Max": f"{arr_v.max():.2f}",
            "Missing": int(np.isnan(arr).sum()),
        })
    desc_html = "<table class='tbl'><tr>" + "".join(
        f"<th>{h}</th>" for h in ["Column","Count","Mean","Std","Min","25%","50%","75%","Max","Missing"]
    ) + "</tr>"
    for dr in desc_rows:
        desc_html += "<tr>" + "".join(f"<td>{dr[h]}</td>" for h in
            ["Column","Count","Mean","Std","Min","25%","50%","75%","Max","Missing"]) + "</tr>"
    desc_html += "</table>"

    # ── 2. Missing data bar chart ──
    miss_fracs = {}
    for k in col_keys:
        frac = sum(1 for r in master if r.get(k) is None) / len(master)
        if frac > 0:
            miss_fracs[k] = frac
    if miss_fracs:
        mk = sorted(miss_fracs, key=miss_fracs.get, reverse=True)
        fig, ax = plt.subplots(figsize=(9, max(3, len(mk) * 0.35)))
        ax.barh(mk, [miss_fracs[k] for k in mk], color=PALETTE[0])
        ax.set_xlabel("Missing fraction"); ax.set_title("Missing Data by Column")
        for bar, v in zip(ax.patches, [miss_fracs[k] for k in mk]):
            ax.text(v + 0.005, bar.get_y() + bar.get_height()/2,
                    f"{v:.0%}", va="center", fontsize=8)
        miss_plot = img(fig_b64(fig), "Proportion of missing values per column")
    else:
        miss_plot = "<p>No missing values in master dataset.</p>"

    # ── 3. Price time-series ──
    price_keys = [k for k in plot_keys if "price" in k]
    if price_keys:
        fig, axes = plt.subplots(len(price_keys), 1,
                                  figsize=(11, 3 * len(price_keys)), sharex=True)
        if len(price_keys) == 1:
            axes = [axes]
        for ax, k in zip(axes, price_keys):
            arr = col_arr(k)
            valid = ~np.isnan(arr)
            ax.plot(np.array(parsed)[valid], arr[valid], color=PALETTE[0], lw=1.5)
            ax.set_ylabel("NGN"); ax.set_title(k.replace("_", " ").title())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.xaxis.set_major_locator(mdates.YearLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        fig.suptitle("Maize Price Time-Series – Adamawa State", fontsize=12, y=1.02)
        fig.tight_layout()
        ts_plot = img(fig_b64(fig), "Monthly maize price trends from all data sources")
    else:
        ts_plot = "<p>No price columns found.</p>"

    # ── 4. Seasonal boxplots (price by month) ──
    month_abbr = ["Jan","Feb","Mar","Apr","May","Jun",
                  "Jul","Aug","Sep","Oct","Nov","Dec"]
    if price_keys:
        fig, axes = plt.subplots(1, min(2, len(price_keys)),
                                  figsize=(6 * min(2, len(price_keys)), 4))
        if not hasattr(axes, "__len__"):
            axes = [axes]
        for ax, k in zip(axes, price_keys[:2]):
            by_month = [[] for _ in range(12)]
            for r in master:
                mo = r.get("month")
                v  = r.get(k)
                if mo and v is not None:
                    by_month[int(mo) - 1].append(v)
            bp = ax.boxplot(by_month, labels=month_abbr, patch_artist=True, notch=False)
            for i, patch in enumerate(bp["boxes"]):
                patch.set_facecolor(PALETTE[i % len(PALETTE)])
                patch.set_alpha(0.75)
            ax.set_title(k.replace("_", " ").title(), fontsize=9)
            ax.set_xlabel("Month")
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        fig.suptitle("Seasonal Maize Price Distribution", fontsize=12)
        fig.tight_layout()
        seasonal_plot = img(fig_b64(fig), "Seasonal price boxplots by calendar month")
    else:
        seasonal_plot = ""

    # ── 5. Correlation heatmap ──
    corr_keys = [k for k in plot_keys if not k.startswith("nfpt_n") and not k.startswith("fews_n")]
    if len(corr_keys) >= 2:
        mat = np.array([col_arr(k) for k in corr_keys])   # shape (n_cols, n_rows)
        # compute correlation ignoring NaN pairs
        n = len(corr_keys)
        corr = np.full((n, n), np.nan)
        for i in range(n):
            for j in range(n):
                x, y = mat[i], mat[j]
                mask = ~(np.isnan(x) | np.isnan(y))
                if mask.sum() > 2:
                    corr[i, j] = np.corrcoef(x[mask], y[mask])[0, 1]
        fig, ax = plt.subplots(figsize=(max(7, n * 0.7), max(5, n * 0.6)))
        mask_upper = np.triu(np.ones((n, n), dtype=bool))
        # mask upper triangle
        corr_plot_mat = np.where(mask_upper, np.nan, corr)
        cmap = cm.RdYlGn
        im = ax.imshow(corr_plot_mat, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(corr_keys, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(corr_keys, fontsize=7)
        if n <= 15:
            for i2 in range(n):
                for j2 in range(n):
                    if not mask_upper[i2, j2] and not np.isnan(corr[i2, j2]):
                        ax.text(j2, i2, f"{corr[i2,j2]:.2f}", ha="center", va="center",
                                fontsize=6, color="black")
        plt.colorbar(im, ax=ax)
        ax.set_title("Pearson Correlation Matrix")
        fig.tight_layout()
        corr_plot = img(fig_b64(fig), "Lower-triangle Pearson correlations")
    else:
        corr_plot = "<p>Not enough numeric columns.</p>"

    # ── 6. Feature correlations with primary price ──
    target = next((k for k in plot_keys if "fews" in k and "price" in k), None) or \
             next((k for k in plot_keys if "price" in k), None)
    feat_corr_plot = ""
    if target:
        others = [k for k in plot_keys if k != target]
        feat_corrs = {}
        t_arr = col_arr(target)
        for k in others:
            f_arr = col_arr(k)
            mask = ~(np.isnan(t_arr) | np.isnan(f_arr))
            if mask.sum() > 2:
                feat_corrs[k] = np.corrcoef(t_arr[mask], f_arr[mask])[0, 1]
        if feat_corrs:
            top = sorted(feat_corrs, key=lambda k: abs(feat_corrs[k]), reverse=True)[:18]
            vals = [feat_corrs[k] for k in top]
            colors = [PALETTE[0] if v >= 0 else PALETTE[3] for v in vals]
            fig, ax = plt.subplots(figsize=(8, max(4, len(top) * 0.42)))
            ax.barh(top, vals, color=colors)
            ax.axvline(0, color="black", lw=0.8)
            ax.set_xlabel(f"Pearson r with '{target}'")
            ax.set_title(f"Feature Correlations with Target: {target}")
            fig.tight_layout()
            feat_corr_plot = img(fig_b64(fig),
                f"Correlation of all features with the primary price target")

    # ── 7. Distributions ──
    ncols = 3
    nrows = -(-len(plot_keys) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.2 * nrows))
    axes_flat = np.array(axes).flatten()
    for i, k in enumerate(plot_keys):
        ax = axes_flat[i]
        arr = col_arr(k)
        arr_v = arr[~np.isnan(arr)]
        if len(arr_v) == 0:
            ax.set_visible(False)
            continue
        ax.hist(arr_v, bins=25, color=PALETTE[i % len(PALETTE)], alpha=0.75, density=True)
        ax.set_title(k, fontsize=8)
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle("Univariate Distributions", fontsize=12)
    fig.tight_layout()
    dist_plot = img(fig_b64(fig), "Histogram of all numeric features")

    # ── Assemble HTML ──
    css = """
<style>
body{font-family:'Segoe UI',sans-serif;max-width:1200px;margin:auto;padding:20px;background:#f8f9fa;color:#212529}
h1{color:#1a5276;border-bottom:3px solid #1a5276;padding-bottom:8px}
h2{color:#2874a6;margin-top:36px}
table.tbl{border-collapse:collapse;width:100%;margin:10px 0;font-size:13px}
table.tbl th,table.tbl td{border:1px solid #dee2e6;padding:6px 10px;text-align:left}
table.tbl th{background:#2874a6;color:#fff}
table.tbl tr:nth-child(even){background:#eaf4fb}
section{background:#fff;border-radius:8px;padding:20px;margin:20px 0;box-shadow:0 1px 4px rgba(0,0,0,.1)}
figure{margin:10px 0;text-align:center}
figcaption{font-size:12px;color:#6c757d;margin-top:4px}
img{border-radius:4px;max-width:100%}
.badge{display:inline-block;background:#1a5276;color:#fff;border-radius:4px;padding:2px 8px;font-size:12px;margin-right:6px}
</style>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>IMTISAL EDA Report</title>{css}</head>
<body>
<h1>IMTISAL — Maize Price Prediction: EDA Report</h1>
<p>Generated: <strong>2026-03-31</strong> &nbsp;|&nbsp; Adamawa State, Nigeria
&nbsp;|&nbsp;<span class="badge">ML-Ready</span></p>

{section("1. Dataset Overview &amp; Descriptive Statistics", ov_html + "<br/>" + desc_html)}
{section("2. Missing Data Analysis", miss_plot)}
{section("3. Maize Price Time-Series", ts_plot)}
{section("4. Seasonal Price Patterns", seasonal_plot)}
{section("5. Univariate Distributions", dist_plot)}
{section("6. Correlation Matrix", corr_plot)}
{section("7. Feature Correlations with Price Target", feat_corr_plot)}
</body></html>"""

    with open(EDA_HTML, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"[EDA] report saved -> {EDA_HTML}")


# ===============================================================================
# MAIN
# ===============================================================================

def main():
    print("=" * 60)
    print("  IMTISAL Maize Price Prediction — Local Pipeline")
    print("=" * 60)
    print(f"  Source data : {SRC_DIR}")
    print(f"  Output      : {OUT_DIR}")
    print(f"  Reports     : {RPT_DIR}\n")

    # 1. Load & clean each dataset
    fews   = load_fews()
    nfpt   = load_nfpt()
    power  = load_power_monthly()
    design = load_power_design()
    temps  = load_temperature_table()

    # 2. Save cleaned files
    print("\n[SAVE] cleaned individual files …")
    save_cleaned("fews_adamawa_maize",    fews)
    save_cleaned("nfpt_adamawa_maize",    nfpt)
    save_cleaned("power_monthly_cloud",   power)
    save_cleaned("power_design_monthly",  design)
    save_cleaned("temperature_monthly",   temps)

    # 3. Merge
    master = merge_all(fews, nfpt, power, design, temps)

    # 4. Write master CSV
    col_keys = write_master(master)

    # 5. EDA
    eda_report(master, col_keys)

    print("\n" + "=" * 60)
    print("  DONE")
    print(f"  Master CSV  : {MASTER_CSV}")
    print(f"  EDA Report  : {EDA_HTML}")
    print("  Open the HTML file in your browser to explore the data.")
    print("=" * 60)


if __name__ == "__main__":
    main()
