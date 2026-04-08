"""
Exploratory Data Analysis — generates a self-contained HTML report.

Sections:
  1. Dataset overview (shape, dtypes, memory)
  2. Missing data heatmap + summary table
  3. Univariate distributions (histograms + KDE for all numeric cols)
  4. Correlation heatmap
  5. Time-series plots for key columns
  6. Boxplots (seasonal patterns by month/quarter)
  7. Pairplot for top correlated features with target
"""

import os
import base64
import warnings
import io

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — works on all systems
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from config import MASTER_CSV, EDA_REPORT, REPORTS_DIR

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted", font_scale=0.95)
COLORS = sns.color_palette("muted")


# ── helpers ───────────────────────────────────────────────────────────────────

def _fig_to_b64(fig) -> str:
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _img_tag(b64: str, caption: str = "") -> str:
    tag = f'<figure><img src="data:image/png;base64,{b64}" style="max-width:100%;"/>'
    if caption:
        tag += f"<figcaption>{caption}</figcaption>"
    tag += "</figure>"
    return tag


def _section(title: str, content: str) -> str:
    return (
        f'<section><h2>{title}</h2>{content}</section>\n'
    )


def _df_to_html(df: pd.DataFrame) -> str:
    return df.to_html(
        classes="tbl", border=0, index=True,
        float_format=lambda x: f"{x:,.4f}",
    )


# ── plot builders ─────────────────────────────────────────────────────────────

def _plot_missing(df: pd.DataFrame) -> str:
    miss = df.isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    if miss.empty:
        return "<p>No missing values in the master dataset.</p>"

    fig, ax = plt.subplots(figsize=(10, max(4, len(miss) * 0.35)))
    miss.plot.barh(ax=ax, color=COLORS[0])
    ax.set_xlabel("Missing fraction")
    ax.set_title("Missing Data by Column")
    for bar, val in zip(ax.patches, miss):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}", va="center", fontsize=8)
    return _img_tag(_fig_to_b64(fig), "Missing data proportions")


def _plot_distributions(df: pd.DataFrame, num_cols: list[str]) -> str:
    if not num_cols:
        return "<p>No numeric columns found.</p>"
    ncols = 3
    nrows = -(-len(num_cols) // ncols)   # ceiling division
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = np.array(axes).flatten()
    for i, col in enumerate(num_cols):
        ax = axes[i]
        data = df[col].dropna()
        ax.hist(data, bins=30, color=COLORS[i % len(COLORS)], alpha=0.7, density=True)
        try:
            data.plot.kde(ax=ax, color="black", lw=1)
        except Exception:
            pass
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Univariate Distributions", fontsize=12, y=1.01)
    fig.tight_layout()
    return _img_tag(_fig_to_b64(fig), "Histogram + KDE for all numeric columns")


def _plot_correlation(df: pd.DataFrame, num_cols: list[str]) -> str:
    if len(num_cols) < 2:
        return "<p>Not enough numeric columns for correlation.</p>"
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(max(8, len(num_cols) * 0.7), max(6, len(num_cols) * 0.6)))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=len(num_cols) <= 20,
        fmt=".2f", cmap="RdYlGn", center=0,
        linewidths=0.5, ax=ax, annot_kws={"size": 7},
    )
    ax.set_title("Pearson Correlation Matrix")
    fig.tight_layout()
    return _img_tag(_fig_to_b64(fig), "Lower-triangle correlation heatmap")


def _plot_timeseries(df: pd.DataFrame, date_col: str, num_cols: list[str]) -> str:
    if date_col not in df.columns:
        return "<p>No date column available for time-series plots.</p>"
    ts = df.copy()
    ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
    ts.sort_values(date_col, inplace=True)

    # Pick up to 6 most informative columns (highest std relative to mean)
    cv = (ts[num_cols].std() / ts[num_cols].mean().abs()).replace([np.inf, -np.inf], np.nan).dropna()
    top_cols = cv.nlargest(min(6, len(cv))).index.tolist()

    ncols = 2
    nrows = -(-len(top_cols) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 3 * nrows), sharex=False)
    axes = np.array(axes).flatten()

    for i, col in enumerate(top_cols):
        ax = axes[i]
        ax.plot(ts[date_col], ts[col], color=COLORS[i % len(COLORS)], lw=1.2)
        ax.set_title(col, fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Time-Series Trends (Top Varying Columns)", fontsize=12, y=1.01)
    fig.tight_layout()
    return _img_tag(_fig_to_b64(fig), "Monthly/yearly trends for key variables")


def _plot_seasonal_boxplots(df: pd.DataFrame, date_col: str, num_cols: list[str]) -> str:
    if "month" not in df.columns:
        return ""
    # Pick up to 4 columns that likely contain 'price' or 'yield'
    priority = [c for c in num_cols if any(k in c for k in ("price", "yield", "rainfall", "production"))]
    cols_to_plot = (priority + num_cols)[:4]
    if not cols_to_plot:
        return ""

    fig, axes = plt.subplots(1, len(cols_to_plot), figsize=(5 * len(cols_to_plot), 4))
    if len(cols_to_plot) == 1:
        axes = [axes]
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    for ax, col in zip(axes, cols_to_plot):
        data = df[[col, "month"]].dropna()
        data["Month"] = data["month"].map(lambda m: month_names[int(m)-1] if 1 <= int(m) <= 12 else str(m))
        order = month_names
        sns.boxplot(data=data, x="Month", y=col, order=order, ax=ax, palette="muted")
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    fig.suptitle("Seasonal Patterns by Month", fontsize=12, y=1.01)
    fig.tight_layout()
    return _img_tag(_fig_to_b64(fig), "Boxplots showing seasonal distribution by month")


def _target_correlations(df: pd.DataFrame, num_cols: list[str]) -> str:
    """Bar chart: correlation of all numeric cols with the primary price column."""
    price_cols = [c for c in num_cols if "price" in c]
    if not price_cols:
        return ""
    target = price_cols[0]
    others = [c for c in num_cols if c != target]
    if not others:
        return ""
    corr = df[others].corrwith(df[target]).sort_values(key=abs, ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(8, max(4, len(corr) * 0.38)))
    colors = [COLORS[0] if v >= 0 else COLORS[3] for v in corr.values]
    ax.barh(corr.index, corr.values, color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel(f"Pearson r with '{target}'")
    ax.set_title(f"Feature Correlations with Target: {target}")
    fig.tight_layout()
    return _img_tag(_fig_to_b64(fig), f"Correlation of features with '{target}'")


# ── main EDA function ─────────────────────────────────────────────────────────

def generate_eda(df: pd.DataFrame | None = None) -> str:
    """
    Generate HTML EDA report.
    If df is None, loads from MASTER_CSV.
    Returns path to the HTML file.
    """
    print("\n[eda] Generating EDA report …")

    if df is None:
        if not os.path.exists(MASTER_CSV):
            raise FileNotFoundError(f"Master CSV not found: {MASTER_CSV}")
        df = pd.read_csv(MASTER_CSV)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude pure calendar columns from most charts
    skip = {"year", "month", "quarter"}
    num_plot_cols = [c for c in num_cols if c not in skip]

    # ── Overview table ──
    overview_data = {
        "Rows"         : df.shape[0],
        "Columns"      : df.shape[1],
        "Numeric cols" : len(num_cols),
        "Date range"   : f"{df['date'].iloc[0]} → {df['date'].iloc[-1]}" if "date" in df.columns else "N/A",
        "Memory (MB)"  : round(df.memory_usage(deep=True).sum() / 1e6, 2),
        "Total missing": int(df.isna().sum().sum()),
        "Duplicates"   : int(df.duplicated().sum()),
    }
    overview_html = "<table class='tbl'>" + "".join(
        f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in overview_data.items()
    ) + "</table>"

    desc_html = _df_to_html(df[num_plot_cols].describe().T)

    # ── Missing summary table ──
    miss_df = pd.DataFrame({
        "Missing count"  : df.isna().sum(),
        "Missing %"      : (df.isna().mean() * 100).round(2),
        "Dtype"          : df.dtypes,
        "Unique values"  : df.nunique(),
    }).sort_values("Missing %", ascending=False)
    miss_html = _df_to_html(miss_df)

    # ── Plots ──
    miss_plot    = _plot_missing(df)
    dist_plot    = _plot_distributions(df, num_plot_cols)
    corr_plot    = _plot_correlation(df, num_plot_cols)
    ts_plot      = _plot_timeseries(df, "date", num_plot_cols)
    box_plot     = _plot_seasonal_boxplots(df, "date", num_plot_cols)
    target_corr  = _target_correlations(df, num_plot_cols)

    # ── Assemble HTML ──
    css = """
    <style>
      body  { font-family: 'Segoe UI', sans-serif; max-width: 1200px; margin: auto; padding: 20px; background:#f9f9f9; color:#222; }
      h1    { color: #1a5276; border-bottom: 3px solid #1a5276; padding-bottom: 8px; }
      h2    { color: #2874a6; margin-top: 40px; }
      table.tbl { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 13px; }
      table.tbl th, table.tbl td { border: 1px solid #ddd; padding: 6px 10px; text-align: left; }
      table.tbl th { background: #2874a6; color: white; }
      table.tbl tr:nth-child(even) { background: #eaf4fb; }
      section { background: white; border-radius: 8px; padding: 20px; margin: 20px 0; box-shadow: 0 1px 4px rgba(0,0,0,.1); }
      figure { margin: 10px 0; text-align: center; }
      figcaption { font-size: 12px; color: #555; margin-top: 4px; }
      img   { border-radius: 4px; }
      .badge { display: inline-block; background:#1a5276; color:white; border-radius:4px; padding:2px 8px; font-size:12px; margin-right:6px; }
    </style>
    """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>IMTISAL EDA Report</title>{css}</head>
<body>
<h1>IMTISAL — Maize Price Prediction: EDA Report</h1>
<p>Generated: <strong>2026-03-31</strong> &nbsp;|&nbsp; Adamawa State, Nigeria &nbsp;|&nbsp;
<span class="badge">ML-Ready Dataset</span></p>

{_section("1. Dataset Overview", overview_html + "<br/>" + desc_html)}
{_section("2. Missing Data Analysis", miss_html + miss_plot)}
{_section("3. Univariate Distributions", dist_plot)}
{_section("4. Correlation Matrix", corr_plot)}
{_section("5. Target Feature Correlations", target_corr)}
{_section("6. Time-Series Trends", ts_plot)}
{_section("7. Seasonal Patterns (Monthly Boxplots)", box_plot)}
</body></html>"""

    with open(EDA_REPORT, "w", encoding="utf-8") as fh:
        fh.write(html)

    print(f"[eda] Report saved → {EDA_REPORT}")
    return EDA_REPORT
