"""
Random Forest Regressor — Adamawa Maize Price Prediction
Features: lag prices, rolling means, calendar, climate
"""

import csv, math, os, pickle, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ── paths ──────────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
DATA_CSV  = os.path.join(BASE, "data", "output", "maize_master_clean.csv")
MODEL_OUT = os.path.join(BASE, "models", "rf_model.pkl")
PLOT_OUT  = os.path.join(BASE, "reports", "rf_forecast.png")
os.makedirs(os.path.join(BASE, "models"),  exist_ok=True)
os.makedirs(os.path.join(BASE, "reports"), exist_ok=True)

# ── 1. Load & filter ───────────────────────────────────────────────────────────
print("=" * 60)
print("  IMTISAL -- Random Forest Maize Price Model")
print("=" * 60)

rows = []
with open(DATA_CSV, encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        rows.append(r)

before = len(rows)
rows = [r for r in rows if r.get("fews_is_imputed", "0").strip() == "0"]
print(f"\n[data] {before} rows loaded, {before - len(rows)} imputed dropped -> {len(rows)} clean rows")

import datetime as dt

def safe_float(v, fallback=None):
    try:
        f = float(v)
        return None if (math.isnan(f) or f == -999) else f
    except Exception:
        return fallback

dates       = [r["date"] for r in rows]
parsed_dt   = [dt.datetime.strptime(d, "%Y-%m-%d") for d in dates]
raw_prices  = [safe_float(r["fews_maize_price_ngn_kg"]) for r in rows]

# Climate columns present in the file
CLIMATE_COLS = [
    "climate_cloud_amt",
    "design_avg_precip_mm",
    "design_avg_wind_ms",
    "temp_dbavg_c",
]

climate = {col: [safe_float(r.get(col, "")) for r in rows] for col in CLIMATE_COLS}

# ── 2. Log transform ──────────────────────────────────────────────────────────
log_prices = [math.log(p) for p in raw_prices]
print(f"[data] Price range : {min(raw_prices):>10,.1f} - {max(raw_prices):,.1f} NGN/kg")
print(f"[data] Log range   : {min(log_prices):>10.4f} - {max(log_prices):.4f}")

# ── 3. Feature engineering ────────────────────────────────────────────────────
LAG_PERIODS     = [1, 2, 3, 6, 12]
ROLLING_WINDOWS = [3, 6]

def build_features(log_p, dates_dt, climate_dict):
    """
    Build feature matrix and target vector.
    Rows with insufficient history (lag_12 needs index >= 12) are dropped.
    Returns X (list of dicts), y (list), valid_dates, valid_raw_prices.
    """
    n = len(log_p)
    X, y, valid_dates = [], [], []

    for i in range(n):
        # Need at least lag_12 history
        if i < 12:
            continue

        row = {}

        # Lag features (on log scale)
        for lag in LAG_PERIODS:
            row[f"price_lag_{lag}"] = log_p[i - lag]

        # Rolling mean features (on log scale)
        for win in ROLLING_WINDOWS:
            row[f"price_rolling_{win}"] = sum(log_p[i - win: i]) / win

        # Calendar features
        row["month"]   = dates_dt[i].month
        row["quarter"] = (dates_dt[i].month - 1) // 3 + 1
        row["year"]    = dates_dt[i].year

        # Climate features
        for col, vals in climate_dict.items():
            v = vals[i]
            row[col] = v if v is not None else 0.0   # fill residual NaN with 0

        X.append(row)
        y.append(log_p[i])
        valid_dates.append(dates_dt[i])

    return X, y, valid_dates

X_dicts, y, valid_dates = build_features(log_prices, parsed_dt, climate)

# Ordered feature names
FEATURE_NAMES = (
    [f"price_lag_{l}" for l in LAG_PERIODS] +
    [f"price_rolling_{w}" for w in ROLLING_WINDOWS] +
    ["month", "quarter", "year"] +
    CLIMATE_COLS
)

X = np.array([[row[f] for f in FEATURE_NAMES] for row in X_dicts], dtype=float)
y = np.array(y, dtype=float)

print(f"\n[features] Matrix shape : {X.shape}  ({len(FEATURE_NAMES)} features)")
print(f"[features] Features     : {FEATURE_NAMES}")
print(f"[features] Date range   : {valid_dates[0].strftime('%Y-%m-%d')} -> "
      f"{valid_dates[-1].strftime('%Y-%m-%d')}")

# ── 4. Train / test split ─────────────────────────────────────────────────────
split     = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
dates_train     = valid_dates[:split]
dates_test      = valid_dates[split:]

# Corresponding raw prices for the test window (for reverse-transform metrics)
# valid_dates starts at index 12 of the original series
raw_test = [raw_prices[12 + split + i] for i in range(len(X_test))]

print(f"\n[split]  Train : {len(X_train)} samples  "
      f"({dates_train[0].strftime('%Y-%m-%d')} -> {dates_train[-1].strftime('%Y-%m-%d')})")
print(f"         Test  : {len(X_test)} samples  "
      f"({dates_test[0].strftime('%Y-%m-%d')} -> {dates_test[-1].strftime('%Y-%m-%d')})")

# ── 5. GridSearchCV with TimeSeriesSplit ──────────────────────────────────────
param_grid = {
    "n_estimators"     : [100, 300, 500],
    "max_depth"        : [3, 5, 10, None],
    "min_samples_split": [2, 5, 10],
}
total_combos = (len(param_grid["n_estimators"]) *
                len(param_grid["max_depth"]) *
                len(param_grid["min_samples_split"]))

print(f"\n[grid] Searching {total_combos} combinations x 5-fold TimeSeriesSplit ...")
print(f"       (this may take 1-3 minutes)")

tscv = TimeSeriesSplit(n_splits=5)
rf   = RandomForestRegressor(random_state=42, n_jobs=-1)

grid_search = GridSearchCV(
    rf, param_grid,
    cv=tscv,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    verbose=1,
    refit=True,
)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_model  = grid_search.best_estimator_
cv_mae_log  = -grid_search.best_score_

print(f"\n[grid] Best params      : {best_params}")
print(f"[grid] CV MAE (log)     : {cv_mae_log:.6f}")

# ── 6. Evaluate on test set ───────────────────────────────────────────────────
y_pred_log = best_model.predict(X_test)
y_pred     = np.exp(y_pred_log)
y_actual   = np.array(raw_test)

mae  = mean_absolute_error(y_actual, y_pred)
rmse = math.sqrt(mean_squared_error(y_actual, y_pred))
mape = float(np.mean(np.abs((y_pred - y_actual) / y_actual)) * 100)

# Also log-scale metrics
mae_log  = mean_absolute_error(y_test, y_pred_log)
rmse_log = math.sqrt(mean_squared_error(y_test, y_pred_log))

print("\n" + "-" * 50)
print("  TEST SET EVALUATION METRICS")
print("-" * 50)
print(f"  MAE   (NGN/kg)  : {mae:>12,.2f}")
print(f"  RMSE  (NGN/kg)  : {rmse:>12,.2f}")
print(f"  MAPE            : {mape:>11.2f} %")
print(f"  MAE  (log scale): {mae_log:>12.6f}")
print(f"  RMSE (log scale): {rmse_log:>12.6f}")
print("-" * 50)

residuals = y_pred - y_actual
print(f"\n  Residuals (NGN/kg):")
print(f"    Mean bias : {np.mean(residuals):>10,.2f}  ({'over' if np.mean(residuals)>0 else 'under'}-predicting)")
print(f"    Std       : {np.std(residuals):>10,.2f}")
print(f"    Min / Max : {np.min(residuals):>10,.2f}  /  {np.max(residuals):,.2f}")

# Per-month breakdown
print(f"\n  Per-month test detail:")
print(f"  {'Date':<13} {'Actual':>10} {'Predicted':>10} {'Error':>10} {'Err%':>7}")
print(f"  {'-'*54}")
for d, act, pred in zip(dates_test, y_actual, y_pred):
    err  = pred - act
    errp = err / act * 100
    marker = " <--" if abs(errp) > 25 else ""
    print(f"  {d.strftime('%Y-%m-%d'):<13} {act:>10,.0f} {pred:>10,.0f} "
          f"{err:>+10,.0f} {errp:>+6.1f}%{marker}")

# ── 7. Feature importance ─────────────────────────────────────────────────────
importances = best_model.feature_importances_
fi_pairs    = sorted(zip(FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True)
print(f"\n  Feature Importances:")
for fname, imp in fi_pairs:
    bar = "#" * int(imp * 50)
    print(f"  {fname:<28} {imp:.4f}  {bar}")

# ── 8. Plots ──────────────────────────────────────────────────────────────────
BLUE  = "#2874a6"
RED   = "#c0392b"
GREEN = "#27ae60"

fig, axes = plt.subplots(3, 1, figsize=(13, 14))

# --- Panel 1: Full series + test predictions ---
ax = axes[0]
all_raw = [raw_prices[i] for i in range(12, len(raw_prices))]
all_dt  = valid_dates
# train region
ax.plot(all_dt, all_raw, color=BLUE, lw=1.5, label="Actual price")
ax.plot(dates_test, y_pred, color=RED, lw=2, linestyle="--", label="RF forecast")
ax.axvline(dates_train[-1], color="gray", lw=1, linestyle=":", label="Train/test split")
ax.fill_betweenx([0, max(all_raw) * 1.15],
                 dates_test[0], dates_test[-1],
                 color=RED, alpha=0.06, label="Test window")
ax.set_title(
    f"Adamawa Maize Price — Random Forest (best: {best_params})\n"
    f"MAE={mae:,.0f}  RMSE={rmse:,.0f}  MAPE={mape:.1f}%",
    fontsize=11)
ax.set_ylabel("NGN / kg")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator())
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.legend(fontsize=9)

# --- Panel 2: Test zoom — actual vs predicted bars ---
ax2 = axes[1]
x  = np.arange(len(dates_test))
w  = 0.38
ax2.bar(x - w/2, y_actual, width=w, label="Actual",    color=BLUE,  alpha=0.82)
ax2.bar(x + w/2, y_pred,   width=w, label="Predicted", color=RED,   alpha=0.82)
ax2.set_xticks(x)
ax2.set_xticklabels([d.strftime("%b %Y") for d in dates_test],
                    rotation=45, ha="right", fontsize=8)
ax2.set_title("Test Period — Actual vs Predicted (original scale)", fontsize=11)
ax2.set_ylabel("NGN / kg")
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax2.legend(fontsize=9)
for xi, (act, pred) in enumerate(zip(y_actual, y_pred)):
    err_pct = (pred - act) / act * 100
    col = RED if abs(err_pct) > 25 else GREEN
    ax2.text(xi, max(act, pred) * 1.01, f"{err_pct:+.1f}%",
             ha="center", va="bottom", fontsize=7, color=col, fontweight="bold")

# --- Panel 3: Feature importance ---
ax3 = axes[2]
fi_names = [p[0] for p in fi_pairs]
fi_vals  = [p[1] for p in fi_pairs]
bar_colors = [BLUE if "price" in n else GREEN if n in ("month","quarter","year")
              else "#e67e22" for n in fi_names]
bars = ax3.barh(fi_names[::-1], fi_vals[::-1], color=bar_colors[::-1], alpha=0.85)
ax3.set_xlabel("Feature Importance (mean decrease in impurity)")
ax3.set_title("Feature Importances — Random Forest", fontsize=11)
for bar, val in zip(bars, fi_vals[::-1]):
    ax3.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
             f"{val:.3f}", va="center", fontsize=8)
# legend
from matplotlib.patches import Patch
ax3.legend(handles=[
    Patch(color=BLUE,    label="Price / lag features"),
    Patch(color=GREEN,   label="Calendar features"),
    Patch(color="#e67e22", label="Climate features"),
], fontsize=8, loc="lower right")

fig.tight_layout(pad=2.5)
fig.savefig(PLOT_OUT, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"\n[plot]  Saved -> {PLOT_OUT}")

# ── 9. Save model ─────────────────────────────────────────────────────────────
with open(MODEL_OUT, "wb") as fh:
    pickle.dump({
        "model"          : best_model,
        "feature_names"  : FEATURE_NAMES,
        "best_params"    : best_params,
        "cv_mae_log"     : cv_mae_log,
        "metrics"        : {"mae": mae, "rmse": rmse, "mape": mape},
        "train_end_date" : dates_train[-1].strftime("%Y-%m-%d"),
        "test_end_date"  : dates_test[-1].strftime("%Y-%m-%d"),
        "log_transformed": True,
        "feature_importances": dict(fi_pairs),
    }, fh)
print(f"[model] Saved -> {MODEL_OUT}")

# ── 10. Final summary ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)
print(f"  Clean rows used     : {len(X)}  (after lag warm-up)")
print(f"  Train / Test        : {len(X_train)} / {len(X_test)} samples")
print(f"  Features            : {len(FEATURE_NAMES)}")
print(f"  Best n_estimators   : {best_params['n_estimators']}")
print(f"  Best max_depth      : {best_params['max_depth']}")
print(f"  Best min_samples_split: {best_params['min_samples_split']}")
print(f"  CV MAE (log scale)  : {cv_mae_log:.6f}")
print(f"  MAE   (NGN/kg)      : {mae:>10,.2f}")
print(f"  RMSE  (NGN/kg)      : {rmse:>10,.2f}")
print(f"  MAPE                : {mape:>9.2f} %")
top3 = fi_pairs[:3]
print(f"  Top-3 features      : {', '.join(f'{n}({v:.3f})' for n,v in top3)}")
print("=" * 60)
