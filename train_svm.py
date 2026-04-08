"""
Support Vector Regression — Adamawa Maize Price Prediction
Features: lag prices, rolling means, calendar, climate
Preprocessing: StandardScaler (required for SVM)
"""

import csv, math, os, pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ── paths ──────────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
DATA_CSV  = os.path.join(BASE, "data", "output", "maize_master_clean.csv")
MODEL_OUT = os.path.join(BASE, "models", "svm_model.pkl")
PLOT_OUT  = os.path.join(BASE, "reports", "svm_forecast.png")
os.makedirs(os.path.join(BASE, "models"),  exist_ok=True)
os.makedirs(os.path.join(BASE, "reports"), exist_ok=True)

# ── 1. Load & filter ───────────────────────────────────────────────────────────
print("=" * 60)
print("  IMTISAL -- SVR Maize Price Model")
print("=" * 60)

rows = []
with open(DATA_CSV, encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        rows.append(r)

before = len(rows)
rows   = [r for r in rows if r.get("fews_is_imputed", "0").strip() == "0"]
print(f"\n[data] {before} rows loaded, {before-len(rows)} imputed dropped -> {len(rows)} clean rows")

import datetime as dt

def safe_float(v):
    try:
        f = float(v)
        return None if (math.isnan(f) or f == -999) else f
    except Exception:
        return None

dates      = [r["date"] for r in rows]
parsed_dt  = [dt.datetime.strptime(d, "%Y-%m-%d") for d in dates]
raw_prices = [safe_float(r["fews_maize_price_ngn_kg"]) for r in rows]

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

# ── 3. Feature engineering (identical to RF) ──────────────────────────────────
LAG_PERIODS     = [1, 2, 3, 6, 12]
ROLLING_WINDOWS = [3, 6]

FEATURE_NAMES = (
    [f"price_lag_{l}" for l in LAG_PERIODS] +
    [f"price_rolling_{w}" for w in ROLLING_WINDOWS] +
    ["month", "quarter", "year"] +
    CLIMATE_COLS
)

X_list, y_list, valid_dates = [], [], []
for i in range(len(log_prices)):
    if i < 12:
        continue
    row = {}
    for lag in LAG_PERIODS:
        row[f"price_lag_{lag}"] = log_prices[i - lag]
    for win in ROLLING_WINDOWS:
        row[f"price_rolling_{win}"] = sum(log_prices[i - win: i]) / win
    row["month"]   = parsed_dt[i].month
    row["quarter"] = (parsed_dt[i].month - 1) // 3 + 1
    row["year"]    = parsed_dt[i].year
    for col, vals in climate.items():
        row[col] = vals[i] if vals[i] is not None else 0.0
    X_list.append([row[f] for f in FEATURE_NAMES])
    y_list.append(log_prices[i])
    valid_dates.append(parsed_dt[i])

X = np.array(X_list, dtype=float)
y = np.array(y_list, dtype=float)

print(f"\n[features] Matrix shape : {X.shape}  ({len(FEATURE_NAMES)} features)")
print(f"[features] Date range   : {valid_dates[0].strftime('%Y-%m-%d')} -> "
      f"{valid_dates[-1].strftime('%Y-%m-%d')}")

# ── 4. Train/test split ───────────────────────────────────────────────────────
split       = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
dates_train     = valid_dates[:split]
dates_test      = valid_dates[split:]
raw_test        = [raw_prices[12 + split + i] for i in range(len(X_test))]

print(f"\n[split]  Train : {len(X_train)} samples  "
      f"({dates_train[0].strftime('%Y-%m-%d')} -> {dates_train[-1].strftime('%Y-%m-%d')})")
print(f"         Test  : {len(X_test)} samples  "
      f"({dates_test[0].strftime('%Y-%m-%d')} -> {dates_test[-1].strftime('%Y-%m-%d')})")

# ── 5. Pipeline: StandardScaler + SVR ────────────────────────────────────────
# Scaler is fit only on train, applied to both — wrapped in Pipeline so
# GridSearchCV never leaks test stats into the scaler fit.
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svr",    SVR()),
])

param_grid = {
    "svr__kernel" : ["rbf", "linear"],
    "svr__C"      : [0.1, 1, 10, 100],
    "svr__epsilon": [0.01, 0.1, 0.5],
    "svr__gamma"  : ["scale", "auto"],   # only used by rbf; ignored for linear
}

total_combos = (len(param_grid["svr__kernel"]) *
                len(param_grid["svr__C"]) *
                len(param_grid["svr__epsilon"]) *
                len(param_grid["svr__gamma"]))

print(f"\n[grid]  Searching {total_combos} combinations x 5-fold TimeSeriesSplit ...")
print(f"        (may take ~1-2 minutes)")

tscv = TimeSeriesSplit(n_splits=5)
grid = GridSearchCV(
    pipeline, param_grid,
    cv=tscv,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    verbose=1,
    refit=True,
)
grid.fit(X_train, y_train)

best_params = grid.best_params_
best_model  = grid.best_estimator_
cv_mae_log  = -grid.best_score_

print(f"\n[grid]  Best params    : {best_params}")
print(f"[grid]  CV MAE (log)   : {cv_mae_log:.6f}")

# Retrieve fitted scaler for display
scaler     = best_model.named_steps["scaler"]
X_train_sc = scaler.transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"\n[scaler] Feature means (train): "
      f"{', '.join(f'{v:.3f}' for v in scaler.mean_[:5])} ...")
print(f"[scaler] Feature stds  (train): "
      f"{', '.join(f'{v:.3f}' for v in scaler.scale_[:5])} ...")

# ── 6. Evaluate ───────────────────────────────────────────────────────────────
y_pred_log = best_model.predict(X_test)
y_pred     = np.exp(y_pred_log)
y_actual   = np.array(raw_test)

mae  = mean_absolute_error(y_actual, y_pred)
rmse = math.sqrt(mean_squared_error(y_actual, y_pred))
mape = float(np.mean(np.abs((y_pred - y_actual) / y_actual)) * 100)

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
print(f"    Mean bias : {np.mean(residuals):>10,.2f}  "
      f"({'over' if np.mean(residuals)>0 else 'under'}-predicting)")
print(f"    Std       : {np.std(residuals):>10,.2f}")
print(f"    Min / Max : {np.min(residuals):>10,.2f}  /  {np.max(residuals):,.2f}")

print(f"\n  Per-month test detail:")
print(f"  {'Date':<13} {'Actual':>10} {'Predicted':>10} {'Error':>10} {'Err%':>7}")
print(f"  {'-'*55}")
for d, act, pred in zip(dates_test, y_actual, y_pred):
    err  = pred - act
    errp = err / act * 100
    flag = " <--" if abs(errp) > 25 else ""
    print(f"  {d.strftime('%Y-%m-%d'):<13} {act:>10,.0f} {pred:>10,.0f} "
          f"{err:>+10,.0f} {errp:>+6.1f}%{flag}")

# ── 7. Plots ──────────────────────────────────────────────────────────────────
BLUE  = "#2874a6"
RED   = "#c0392b"
GREEN = "#27ae60"
ORNG  = "#e67e22"

fig, axes = plt.subplots(2, 1, figsize=(13, 10))

# --- Panel 1: Full series + test forecast ---
ax = axes[0]
all_raw = [raw_prices[i] for i in range(12, len(raw_prices))]
ax.plot(valid_dates, all_raw,  color=BLUE, lw=1.5, label="Actual price")
ax.plot(dates_test,  y_pred,   color=RED,  lw=2, linestyle="--", label="SVR forecast")
ax.axvline(dates_train[-1], color="gray", lw=1, linestyle=":", label="Train/test split")
ax.fill_betweenx(
    [0, max(all_raw) * 1.15],
    dates_test[0], dates_test[-1],
    color=RED, alpha=0.06, label="Test window"
)
ax.set_title(
    f"Adamawa Maize Price — SVR "
    f"(kernel={best_params['svr__kernel']}, C={best_params['svr__C']}, "
    f"eps={best_params['svr__epsilon']}, gamma={best_params['svr__gamma']})\n"
    f"MAE={mae:,.0f}  RMSE={rmse:,.0f}  MAPE={mape:.1f}%",
    fontsize=10,
)
ax.set_ylabel("NGN / kg")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator())
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.legend(fontsize=9)

# --- Panel 2: Test zoom — actual vs predicted + error % ---
ax2 = axes[1]
x  = np.arange(len(dates_test))
w  = 0.38
ax2.bar(x - w/2, y_actual, width=w, label="Actual",    color=BLUE, alpha=0.82)
ax2.bar(x + w/2, y_pred,   width=w, label="Predicted", color=RED,  alpha=0.82)
ax2.set_xticks(x)
ax2.set_xticklabels(
    [d.strftime("%b %Y") for d in dates_test],
    rotation=45, ha="right", fontsize=8,
)
ax2.set_title("Test Period — Actual vs Predicted (original scale)", fontsize=11)
ax2.set_ylabel("NGN / kg")
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax2.legend(fontsize=9)
for xi, (act, pred) in enumerate(zip(y_actual, y_pred)):
    errp = (pred - act) / act * 100
    col  = RED if abs(errp) > 25 else GREEN
    ax2.text(xi, max(act, pred) * 1.01, f"{errp:+.1f}%",
             ha="center", va="bottom", fontsize=7, color=col, fontweight="bold")

fig.tight_layout(pad=2.5)
fig.savefig(PLOT_OUT, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"\n[plot]  Saved -> {PLOT_OUT}")

# ── 8. Save model ─────────────────────────────────────────────────────────────
with open(MODEL_OUT, "wb") as fh:
    pickle.dump({
        "model"          : best_model,       # Pipeline (scaler + SVR)
        "feature_names"  : FEATURE_NAMES,
        "best_params"    : best_params,
        "cv_mae_log"     : cv_mae_log,
        "metrics"        : {"mae": mae, "rmse": rmse, "mape": mape},
        "train_end_date" : dates_train[-1].strftime("%Y-%m-%d"),
        "test_end_date"  : dates_test[-1].strftime("%Y-%m-%d"),
        "log_transformed": True,
    }, fh)
print(f"[model] Saved -> {MODEL_OUT}")

# ── 9. Final summary ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)
print(f"  Clean rows used       : {len(X)}  (after lag warm-up)")
print(f"  Train / Test          : {len(X_train)} / {len(X_test)} samples")
print(f"  Features              : {len(FEATURE_NAMES)}")
print(f"  Best kernel           : {best_params['svr__kernel']}")
print(f"  Best C                : {best_params['svr__C']}")
print(f"  Best epsilon          : {best_params['svr__epsilon']}")
print(f"  Best gamma            : {best_params['svr__gamma']}")
print(f"  CV MAE (log scale)    : {cv_mae_log:.6f}")
print(f"  MAE   (NGN/kg)        : {mae:>10,.2f}")
print(f"  RMSE  (NGN/kg)        : {rmse:>10,.2f}")
print(f"  MAPE                  : {mape:>9.2f} %")
print("=" * 60)
