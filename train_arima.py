"""
ARIMA model training for Adamawa maize price prediction.
Uses ~/imtisal_project/data/output/maize_master_clean.csv
"""

import csv, math, os, pickle, sys, io, base64

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pmdarima as pm
from pmdarima.arima import auto_arima

# ── paths ──────────────────────────────────────────────────────────────────────
BASE     = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(BASE, "data", "output", "maize_master_clean.csv")
MODEL_OUT= os.path.join(BASE, "models", "arima_model.pkl")
PLOT_OUT = os.path.join(BASE, "reports", "arima_forecast.png")
os.makedirs(os.path.join(BASE, "models"),  exist_ok=True)
os.makedirs(os.path.join(BASE, "reports"), exist_ok=True)

# ── 1. Load data, drop imputed rows ───────────────────────────────────────────
print("=" * 58)
print("  IMTISAL — ARIMA Maize Price Model")
print("=" * 58)

rows = []
with open(DATA_CSV, encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        rows.append(r)

# Drop imputed
before = len(rows)
rows = [r for r in rows if r.get("fews_is_imputed", "0").strip() == "0"]
print(f"\n[data] Loaded {before} rows, dropped {before - len(rows)} imputed -> {len(rows)} clean rows")

dates  = [r["date"] for r in rows]
prices = [float(r["fews_maize_price_ngn_kg"]) for r in rows]

import datetime as dt
parsed_dates = [dt.datetime.strptime(d, "%Y-%m-%d") for d in dates]

# ── 2. Log transformation ──────────────────────────────────────────────────────
log_prices = [math.log(p) for p in prices]
print(f"[data] Price range: {min(prices):,.1f} - {max(prices):,.1f} NGN/kg")
print(f"[data] Log-price range: {min(log_prices):.4f} - {max(log_prices):.4f}")

# ── 3. Train / test split (80/20, time-ordered) ───────────────────────────────
split = int(len(log_prices) * 0.8)
train_log, test_log   = log_prices[:split], log_prices[split:]
train_dates           = parsed_dates[:split]
test_dates            = parsed_dates[split:]
train_prices          = prices[:split]
test_prices           = prices[split:]

print(f"\n[split] Train: {len(train_log)} months  ({dates[0]} -> {dates[split-1]})")
print(f"        Test : {len(test_log)} months  ({dates[split]} -> {dates[-1]})")

# ── 4. auto_arima ─────────────────────────────────────────────────────────────
print("\n[arima] Running auto_arima (this may take ~30-60 s) ...")

model = auto_arima(
    train_log,
    start_p=0, max_p=4,
    start_q=0, max_q=4,
    d=None,              # auto-determine differencing
    seasonal=True,
    m=12,                # monthly seasonality
    start_P=0, max_P=2,
    start_Q=0, max_Q=2,
    D=None,
    information_criterion="aic",
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    trace=True,
)

order         = model.order
seasonal_order= model.seasonal_order
print(f"\n[arima] Best order       : ARIMA{order}")
print(f"[arima] Seasonal order   : {seasonal_order}")
print(f"[arima] AIC              : {model.aic():.4f}")

# ── 5. Forecast on test set ───────────────────────────────────────────────────
n_test = len(test_log)
log_forecast, conf_int = model.predict(n_periods=n_test, return_conf_int=True, alpha=0.05)

# Reverse log transform
fc_prices   = np.exp(log_forecast)
ci_lower    = np.exp(conf_int[:, 0])
ci_upper    = np.exp(conf_int[:, 1])
actual      = np.array(test_prices)

# ── 6. Metrics ────────────────────────────────────────────────────────────────
mae  = float(np.mean(np.abs(fc_prices - actual)))
rmse = float(np.sqrt(np.mean((fc_prices - actual) ** 2)))
mape = float(np.mean(np.abs((fc_prices - actual) / actual)) * 100)

# Also compute on log scale (model-native)
mae_log  = float(np.mean(np.abs(log_forecast - np.array(test_log))))
rmse_log = float(np.sqrt(np.mean((log_forecast - np.array(test_log)) ** 2)))

print("\n" + "-" * 45)
print("  TEST SET EVALUATION METRICS")
print("-" * 45)
print(f"  MAE   (NGN/kg)  : {mae:>12,.2f}")
print(f"  RMSE  (NGN/kg)  : {rmse:>12,.2f}")
print(f"  MAPE            : {mape:>11.2f} %")
print(f"  MAE  (log scale): {mae_log:>12.6f}")
print(f"  RMSE (log scale): {rmse_log:>12.6f}")
print("-" * 45)

# Residuals summary
residuals = fc_prices - actual
print(f"\n  Residuals (NGN/kg):")
print(f"    Mean  : {np.mean(residuals):>10,.2f}")
print(f"    Std   : {np.std(residuals):>10,.2f}")
print(f"    Min   : {np.min(residuals):>10,.2f}")
print(f"    Max   : {np.max(residuals):>10,.2f}")

# ── 7. Plot ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 9))

# -- top panel: full series + forecast --
ax = axes[0]
ax.plot(parsed_dates, prices, color="#2874a6", lw=1.5, label="Actual (all)")
ax.plot(test_dates, fc_prices, color="#e74c3c", lw=2,
        linestyle="--", label="ARIMA forecast")
ax.fill_between(test_dates, ci_lower, ci_upper,
                color="#e74c3c", alpha=0.15, label="95% CI")
ax.axvline(train_dates[-1], color="gray", lw=1, linestyle=":", label="Train/test split")
ax.set_title(f"Adamawa Maize Price — ARIMA{order} x {seasonal_order}\n"
             f"MAE={mae:,.0f}  RMSE={rmse:,.0f}  MAPE={mape:.1f}%",
             fontsize=11)
ax.set_ylabel("NGN / kg")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator())
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
ax.legend(fontsize=9)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))

# -- bottom panel: test period zoom with actual vs predicted --
ax2 = axes[1]
x = np.arange(len(test_dates))
w = 0.35
ax2.bar(x - w/2, actual,   width=w, label="Actual",   color="#2874a6", alpha=0.8)
ax2.bar(x + w/2, fc_prices,width=w, label="Forecast", color="#e74c3c", alpha=0.8)
ax2.set_xticks(x)
ax2.set_xticklabels([d.strftime("%b %Y") for d in test_dates],
                    rotation=45, ha="right", fontsize=8)
ax2.set_title("Test Period — Actual vs Forecast (original scale)", fontsize=11)
ax2.set_ylabel("NGN / kg")
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax2.legend(fontsize=9)

# annotate each pair with % error
for xi, (act, fc) in enumerate(zip(actual, fc_prices)):
    err = (fc - act) / act * 100
    color = "#c0392b" if abs(err) > 20 else "#27ae60"
    ax2.text(xi, max(act, fc) * 1.02, f"{err:+.1f}%",
             ha="center", va="bottom", fontsize=7, color=color)

fig.tight_layout(pad=2.5)
fig.savefig(PLOT_OUT, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"\n[plot] Saved -> {PLOT_OUT}")

# ── 8. Save model ─────────────────────────────────────────────────────────────
with open(MODEL_OUT, "wb") as fh:
    pickle.dump({
        "model"         : model,
        "order"         : order,
        "seasonal_order": seasonal_order,
        "aic"           : model.aic(),
        "metrics"       : {"mae": mae, "rmse": rmse, "mape": mape},
        "train_end_date": dates[split - 1],
        "log_transformed": True,
    }, fh)
print(f"[model] Saved -> {MODEL_OUT}")

# ── 9. Final summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 58)
print("  SUMMARY")
print("=" * 58)
print(f"  Clean rows used   : {len(rows)}  (imputed dropped)")
print(f"  Train / Test      : {split} / {n_test} months")
print(f"  Date range        : {dates[0]} -> {dates[-1]}")
print(f"  Best ARIMA order  : {order}")
print(f"  Seasonal order    : {seasonal_order}")
print(f"  AIC               : {model.aic():.4f}")
print(f"  MAE               : {mae:>10,.2f} NGN/kg")
print(f"  RMSE              : {rmse:>10,.2f} NGN/kg")
print(f"  MAPE              : {mape:>9.2f} %")
print("=" * 58)
