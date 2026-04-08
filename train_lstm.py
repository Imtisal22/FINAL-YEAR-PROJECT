"""
LSTM model — Adamawa Maize Price Prediction
2 LSTM layers (64->32) + Dropout(0.2) + Dense(1)
Lookback window: 12 months
Scaler: MinMaxScaler on all features + target
"""

import csv, math, os, pickle, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── TensorFlow / Keras ─────────────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"          # suppress C++ info logs
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("TensorFlow version:", tf.__version__)

# ── paths ──────────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
DATA_CSV  = os.path.join(BASE, "data", "output", "maize_master_clean.csv")
MODEL_OUT = os.path.join(BASE, "models",  "lstm_model.keras")
PLOT_OUT  = os.path.join(BASE, "reports", "lstm_forecast.png")
os.makedirs(os.path.join(BASE, "models"),  exist_ok=True)
os.makedirs(os.path.join(BASE, "reports"), exist_ok=True)

LOOKBACK  = 12    # months
EPOCHS    = 200
BATCH     = 16
PATIENCE  = 15

# ── 1. Load & filter ───────────────────────────────────────────────────────────
print("=" * 60)
print("  IMTISAL -- LSTM Maize Price Model")
print("=" * 60)

rows = []
with open(DATA_CSV, encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        rows.append(r)

before = len(rows)
rows   = [r for r in rows if r.get("fews_is_imputed", "0").strip() == "0"]
print(f"\n[data]  {before} rows loaded, {before-len(rows)} imputed dropped -> {len(rows)} clean rows")

import datetime as dt

def safe_float(v, fallback=0.0):
    try:
        f = float(v)
        return fallback if (math.isnan(f) or f == -999) else f
    except Exception:
        return fallback

dates      = [r["date"] for r in rows]
parsed_dt  = [dt.datetime.strptime(d, "%Y-%m-%d") for d in dates]
raw_prices = [safe_float(r["fews_maize_price_ngn_kg"]) for r in rows]

# ── 2. Log-transform target ───────────────────────────────────────────────────
log_prices = [math.log(p) for p in raw_prices]
print(f"[data]  Price range: {min(raw_prices):,.1f} - {max(raw_prices):,.1f} NGN/kg")
print(f"[data]  Log range  : {min(log_prices):.4f} - {max(log_prices):.4f}")

# ── 3. Build feature matrix ───────────────────────────────────────────────────
# Features: log_price (target, also used as input), climate cols, calendar
CLIMATE_COLS = [
    "climate_cloud_amt",
    "design_avg_precip_mm",
    "design_avg_wind_ms",
    "temp_dbavg_c",
]
FEATURE_NAMES = ["log_price"] + CLIMATE_COLS + ["month", "quarter", "year"]
N_FEATURES    = len(FEATURE_NAMES)

raw_matrix = []   # shape: (n_timesteps, N_FEATURES)
for i, r in enumerate(rows):
    row_vec = [log_prices[i]]
    for col in CLIMATE_COLS:
        row_vec.append(safe_float(r.get(col, ""), fallback=0.0))
    row_vec.append(parsed_dt[i].month)
    row_vec.append((parsed_dt[i].month - 1) // 3 + 1)
    row_vec.append(parsed_dt[i].year)
    raw_matrix.append(row_vec)

raw_matrix = np.array(raw_matrix, dtype=float)   # (106, 9)
print(f"\n[features] Raw matrix  : {raw_matrix.shape}  {FEATURE_NAMES}")

# ── 4. MinMaxScaler on the ENTIRE feature matrix ──────────────────────────────
# Scaler is fit on train rows only (first split_idx rows); applied to all.
# We split BEFORE fitting the scaler to avoid leakage.
split_idx = int(len(raw_matrix) * 0.8)

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(raw_matrix[:split_idx])                       # fit on train only
scaled_matrix = scaler.transform(raw_matrix)             # transform all

# ── 5. Build sequences of length LOOKBACK ────────────────────────────────────
def make_sequences(data, lookback):
    """
    data   : (T, F) scaled array
    returns: X (T-lookback, lookback, F),  y (T-lookback,)  [target = col 0]
    """
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback: i, :])   # all features, past 12 steps
        y.append(data[i, 0])                  # log_price at step i (col 0)
    return np.array(X), np.array(y)

X_all, y_all = make_sequences(scaled_matrix, LOOKBACK)
# Corresponding dates and raw prices (offset by lookback)
seq_dates  = parsed_dt[LOOKBACK:]
seq_raw    = raw_prices[LOOKBACK:]

print(f"[sequences] X shape : {X_all.shape}   y shape: {y_all.shape}")
print(f"[sequences] Date range: {seq_dates[0].strftime('%Y-%m-%d')} -> "
      f"{seq_dates[-1].strftime('%Y-%m-%d')}")

# ── 6. Train / test split ─────────────────────────────────────────────────────
# split_idx was based on pre-sequence rows; adjust for the lookback offset
# sequences start at index LOOKBACK in the original series,
# so train/test boundary is at (split_idx - LOOKBACK) in the sequence array
seq_split = split_idx - LOOKBACK

X_train, X_test = X_all[:seq_split], X_all[seq_split:]
y_train, y_test = y_all[:seq_split], y_all[seq_split:]
dates_train     = seq_dates[:seq_split]
dates_test      = seq_dates[seq_split:]
raw_test        = seq_raw[seq_split:]

print(f"\n[split]  Train : {len(X_train)} sequences  "
      f"({dates_train[0].strftime('%Y-%m-%d')} -> {dates_train[-1].strftime('%Y-%m-%d')})")
print(f"         Test  : {len(X_test)} sequences  "
      f"({dates_test[0].strftime('%Y-%m-%d')} -> {dates_test[-1].strftime('%Y-%m-%d')})")

# Use last 15% of train as validation during training
val_split = int(len(X_train) * 0.85)
X_tr, X_val = X_train[:val_split], X_train[val_split:]
y_tr, y_val = y_train[:val_split], y_train[val_split:]
print(f"         Val   : {len(X_val)} sequences (last 15% of train, for early stopping)")

# ── 7. Build LSTM model ───────────────────────────────────────────────────────
tf.random.set_seed(42)

model = Sequential([
    LSTM(64, input_shape=(LOOKBACK, N_FEATURES),
         return_sequences=True,
         name="lstm_1"),
    Dropout(0.2, name="drop_1"),
    LSTM(32, return_sequences=False, name="lstm_2"),
    Dropout(0.2, name="drop_2"),
    Dense(1, name="output"),
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
model.summary()

# ── 8. Train ──────────────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor="val_loss", patience=PATIENCE,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7,
                      min_lr=1e-6, verbose=1),
]

print(f"\n[train] Max epochs={EPOCHS}  batch={BATCH}  early_stop patience={PATIENCE}")
history = model.fit(
    X_tr, y_tr,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH,
    callbacks=callbacks,
    verbose=1,
)

epochs_trained = len(history.history["loss"])
best_val_loss  = min(history.history["val_loss"])
print(f"\n[train] Stopped at epoch {epochs_trained}  |  best val_loss={best_val_loss:.6f}")

# ── 9. Predict & inverse-transform ───────────────────────────────────────────
y_pred_scaled = model.predict(X_test, verbose=0).flatten()

# Inverse-scale: reconstruct a dummy matrix with all features set to 0
# except col 0 (log_price), then call scaler.inverse_transform
def inverse_scale_target(scaled_vals, scaler, col_idx=0, n_features=N_FEATURES):
    dummy = np.zeros((len(scaled_vals), n_features))
    dummy[:, col_idx] = scaled_vals
    inv = scaler.inverse_transform(dummy)
    return inv[:, col_idx]

y_pred_log = inverse_scale_target(y_pred_scaled, scaler)
y_pred     = np.exp(y_pred_log)
y_actual   = np.array(raw_test)

# Also inverse-scale the true test targets for log-scale metrics
y_test_log = inverse_scale_target(y_test, scaler)

# ── 10. Metrics ───────────────────────────────────────────────────────────────
mae  = mean_absolute_error(y_actual, y_pred)
rmse = math.sqrt(mean_squared_error(y_actual, y_pred))
mape = float(np.mean(np.abs((y_pred - y_actual) / y_actual)) * 100)
mae_log  = mean_absolute_error(y_test_log, y_pred_log)
rmse_log = math.sqrt(mean_squared_error(y_test_log, y_pred_log))

print("\n" + "-" * 50)
print("  TEST SET EVALUATION METRICS")
print("-" * 50)
print(f"  Epochs trained  : {epochs_trained} / {EPOCHS}")
print(f"  MAE   (NGN/kg)  : {mae:>12,.2f}")
print(f"  RMSE  (NGN/kg)  : {rmse:>12,.2f}")
print(f"  MAPE            : {mape:>11.2f} %")
print(f"  MAE  (log scale): {mae_log:>12.6f}")
print(f"  RMSE (log scale): {rmse_log:>12.6f}")
print("-" * 50)

residuals = y_pred - y_actual
print(f"\n  Residuals (NGN/kg):")
print(f"    Mean bias : {np.mean(residuals):>10,.2f}  "
      f"({'over' if np.mean(residuals) > 0 else 'under'}-predicting)")
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

# ── 11. Plots ─────────────────────────────────────────────────────────────────
BLUE  = "#2874a6"
RED   = "#c0392b"
GREEN = "#27ae60"

fig, axes = plt.subplots(3, 1, figsize=(13, 14))

# --- Panel 1: Full series + test predictions ---
ax = axes[0]
ax.plot(seq_dates, seq_raw, color=BLUE, lw=1.5, label="Actual price")
ax.plot(dates_test, y_pred, color=RED,  lw=2, linestyle="--", label="LSTM forecast")
ax.axvline(dates_train[-1], color="gray", lw=1, linestyle=":", label="Train/test split")
ax.fill_betweenx([0, max(seq_raw) * 1.15],
                 dates_test[0], dates_test[-1],
                 color=RED, alpha=0.06, label="Test window")
ax.set_title(
    f"Adamawa Maize Price — LSTM (64->32, lookback=12, dropout=0.2)\n"
    f"Epochs={epochs_trained}  |  MAE={mae:,.0f}  RMSE={rmse:,.0f}  MAPE={mape:.1f}%",
    fontsize=11,
)
ax.set_ylabel("NGN / kg")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator())
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.legend(fontsize=9)

# --- Panel 2: Test zoom — actual vs predicted ---
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

# --- Panel 3: Training loss curve ---
ax3 = axes[2]
ep  = range(1, epochs_trained + 1)
ax3.plot(ep, history.history["loss"],     color=BLUE, lw=1.5, label="Train loss (MSE)")
ax3.plot(ep, history.history["val_loss"], color=RED,  lw=1.5, linestyle="--",
         label="Val loss (MSE)")
best_ep = history.history["val_loss"].index(min(history.history["val_loss"])) + 1
ax3.axvline(best_ep, color="gray", lw=1, linestyle=":", label=f"Best epoch ({best_ep})")
ax3.set_title("LSTM Training Loss Curve", fontsize=11)
ax3.set_xlabel("Epoch")
ax3.set_ylabel("MSE Loss (scaled)")
ax3.legend(fontsize=9)
ax3.set_yscale("log")    # log scale makes convergence easier to see

fig.tight_layout(pad=2.5)
fig.savefig(PLOT_OUT, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"\n[plot]  Saved -> {PLOT_OUT}")

# ── 12. Save model + scaler ───────────────────────────────────────────────────
model.save(MODEL_OUT)
scaler_path = os.path.join(BASE, "models", "lstm_scaler.pkl")
with open(scaler_path, "wb") as fh:
    pickle.dump({
        "scaler"         : scaler,
        "feature_names"  : FEATURE_NAMES,
        "lookback"       : LOOKBACK,
        "n_features"     : N_FEATURES,
        "metrics"        : {"mae": mae, "rmse": rmse, "mape": mape},
        "epochs_trained" : epochs_trained,
        "train_end_date" : dates_train[-1].strftime("%Y-%m-%d"),
        "test_end_date"  : dates_test[-1].strftime("%Y-%m-%d"),
        "log_transformed": True,
    }, fh)
print(f"[model] Keras model saved -> {MODEL_OUT}")
print(f"[model] Scaler saved      -> {scaler_path}")

# ── 13. Summary ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)
print(f"  Clean rows used     : {len(rows)}  (imputed dropped)")
print(f"  Sequences (lookback={LOOKBACK}) : {len(X_all)}")
print(f"  Train / Test        : {len(X_train)} / {len(X_test)} sequences")
print(f"  Features            : {N_FEATURES}  {FEATURE_NAMES}")
print(f"  Epochs trained      : {epochs_trained} / {EPOCHS}  (early stop)")
print(f"  Best val loss (MSE) : {best_val_loss:.6f}")
print(f"  MAE   (NGN/kg)      : {mae:>10,.2f}")
print(f"  RMSE  (NGN/kg)      : {rmse:>10,.2f}")
print(f"  MAPE                : {mape:>9.2f} %")
print("=" * 60)
