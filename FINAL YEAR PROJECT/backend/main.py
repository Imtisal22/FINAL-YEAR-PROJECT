"""
IMTISAL Maize Price Prediction — FastAPI Backend
Run: uvicorn main:app --reload --port 8000  (from backend/ directory)
     OR:  python main.py

Model files are downloaded from Hugging Face Hub at startup when
running on Render (or any host where the local models/ dir is absent).
HF repo: https://huggingface.co/Alim-7/imtisal-maize-models
"""

import csv, math, os, pickle, sys, datetime
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── suppress TF noise ─────────────────────────────────────────────────────────
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# ── paths ─────────────────────────────────────────────────────────────────────
BACKEND_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

# On Render: models live in /tmp/models (writable, ephemeral).
# Locally:   models live in PROJECT_ROOT/models (pre-trained files present).
_LOCAL_MODELS = os.path.join(PROJECT_ROOT, "models")
MODELS_DIR    = _LOCAL_MODELS if os.path.isdir(_LOCAL_MODELS) else "/tmp/models"
DATA_CSV      = os.path.join(PROJECT_ROOT, "data", "output", "maize_master_clean.csv")

# ── Hugging Face config ───────────────────────────────────────────────────────
HF_REPO_ID   = "Alim-7/imtisal-maize-models"
MODEL_FILES  = [
    "arima_model.pkl",
    "rf_model.pkl",
    "svm_model.pkl",
    "lstm_model.keras",
    "lstm_scaler.pkl",
]


def download_models_from_hf() -> None:
    """
    Download model files from Hugging Face Hub into MODELS_DIR.
    Skips files that already exist locally (avoids re-downloading on warm restarts).
    Only runs when MODELS_DIR != the local models/ folder (i.e. on Render).
    """
    if MODELS_DIR == _LOCAL_MODELS:
        print("[models] Using local models directory — no HF download needed.")
        return

    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"[models] Downloading model files from HF Hub: {HF_REPO_ID}")

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise RuntimeError(
            "huggingface_hub not installed. Add it to requirements.txt."
        )

    for fname in MODEL_FILES:
        dest = os.path.join(MODELS_DIR, fname)
        if os.path.exists(dest):
            print(f"  [skip] {fname} already present.")
            continue
        print(f"  [download] {fname} ...", end=" ", flush=True)
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=fname,
            repo_type="model",
            local_dir=MODELS_DIR,
        )
        size_kb = os.path.getsize(dest) // 1024
        print(f"done ({size_kb} KB)")

    print("[models] All model files ready.")

# ══════════════════════════════════════════════════════════════════════════════
# Model store — loaded once at startup
# ══════════════════════════════════════════════════════════════════════════════

store: dict = {}   # populated by lifespan


def _load_csv(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _safe_float(v, fallback=0.0):
    try:
        f = float(v)
        return fallback if (math.isnan(f) or f == -999) else f
    except Exception:
        return fallback


def _load_all_models():
    loaded = {}

    # ── SVM ───────────────────────────────────────────────────────────────────
    svm_path = os.path.join(MODELS_DIR, "svm_model.pkl")
    with open(svm_path, "rb") as f:
        svm_data = pickle.load(f)
    loaded["svm"] = svm_data
    print(f"  [ok] SVM loaded   — MAPE {svm_data['metrics']['mape']:.2f}%")

    # ── Random Forest ─────────────────────────────────────────────────────────
    rf_path = os.path.join(MODELS_DIR, "rf_model.pkl")
    with open(rf_path, "rb") as f:
        rf_data = pickle.load(f)
    loaded["rf"] = rf_data
    print(f"  [ok] RF  loaded   — MAPE {rf_data['metrics']['mape']:.2f}%")

    # ── ARIMA ─────────────────────────────────────────────────────────────────
    arima_path = os.path.join(MODELS_DIR, "arima_model.pkl")
    with open(arima_path, "rb") as f:
        arima_data = pickle.load(f)
    loaded["arima"] = arima_data
    print(f"  [ok] ARIMA loaded — MAPE {arima_data['metrics']['mape']:.2f}%")

    # ── LSTM ──────────────────────────────────────────────────────────────────
    try:
        from tensorflow import keras
        lstm_keras_path  = os.path.join(MODELS_DIR, "lstm_model.keras")
        lstm_scaler_path = os.path.join(MODELS_DIR, "lstm_scaler.pkl")
        lstm_model = keras.models.load_model(lstm_keras_path)
        with open(lstm_scaler_path, "rb") as f:
            lstm_meta = pickle.load(f)
        loaded["lstm"] = {"model": lstm_model, **lstm_meta}
        print(f"  [ok] LSTM loaded  — MAPE {lstm_meta['metrics']['mape']:.2f}%")
    except Exception as exc:
        print(f"  [warn] LSTM load failed: {exc}")
        loaded["lstm"] = None

    return loaded


def _load_historical() -> list[dict]:
    rows = _load_csv(DATA_CSV)
    clean = [r for r in rows if r.get("fews_is_imputed", "0").strip() == "0"]
    return clean


# ── feature builders ──────────────────────────────────────────────────────────

# Feature names expected by SVM and RF (must match training order exactly)
SVM_RF_FEATURES = [
    "price_lag_1", "price_lag_2", "price_lag_3", "price_lag_6", "price_lag_12",
    "price_rolling_3", "price_rolling_6",
    "month", "quarter", "year",
    "climate_cloud_amt", "design_avg_precip_mm", "design_avg_wind_ms", "temp_dbavg_c",
]

LSTM_FEATURES = [
    "log_price", "climate_cloud_amt", "design_avg_precip_mm",
    "design_avg_wind_ms", "temp_dbavg_c", "month", "quarter", "year",
]
LSTM_LOOKBACK = 12


def _build_svm_rf_features(
    hist_log_prices: list[float],
    month: int, year: int,
    cloud: float, precip: float, wind: float, temp: float,
) -> np.ndarray:
    """
    Build the 14-feature vector for SVM / RF from recent log-price history.
    hist_log_prices must have at least 12 values (most-recent last).
    """
    if len(hist_log_prices) < 12:
        raise ValueError("Need at least 12 historical price points for lag features.")

    lp = hist_log_prices  # shorthand
    quarter = (month - 1) // 3 + 1
    vec = [
        lp[-1],                            # lag_1
        lp[-2],                            # lag_2
        lp[-3],                            # lag_3
        lp[-6],                            # lag_6
        lp[-12],                           # lag_12
        sum(lp[-3:]) / 3,                  # rolling_3
        sum(lp[-6:]) / 6,                  # rolling_6
        month, quarter, year,
        cloud, precip, wind, temp,
    ]
    return np.array(vec, dtype=float).reshape(1, -1)


def _build_lstm_sequence(
    hist_rows: list[dict],
    scaler,
    month: int, year: int,
    cloud: float, precip: float, wind: float, temp: float,
    n_features: int,
) -> np.ndarray:
    """
    Build a (1, 12, n_features) scaled input tensor for the LSTM.
    Uses the last 11 real historical rows + the new target row as step 12.
    """
    quarter = (month - 1) // 3 + 1

    raw_seq = []
    for r in hist_rows[-11:]:
        raw_seq.append([
            _safe_float(r.get("fews_maize_price_ngn_kg", 0)),   # will be log'd below
            _safe_float(r.get("climate_cloud_amt", 0)),
            _safe_float(r.get("design_avg_precip_mm", 0)),
            _safe_float(r.get("design_avg_wind_ms", 0)),
            _safe_float(r.get("temp_dbavg_c", 0)),
            _safe_float(r.get("month", month)),
            _safe_float(r.get("quarter", quarter)),
            _safe_float(r.get("year", year)),
        ])

    # Log-transform the price column (index 0) for the historical rows
    for row in raw_seq:
        row[0] = math.log(row[0]) if row[0] > 0 else 0.0

    # The synthetic "next" step uses the last known log price as placeholder
    last_log_price = raw_seq[-1][0] if raw_seq else 0.0
    new_step = [last_log_price, cloud, precip, wind, temp, month, quarter, year]
    raw_seq.append(new_step)

    arr = np.array(raw_seq, dtype=float)                 # (12, n_features)
    arr_scaled = scaler.transform(arr)                   # MinMaxScaler
    return arr_scaled.reshape(1, LSTM_LOOKBACK, n_features)


def _arima_steps_ahead(train_end_date: str, target_year: int, target_month: int) -> int:
    """Months from one past train_end to target month (min 1)."""
    end = datetime.datetime.strptime(train_end_date, "%Y-%m-%d")
    one_past = end + datetime.timedelta(days=32)
    one_past = one_past.replace(day=1)
    target = datetime.datetime(target_year, target_month, 1)
    delta_months = (target.year - one_past.year) * 12 + (target.month - one_past.month) + 1
    return max(1, delta_months)


# ══════════════════════════════════════════════════════════════════════════════
# Lifespan — load everything once on startup
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n[startup] Checking model files ...")
    download_models_from_hf()          # no-op locally; downloads on Render
    print("[startup] Loading models and data ...")
    store["models"]     = _load_all_models()
    store["historical"] = _load_historical()
    store["log_prices"] = [
        math.log(_safe_float(r["fews_maize_price_ngn_kg"], 1))
        for r in store["historical"]
    ]
    print(f"[startup] {len(store['historical'])} clean historical rows loaded.")
    print("[startup] Ready.\n")
    yield
    store.clear()


# ══════════════════════════════════════════════════════════════════════════════
# App
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="IMTISAL Maize Price Prediction API",
    description="Adamawa State, Nigeria — ARIMA / Random Forest / SVR / LSTM",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
# Schemas
# ══════════════════════════════════════════════════════════════════════════════

class PredictRequest(BaseModel):
    month: int               = Field(..., ge=1, le=12, description="Target month (1-12)")
    year: int                = Field(default=2024,    description="Target year")
    climate_cloud_amt: float = Field(default=55.0,    description="Cloud amount (%)")
    design_avg_precip_mm: float = Field(default=80.0, description="Avg precipitation (mm)")
    design_avg_wind_ms: float   = Field(default=3.0,  description="Avg wind speed (m/s)")
    temp_dbavg_c: float         = Field(default=29.0, description="Avg dry-bulb temp (C)")
    use_model: str           = Field(default="svm",
                                     description="Model: 'svm', 'rf', 'arima', 'lstm'")


class PredictResponse(BaseModel):
    model_used: str
    target_month: str
    predicted_price_ngn_kg: float
    predicted_price_ngn_bag100kg: float
    confidence_note: str


class ModelMetrics(BaseModel):
    model: str
    mae_ngn_kg: float
    rmse_ngn_kg: float
    mape_pct: float
    train_end: str
    test_end: str
    best_params: Optional[dict] = None


# ══════════════════════════════════════════════════════════════════════════════
# Endpoints
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Health"])
def health_check():
    """Health check — confirms API is running and models are loaded."""
    models_ok = {k: (v is not None) for k, v in store["models"].items()}
    return {
        "status"      : "ok",
        "service"     : "IMTISAL Maize Price Prediction API",
        "version"     : "1.0.0",
        "models_loaded": models_ok,
        "historical_rows": len(store.get("historical", [])),
        "timestamp"   : datetime.datetime.utcnow().isoformat() + "Z",
    }


@app.get("/api/models", response_model=list[ModelMetrics], tags=["Models"])
def list_models():
    """Return performance metrics for all 4 trained models."""
    m = store["models"]
    results = []

    for key, label in [("svm","SVR"), ("rf","Random Forest"),
                       ("arima","ARIMA"), ("lstm","LSTM")]:
        data = m[key]
        if data is None:
            continue
        metrics  = data["metrics"]
        t_end    = data.get("train_end_date", "N/A")
        te_end   = data.get("test_end_date",  "N/A")
        params   = data.get("best_params") or data.get("order") or None
        if isinstance(params, tuple):
            params = {"order": str(params),
                      "seasonal_order": str(data.get("seasonal_order", ""))}
        results.append(ModelMetrics(
            model       = label,
            mae_ngn_kg  = round(metrics["mae"],  2),
            rmse_ngn_kg = round(metrics["rmse"], 2),
            mape_pct    = round(metrics["mape"], 2),
            train_end   = t_end,
            test_end    = te_end,
            best_params = params,
        ))
    return results


@app.get("/api/model-stats", tags=["Models"])
def model_stats():
    """MAE, RMSE, MAPE for all 4 models as flat JSON — useful for dashboards."""
    m = store["models"]
    out = {}
    for key in ("svm", "rf", "arima", "lstm"):
        data = m[key]
        if data is None:
            out[key] = None
            continue
        mt = data["metrics"]
        out[key] = {
            "mae_ngn_kg" : round(mt["mae"],  2),
            "rmse_ngn_kg": round(mt["rmse"], 2),
            "mape_pct"   : round(mt["mape"], 2),
        }
    out["best_model"] = "svm"
    out["note"] = "SVR achieved lowest MAPE (15.73%) on 2022-11 to 2024-05 test set."
    return out


@app.get("/api/historical", tags=["Data"])
def get_historical(limit: int = 200, offset: int = 0):
    """Return rows from maize_master_clean.csv (imputed rows excluded)."""
    hist = store["historical"]
    total = len(hist)
    page  = hist[offset: offset + limit]
    return {
        "total"  : total,
        "offset" : offset,
        "limit"  : limit,
        "count"  : len(page),
        "data"   : page,
    }


@app.post("/api/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(req: PredictRequest):
    """
    Predict maize price for a given month using the selected model.
    Lag features are derived automatically from historical data.
    """
    model_key = req.use_model.lower().strip()
    if model_key not in ("svm", "rf", "arima", "lstm"):
        raise HTTPException(status_code=400,
            detail="use_model must be one of: svm, rf, arima, lstm")

    m    = store["models"]
    data = m[model_key]
    if data is None:
        raise HTTPException(status_code=503,
            detail=f"Model '{model_key}' failed to load at startup.")

    hist      = store["historical"]
    log_prices = store["log_prices"]
    target_label = f"{req.year}-{req.month:02d}"

    try:
        # ── SVR / RF ───────────────────────────────────────────────────────────
        if model_key in ("svm", "rf"):
            feat_vec = _build_svm_rf_features(
                log_prices,
                req.month, req.year,
                req.climate_cloud_amt, req.design_avg_precip_mm,
                req.design_avg_wind_ms, req.temp_dbavg_c,
            )
            log_pred = float(data["model"].predict(feat_vec)[0])
            price    = math.exp(log_pred)

        # ── ARIMA ─────────────────────────────────────────────────────────────
        elif model_key == "arima":
            arima_model = data["model"]
            train_end   = data["train_end_date"]
            n_steps     = _arima_steps_ahead(train_end, req.year, req.month)
            log_preds, _ = arima_model.predict(
                n_periods=n_steps, return_conf_int=True)
            log_pred = float(log_preds[-1])
            price    = math.exp(log_pred)

        # ── LSTM ──────────────────────────────────────────────────────────────
        elif model_key == "lstm":
            scaler     = data["scaler"]
            lstm_model = data["model"]
            n_features = data["n_features"]

            seq = _build_lstm_sequence(
                hist, scaler,
                req.month, req.year,
                req.climate_cloud_amt, req.design_avg_precip_mm,
                req.design_avg_wind_ms, req.temp_dbavg_c,
                n_features,
            )
            scaled_pred = float(lstm_model.predict(seq, verbose=0)[0][0])

            # Inverse-scale the target column (col 0 = log_price)
            dummy = np.zeros((1, n_features))
            dummy[0, 0] = scaled_pred
            log_pred = float(scaler.inverse_transform(dummy)[0, 0])
            price    = math.exp(log_pred)

    except Exception as exc:
        raise HTTPException(status_code=500,
            detail=f"Prediction error ({model_key}): {str(exc)}")

    # Confidence note
    mape = data["metrics"]["mape"]
    if mape < 20:
        note = f"High confidence — model MAPE={mape:.1f}% on test set."
    elif mape < 35:
        note = f"Moderate confidence — model MAPE={mape:.1f}% on test set."
    else:
        note = (f"Low confidence — model MAPE={mape:.1f}%. "
                "Training data predates 2023 inflation surge; use SVR for best results.")

    return PredictResponse(
        model_used                  = model_key.upper(),
        target_month                = target_label,
        predicted_price_ngn_kg      = round(price, 2),
        predicted_price_ngn_bag100kg= round(price * 100, 2),
        confidence_note             = note,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Validation endpoints
# ══════════════════════════════════════════════════════════════════════════════

# Test period: 2022-11-01 to 2024-05-01  (matches SVM/RF test set)
TEST_START = "2022-11"
TEST_END   = "2024-05"


def _test_period_rows() -> list[dict]:
    """Return historical rows that fall within the test period."""
    rows = []
    for r in store["historical"]:
        d = r.get("date", "")[:7]   # "YYYY-MM"
        if TEST_START <= d <= TEST_END:
            rows.append(r)
    return rows


def _predict_svm_rf_for_row(model_key: str, r: dict, log_prices_up_to: list[float]) -> float:
    """Run SVM or RF prediction for a single historical row using only
    log-prices available at that point in time."""
    m = store["models"][model_key]
    month  = int(r["month"])
    year   = int(r["year"])
    cloud  = _safe_float(r.get("climate_cloud_amt", 0))
    precip = _safe_float(r.get("design_avg_precip_mm", 0))
    wind   = _safe_float(r.get("design_avg_wind_ms", 0))
    temp   = _safe_float(r.get("temp_dbavg_c", 0))
    feat   = _build_svm_rf_features(log_prices_up_to, month, year,
                                    cloud, precip, wind, temp)
    log_pred = float(m["model"].predict(feat)[0])
    return round(math.exp(log_pred), 2)


def _validation_for_model(model_key: str) -> list[dict]:
    """
    Replay the test period for one model and return a list of
    {date, actual_price, predicted_price, model} dicts.

    For SVM/RF  : rebuild features row-by-row using only train-time log-prices.
    For ARIMA   : predict n steps ahead from train_end.
    For LSTM    : step through with a growing prefix of actual data.
    """
    hist       = store["historical"]
    model_data = store["models"][model_key]
    test_rows  = _test_period_rows()
    results    = []

    # ── SVM / RF ──────────────────────────────────────────────────────────────
    if model_key in ("svm", "rf"):
        # Build the full log-price list up to (not including) the test period
        # so that each test-step is predicted with exactly the information
        # the model had at training time.
        train_log_prices = []
        for r in hist:
            d = r.get("date", "")[:7]
            if d < TEST_START:
                p = _safe_float(r.get("fews_maize_price_ngn_kg", 0), 1)
                train_log_prices.append(math.log(p) if p > 0 else 0.0)

        # We need at least 12 training prices; if fewer just skip
        if len(train_log_prices) < 12:
            return []

        # Rolling: after predicting each row, append the ACTUAL log-price
        # so lag features stay realistic for the next step
        rolling_log = list(train_log_prices)
        for r in test_rows:
            actual = _safe_float(r.get("fews_maize_price_ngn_kg", 0))
            try:
                pred = _predict_svm_rf_for_row(model_key, r, rolling_log)
            except Exception:
                pred = None
            results.append({
                "date"           : r["date"][:7],
                "actual_price"   : round(actual, 2),
                "predicted_price": pred,
                "model"          : model_key.upper(),
            })
            # Append the actual log-price so future steps see real history
            if actual > 0:
                rolling_log.append(math.log(actual))

    # ── ARIMA ─────────────────────────────────────────────────────────────────
    elif model_key == "arima":
        arima_model = model_data["model"]
        train_end   = model_data["train_end_date"]   # e.g. "2022-07-01"
        # Forecast enough steps to cover the entire test period
        last_test_row = test_rows[-1] if test_rows else None
        if not last_test_row:
            return []
        last_year  = int(last_test_row["year"])
        last_month = int(last_test_row["month"])
        n_steps    = _arima_steps_ahead(train_end, last_year, last_month)

        try:
            log_preds, _ = arima_model.predict(n_periods=n_steps, return_conf_int=True)
        except Exception:
            return []

        # Map predictions back to test-period dates
        # Step 1 is one month after train_end
        from datetime import datetime, timedelta
        end_dt   = datetime.strptime(train_end, "%Y-%m-%d")
        step_dt  = (end_dt.replace(day=28) + timedelta(days=4)).replace(day=1)

        pred_map: dict[str, float] = {}
        for i, lp in enumerate(log_preds):
            key = step_dt.strftime("%Y-%m")
            pred_map[key] = round(math.exp(float(lp)), 2)
            step_dt = (step_dt.replace(day=28) + timedelta(days=4)).replace(day=1)

        for r in test_rows:
            key    = r["date"][:7]
            actual = _safe_float(r.get("fews_maize_price_ngn_kg", 0))
            results.append({
                "date"           : key,
                "actual_price"   : round(actual, 2),
                "predicted_price": pred_map.get(key),
                "model"          : "ARIMA",
            })

    # ── LSTM ──────────────────────────────────────────────────────────────────
    elif model_key == "lstm":
        if model_data is None:
            return []
        scaler     = model_data["scaler"]
        lstm_model = model_data["model"]
        n_features = model_data["n_features"]

        for r in test_rows:
            actual = _safe_float(r.get("fews_maize_price_ngn_kg", 0))
            month  = int(r["month"])
            year   = int(r["year"])
            cloud  = _safe_float(r.get("climate_cloud_amt", 0))
            precip = _safe_float(r.get("design_avg_precip_mm", 0))
            wind   = _safe_float(r.get("design_avg_wind_ms", 0))
            temp   = _safe_float(r.get("temp_dbavg_c", 0))
            try:
                # Use all hist rows up to (not including) this row for context
                date_key = r["date"][:7]
                prefix   = [h for h in hist if h.get("date", "")[:7] < date_key]
                seq      = _build_lstm_sequence(
                    prefix, scaler, month, year,
                    cloud, precip, wind, temp, n_features,
                )
                scaled_pred = float(lstm_model.predict(seq, verbose=0)[0][0])
                dummy = np.zeros((1, n_features))
                dummy[0, 0] = scaled_pred
                log_pred = float(scaler.inverse_transform(dummy)[0, 0])
                pred = round(math.exp(log_pred), 2)
            except Exception:
                pred = None
            results.append({
                "date"           : r["date"][:7],
                "actual_price"   : round(actual, 2),
                "predicted_price": pred,
                "model"          : "LSTM",
            })

    return results


@app.get("/api/validation", tags=["Validation"])
def get_validation():
    """
    Replay the SVM model over the 2022-11 to 2024-05 test period.
    Returns [{date, actual_price, predicted_price, model}] sorted by date.
    """
    return _validation_for_model("svm")


@app.get("/api/validation/all", tags=["Validation"])
def get_validation_all():
    """
    Replay all 4 models over the 2022-11 to 2024-05 test period.
    Returns [{date, actual_price, svm, rf, arima, lstm}] sorted by date.
    """
    svm_rows   = {r["date"]: r for r in _validation_for_model("svm")}
    rf_rows    = {r["date"]: r for r in _validation_for_model("rf")}
    arima_rows = {r["date"]: r for r in _validation_for_model("arima")}
    lstm_rows  = {r["date"]: r for r in _validation_for_model("lstm")}

    all_dates = sorted(set(svm_rows) | set(rf_rows) | set(arima_rows) | set(lstm_rows))

    out = []
    for d in all_dates:
        actual = (
            svm_rows.get(d) or rf_rows.get(d) or
            arima_rows.get(d) or lstm_rows.get(d) or {}
        ).get("actual_price")
        out.append({
            "date"        : d,
            "actual_price": actual,
            "svm"         : svm_rows.get(d, {}).get("predicted_price"),
            "rf"          : rf_rows.get(d, {}).get("predicted_price"),
            "arima"       : arima_rows.get(d, {}).get("predicted_price"),
            "lstm"        : lstm_rows.get(d, {}).get("predicted_price"),
        })
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Run
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
