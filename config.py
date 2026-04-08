"""
Configuration for IMTISAL Maize Price Prediction Pipeline
"""
import os

# ── Google Drive ──────────────────────────────────────────────────────────────
GDRIVE_FOLDER_NAME = "IMTISAL"          # exact name of your Drive folder
CREDENTIALS_FILE   = "credentials.json" # OAuth2 desktop-app creds from GCP
TOKEN_FILE         = "token.json"        # auto-created after first auth

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
RAW_DIR       = os.path.join(BASE_DIR, "data", "raw")
CLEAN_DIR     = os.path.join(BASE_DIR, "data", "cleaned")
OUTPUT_DIR    = os.path.join(BASE_DIR, "data", "output")
REPORTS_DIR   = os.path.join(BASE_DIR, "reports")

for _d in (RAW_DIR, CLEAN_DIR, OUTPUT_DIR, REPORTS_DIR):
    os.makedirs(_d, exist_ok=True)

MASTER_CSV  = os.path.join(OUTPUT_DIR, "maize_master.csv")
EDA_REPORT  = os.path.join(REPORTS_DIR, "eda_report.html")

# ── Dataset hints ─────────────────────────────────────────────────────────────
# Map keywords in filenames → dataset type (used for smart cleaning)
DATASET_KEYWORDS = {
    "price"      : "price",
    "market"     : "market",
    "production" : "production",
    "climate"    : "climate",
    "rainfall"   : "climate",
    "weather"    : "climate",
    "yield"      : "production",
    "trade"      : "market",
}

# Expected date column name variants (will be normalised to "date")
DATE_COLUMN_VARIANTS = [
    "date", "dates", "year", "month", "period",
    "time", "week", "yearmonth", "year_month",
]

# Outlier IQR multiplier (1.5 = standard; 3.0 = only extreme outliers)
OUTLIER_IQR_MULTIPLIER = 2.5

# Merge key — all cleaned datasets must have this column after cleaning
MERGE_KEY = "date"
