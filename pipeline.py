"""
IMTISAL Maize Price Prediction — Main Pipeline
==============================================

Usage
-----
  # Full pipeline (auth → download → clean → merge → EDA):
  python pipeline.py

  # Skip download (use files already in data/raw/):
  python pipeline.py --skip-download

  # Skip download AND cleaning (use files already in data/cleaned/):
  python pipeline.py --skip-download --skip-clean

  # EDA only on existing master CSV:
  python pipeline.py --eda-only

Credentials
-----------
Place 'credentials.json' (OAuth2 Desktop Client from Google Cloud Console)
in the same directory as this script before running.
"""

import argparse
import glob
import json
import os
import sys
import time

# ── ensure project root is on sys.path ───────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import RAW_DIR, CLEAN_DIR, OUTPUT_DIR, REPORTS_DIR, MASTER_CSV


def banner(text: str):
    bar = "═" * (len(text) + 4)
    print(f"\n╔{bar}╗\n║  {text}  ║\n╚{bar}╝")


def collect_raw_files() -> list[str]:
    patterns = [
        os.path.join(RAW_DIR, "*.csv"),
        os.path.join(RAW_DIR, "*.xlsx"),
        os.path.join(RAW_DIR, "*.xls"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    return sorted(files)


def collect_clean_files() -> list[str]:
    return sorted(glob.glob(os.path.join(CLEAN_DIR, "*_clean.csv")))


def main():
    parser = argparse.ArgumentParser(description="IMTISAL Maize ML Pipeline")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip Google Drive download; use existing raw files")
    parser.add_argument("--skip-clean",    action="store_true",
                        help="Skip cleaning; use existing cleaned files")
    parser.add_argument("--eda-only",      action="store_true",
                        help="Only regenerate the EDA report from existing master CSV")
    args = parser.parse_args()

    t0 = time.time()
    banner("IMTISAL Maize Price Prediction Pipeline")
    print(f"Working dir : {os.path.dirname(os.path.abspath(__file__))}")
    print(f"Raw data    : {RAW_DIR}")
    print(f"Cleaned data: {CLEAN_DIR}")
    print(f"Output      : {OUTPUT_DIR}")
    print(f"Reports     : {REPORTS_DIR}")

    # ── EDA-only shortcut ──────────────────────────────────────────────────────
    if args.eda_only:
        banner("Step 4: EDA Report (only)")
        from eda import generate_eda
        report_path = generate_eda()
        print(f"\nDone in {time.time()-t0:.1f}s  →  open {report_path}")
        return

    # ── Step 1: Google Drive download ──────────────────────────────────────────
    raw_files = []
    if not args.skip_download:
        banner("Step 1: Authenticate & Download from Google Drive")
        from gdrive_auth import get_drive_service, download_all
        service   = get_drive_service()
        raw_files = download_all(service)
        if not raw_files:
            print("[warn] No files downloaded. Check folder name / permissions.")
    else:
        banner("Step 1: Skipped (--skip-download)")
        raw_files = collect_raw_files()
        if not raw_files:
            print(f"[warn] No raw files found in {RAW_DIR}")
            print("       Place your CSV/Excel files there and re-run, or remove --skip-download.")
            sys.exit(1)
        print(f"       Using {len(raw_files)} existing raw file(s).")

    # ── Step 2: Clean ─────────────────────────────────────────────────────────
    clean_dfs = {}
    if not args.skip_clean:
        banner("Step 2: Clean Datasets")
        from cleaner import clean_all
        clean_dfs, reports = clean_all(raw_files)

        # Save cleaning report
        report_path = os.path.join(REPORTS_DIR, "cleaning_report.json")
        with open(report_path, "w") as fh:
            json.dump(reports, fh, indent=2, default=str)
        print(f"\n[clean] Cleaning report saved → {report_path}")
    else:
        banner("Step 2: Skipped (--skip-clean)")
        clean_paths = collect_clean_files()
        if not clean_paths:
            print(f"[warn] No cleaned files found in {CLEAN_DIR}. Run without --skip-clean first.")
            sys.exit(1)
        import pandas as pd
        for p in clean_paths:
            clean_dfs[os.path.basename(p)] = pd.read_csv(p)
        print(f"       Loaded {len(clean_dfs)} cleaned file(s).")

    if not clean_dfs:
        print("[ERROR] No cleaned datasets available for merging.")
        sys.exit(1)

    # ── Step 3: Merge ─────────────────────────────────────────────────────────
    banner("Step 3: Merge into Master Dataset")
    from merger import merge_datasets
    master_df = merge_datasets(clean_dfs)

    # ── Step 4: EDA ───────────────────────────────────────────────────────────
    banner("Step 4: Generate EDA Report")
    from eda import generate_eda
    eda_path = generate_eda(master_df)

    # ── Summary ───────────────────────────────────────────────────────────────
    banner("Pipeline Complete")
    print(f"  Master CSV   : {MASTER_CSV}")
    print(f"  EDA report   : {eda_path}")
    print(f"  Total time   : {time.time()-t0:.1f}s")
    print()
    print("Next steps:")
    print("  1. Open reports/eda_report.html in your browser to explore the data.")
    print("  2. Use data/output/maize_master.csv as input to your ML model.")
    print("  3. Suggested target column: the primary 'price' column in the merged data.")


if __name__ == "__main__":
    main()
