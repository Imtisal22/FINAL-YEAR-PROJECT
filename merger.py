"""
Merge all cleaned datasets into one master CSV for ML training.

Strategy:
  - All datasets that share a 'date' column are outer-merged on date.
  - Datasets without a date column are broadcast (added as scalar features).
  - After merging, a final missing-value pass fills remaining NaNs.
  - The master CSV is saved to OUTPUT_DIR/maize_master.csv.
"""

import os
import numpy as np
import pandas as pd
from functools import reduce

from config import MASTER_CSV, MERGE_KEY


def merge_datasets(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    dfs: {filename: cleaned_dataframe}
    Returns merged master DataFrame.
    """
    print("\n[merge] ── Merging datasets ──")

    has_date, no_date = [], []
    for fname, df in dfs.items():
        if MERGE_KEY in df.columns:
            has_date.append((fname, df))
        else:
            no_date.append((fname, df))

    if not has_date:
        raise ValueError("[merge] No datasets with a 'date' column found — cannot merge.")

    # Sort each frame by date before merging
    frames_sorted = []
    for fname, df in has_date:
        df = df.copy()
        df[MERGE_KEY] = pd.to_datetime(df[MERGE_KEY], errors="coerce")
        df.sort_values(MERGE_KEY, inplace=True)
        df[MERGE_KEY] = df[MERGE_KEY].dt.strftime("%Y-%m-%d")
        # Prefix non-key columns with dataset type to avoid collisions
        src = df.get("dataset_source", pd.Series(["unknown"])).iloc[0]
        rename_map = {
            c: f"{src}_{c}" if c not in (MERGE_KEY, "dataset_source") else c
            for c in df.columns
        }
        df.rename(columns=rename_map, inplace=True)
        df.drop(columns=["dataset_source"], errors="ignore", inplace=True)
        frames_sorted.append(df)
        print(f"        {fname}: {len(df)} rows  [{df[MERGE_KEY].iloc[0]} → {df[MERGE_KEY].iloc[-1]}]")

    # Outer-merge all frames on date
    master = reduce(
        lambda left, right: pd.merge(left, right, on=MERGE_KEY, how="outer"),
        frames_sorted,
    )

    # Deduplicate column names that appeared in multiple sources
    master = master.loc[:, ~master.columns.duplicated()]

    # Broadcast date-less datasets (single-row summary features)
    for fname, df in no_date:
        print(f"        {fname}: no date column — broadcasting {len(df.columns)} columns")
        for col in df.columns:
            if col != "dataset_source":
                master[col] = df[col].iloc[0] if len(df) == 1 else df[col].mean()

    # Final missing-value fill
    master.sort_values(MERGE_KEY, inplace=True)
    master.reset_index(drop=True, inplace=True)

    num_cols = master.select_dtypes(include=[np.number]).columns
    master[num_cols] = master[num_cols].fillna(master[num_cols].median())

    cat_cols = master.select_dtypes(include=["object"]).columns.difference([MERGE_KEY])
    for col in cat_cols:
        if master[col].isna().any():
            master[col].fillna(master[col].mode().iloc[0] if not master[col].mode().empty else "unknown", inplace=True)

    # Add calendar features useful for ML
    master["date_parsed"] = pd.to_datetime(master[MERGE_KEY], errors="coerce")
    master["year"]  = master["date_parsed"].dt.year
    master["month"] = master["date_parsed"].dt.month
    master["quarter"] = master["date_parsed"].dt.quarter
    master.drop(columns=["date_parsed"], inplace=True)

    master.to_csv(MASTER_CSV, index=False)
    print(f"\n[merge] Master CSV saved → {MASTER_CSV}")
    print(f"        Shape: {master.shape[0]} rows × {master.shape[1]} columns")
    print(f"        Date range: {master[MERGE_KEY].iloc[0]} → {master[MERGE_KEY].iloc[-1]}")
    print(f"        Columns: {list(master.columns)}")

    return master
