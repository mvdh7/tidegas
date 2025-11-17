# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 13:17:20 2025

@author: nicor

Rename Bobometer CO2 data files using internal UTC timestamps.

"""

import os
import pandas as pd
from datetime import datetime

# Constants
FARADAY = 96485.3321  # C/mol

# Paths
folder_path = "data/vindta/r2co2/Bobometer"
main_csv = "DIC_logbook.csv"

# -------------------------------------------------------------------
# 1. Rename new-format files using internal UTC time
# -------------------------------------------------------------------

def extract_first_timestamp(file_path):
    """Extract the first valid UTC timestamp from the 'Time' column in a .txt file."""
    try:
        df = pd.read_csv(file_path)
        # Find a column named 'Time' (case-insensitive, ignore spaces)
        time_col = next((c for c in df.columns if c.strip().lower() == "time"), None)
        if time_col is None:
            return None

        first_time = df[time_col].dropna().iloc[0]
        dt = pd.to_datetime(first_time, utc=True)
        return dt
    except Exception as e:
        print(f"⚠️ Could not extract timestamp from {os.path.basename(file_path)}: {e}")
        return None


def rename_files_by_internal_time(folder):
    txt_files = [f for f in os.listdir(folder) if f.endswith(".txt")]
    timestamps = []

    for f in txt_files:
        fpath = os.path.join(folder, f)
        dt = extract_first_timestamp(fpath)
        if dt is not None:
            timestamps.append((f, dt))
        else:
            print(f"⚠️ No valid time found in {f}, skipping...")

    if not timestamps:
        print("❌ No valid files with timestamps found.")
        return

    # Sort files by internal UTC timestamp
    timestamps.sort(key=lambda x: x[1])

    # Group by date
    grouped = {}
    for fname, dt in timestamps:
        date_str = dt.strftime("%d-%m")
        grouped.setdefault(date_str, []).append((fname, dt))

    # Rename files by day and sequential order
    for date_str, files_for_day in grouped.items():
        for i, (orig_name, dt) in enumerate(files_for_day, start=1):
            new_name = f"co2data {date_str} ({i}).txt"
            src = os.path.join(folder, orig_name)
            dst = os.path.join(folder, new_name)

            if os.path.exists(dst):
                print(f"⚠️ Skipped (already exists): {new_name}")
                continue

            try:
                os.rename(src, dst)
                print(f"✅ Renamed: {orig_name} → {new_name}")
            except Exception as e:
                print(f"❌ Error renaming {orig_name}: {e}")


# Run it
rename_files_by_internal_time(folder_path)
