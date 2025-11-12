# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 11:30:35 2025

@author: nicor
"""

import os
import pandas as pd
import re
from datetime import datetime
import numpy as np

FARADAY = 96485.3321  # C/mol

# -------------------------------------------------------------------
# 1. Config
# -------------------------------------------------------------------
folder_path = "data/vindta/r2co2/Bobometer"
csv_path = "DIC_logbook.csv"  # same file used for input/output

# -------------------------------------------------------------------
# 2. Detect timestamps for all files
# -------------------------------------------------------------------

timestamp_pattern = re.compile(r"co2data\s*-\s*(\d{4}-\d{2}-\d{2}T\d{6}\.\d{3})\.txt", re.IGNORECASE)
txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]

files_info = []

for fname in txt_files:
    fpath = os.path.join(folder_path, fname)
    m = timestamp_pattern.match(fname)
    if m:
        try:
            dt = datetime.strptime(m.group(1), "%Y-%m-%dT%H%M%S.%f")
        except ValueError:
            dt = datetime.fromtimestamp(os.path.getmtime(fpath))
    else:
        dt = datetime.fromtimestamp(os.path.getmtime(fpath))
    files_info.append((fname, dt))

# Keep sorted by datetime for new-batch detection
files_info.sort(key=lambda x: x[1])

# -------------------------------------------------------------------
# 3. Build metadata (date + index per day)
# -------------------------------------------------------------------

records = []
daily_counts = {}

for fname, dt in files_info:
    date_str = dt.strftime("%Y-%m-%d")
    daily_counts.setdefault(date_str, 0)
    daily_counts[date_str] += 1
    date_idx = daily_counts[date_str]
    timestamp = dt.strftime("%H:%M:%S")
    records.append({
        "DIC file name": fname,
        "File date": date_str,
        "Date index": date_idx,
        "timestamp (UTC)": timestamp,
    })

files_df = pd.DataFrame(records)
print(f"🧾 Found {len(files_df)} files across {len(daily_counts)} dates.")

# -------------------------------------------------------------------
# 4. Extract integrated current from files
# -------------------------------------------------------------------

def extract_data_from_DIC_file(file_path):
    """Compute and return raw and corrected integrated current values."""
    try:
        file = pd.read_csv(file_path)
        uAs_col = next((col for col in file.columns if "cell" in col.lower() and "uas" in col.lower()), None)
        if not uAs_col or file[uAs_col].dropna().empty:
            return [np.nan] * 4

        vals = file[uAs_col].copy()
        raw_current = vals.dropna().iloc[-1]

        # Spike correction
        vals = vals.mask(vals >= 1e7, (vals.shift(1) + vals.shift(-1)) / 2)

        # Remove negative slopes
        diff = vals.diff().clip(lower=0)
        integrated = diff.sum()

        return [raw_current, round(integrated, 2), round(integrated - raw_current, 2), len(vals)]

    except Exception as e:
        print(f"⚠️ Error reading {file_path}: {e}")
        return [np.nan] * 4

files_df[[
    "Integrated current raw [uAs]",
    "Negative slope corrected current [uAs]",
    "Difference negative removal",
    "Integration time"
]] = [extract_data_from_DIC_file(os.path.join(folder_path, f)) for f in files_df["DIC file name"]]

# -------------------------------------------------------------------
# 5. Load or create live logbook
# -------------------------------------------------------------------

main_table = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()

# Ensure required columns exist
required_cols = [
    "DIC file name", "File date", "Date index",
    "Integrated current raw [uAs]", "Negative slope corrected current [uAs]",
    "Difference negative removal", "Integration time",
    "Raw DIC (umol/L)", "timestamp (UTC)",
    "Corresponding Vindta", "Reference", "Volume (mL)", "Sample type"
]
for col in required_cols:
    if col not in main_table.columns:
        main_table[col] = np.nan

# Preserve old order before we modify
original_order = main_table["DIC file name"].dropna().tolist() if not main_table.empty else []

# -------------------------------------------------------------------
# 6. Merge/update by DIC file name (unique key)
# -------------------------------------------------------------------

update_cols = [
    "File date", "Date index",
    "Integrated current raw [uAs]", "Negative slope corrected current [uAs]",
    "Difference negative removal", "Integration time", "timestamp (UTC)"
]

# Drop duplicates
files_df = files_df.drop_duplicates(subset="DIC file name", keep="last")
main_table = main_table.drop_duplicates(subset="DIC file name", keep="last")

if main_table.empty:
    main_table = files_df.copy()
else:
    main_table = main_table.set_index("DIC file name")
    files_df = files_df.set_index("DIC file name")

    # Update existing rows — fill missing values, but don't overwrite valid ones
    for col in update_cols:
        if col in main_table.columns and col in files_df.columns:
            s_main = main_table[col]
            s_new = files_df[col]
            main_table.loc[:, col] = s_main.where(s_main.notna(), s_new)

    # Add new rows (new files)
    new_rows = files_df.loc[files_df.index.difference(main_table.index)]
    if not new_rows.empty:
        print(f"🆕 Adding {len(new_rows)} new files to logbook.")
        main_table = pd.concat([main_table, new_rows])

    main_table.reset_index(inplace=True)
    files_df.reset_index(inplace=True)

# -------------------------------------------------------------------
# 7. Preserve original order and append new ones in sorted order
# -------------------------------------------------------------------

def natural_sort_key(s):
    # Extract all numbers in the filename, convert to int
    nums = re.findall(r'\d+', s)
    return [int(n) for n in nums]

# Build final order
existing_files = [fn for fn in original_order if fn in main_table["DIC file name"].values]
new_files = sorted(
    [fn for fn in main_table["DIC file name"].values if isinstance(fn, str) and fn not in existing_files],
    key=natural_sort_key
)

final_order = existing_files + new_files

# Apply order
order_map = {fn: i for i, fn in enumerate(final_order)}
main_table["__order_key__"] = main_table["DIC file name"].map(order_map)
main_table.sort_values("__order_key__", inplace=True, na_position="last")
main_table.drop(columns="__order_key__", inplace=True)
main_table.reset_index(drop=True, inplace=True)

# -------------------------------------------------------------------
# 8. Ensure default Volume
# -------------------------------------------------------------------

main_table["Volume (mL)"] = main_table["Volume (mL)"].fillna(25.0)

# -------------------------------------------------------------------
# 9. Compute Raw and Corrected DIC
# -------------------------------------------------------------------

vol_l = main_table["Volume (mL)"] / 1000
main_table["Raw DIC (umol/L)"] = (
    main_table["Integrated current raw [uAs]"] / (vol_l * FARADAY)
).round(4)
main_table["Negative removed DIC (umol/L)"] = (
    main_table["Negative slope corrected current [uAs]"] / (vol_l * FARADAY)
).round(4)

# -------------------------------------------------------------------
# 10. Compute daily reference averages (by date + sample type)
# -------------------------------------------------------------------

daily_ref = (
    main_table.loc[main_table["Reference"] == 1]
    .groupby(["File date", "Sample type"])["Negative removed DIC (umol/L)"]
    .mean()
    .round(5)
)

main_table["Daily reference DIC (umol/L)"] = main_table.set_index(
    ["File date", "Sample type"]
).index.map(daily_ref)

# -------------------------------------------------------------------
# 11. Save
# -------------------------------------------------------------------

main_table.to_csv(csv_path, index=False)
print(f"✅ Live logbook updated ({len(main_table)} entries) and saved to {csv_path}")
