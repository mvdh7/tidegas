# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 11:30:35 2025

@author: nicor (modified)
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
# Helpers: parse date from filename if present (several common patterns)
# -------------------------------------------------------------------
def parse_date_from_filename(fname):
    """
    Try multiple filename patterns to extract a date string in YYYY-MM-DD form.
    Returns a datetime.date or None if not found.
    Patterns attempted (in order):
      - ISO-like embedded: 2025-11-07T120000.123  (already in your code)
      - DD-MM-YYYY or D-M-YYYY or DD_MM_YYYY
      - DD-MM or D-M  (assume current year)
      - MM-DD or MM-DD-YYYY (common US-ish)
      - '10-11' could be day-month; we assume day-month if ambiguous
    """
    # 1) ISO timestamp used earlier: 2025-11-07T120000.123
    m = re.search(r"(\d{4}-\d{2}-\d{2}T\d{6}\.\d{3})", fname)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y-%m-%dT%H%M%S.%f").date()
        except Exception:
            pass

    # 2) DD-MM-YYYY or DD_MM_YYYY or D-M-YYYY
    m = re.search(r"(\b\d{1,2}[-_]\d{1,2}[-_]\d{4}\b)", fname)
    if m:
        for fmt in ("%d-%m-%Y", "%d_%m_%Y", "%m-%d-%Y", "%m_%d_%Y"):
            try:
                return datetime.strptime(m.group(1).replace("_", "-"), fmt).date()
            except Exception:
                pass

    # 3) DD-MM or D-M (assume current year)
    m = re.search(r"\b(\d{1,2})[-_](\d{1,2})(?![-_]\d{2,4})\b", fname)
    if m:
        day = int(m.group(1))
        month = int(m.group(2))
        year = datetime.now().year
        try:
            return datetime(year, month, day).date()
        except Exception:
            pass

    # 4) MM-DD (assume current year) if previous didn't match (less likely for your files)
    m = re.search(r"\b(\d{1,2})[-_](\d{1,2})\b", fname)
    if m:
        # ambiguous; prefer treating as day-month (as you indicated day-first)
        day = int(m.group(1))
        month = int(m.group(2))
        year = datetime.now().year
        try:
            return datetime(year, month, day).date()
        except Exception:
            pass

    return None

# -------------------------------------------------------------------
# 2. Detect timestamps for all files (prefer filename date when available)
# -------------------------------------------------------------------

timestamp_pattern = re.compile(r"co2data\s*-\s*(\d{4}-\d{2}-\d{2}T\d{6}\.\d{3})\.txt", re.IGNORECASE)
txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]

files_info = []

for fname in txt_files:
    fpath = os.path.join(folder_path, fname)

    # Try parse from filename first (preferred)
    parsed_date = parse_date_from_filename(fname)
    if parsed_date is not None:
        # Use midnight for time component when only date is known
        dt = datetime.combine(parsed_date, datetime.min.time())
    else:
        # fallback to timestamp embedded in filename (original pattern)
        m = timestamp_pattern.search(fname)
        if m:
            try:
                dt = datetime.strptime(m.group(1), "%Y-%m-%dT%H%M%S.%f")
            except ValueError:
                dt = datetime.fromtimestamp(os.path.getmtime(fpath))
        else:
            # final fallback: file modification time
            dt = datetime.fromtimestamp(os.path.getmtime(fpath))

    files_info.append((fname, dt))

# keep deterministic order: sort by datetime then filename (but we will preserve main_table order later)
files_info.sort(key=lambda x: (x[1], x[0].lower()))

# -------------------------------------------------------------------
# 3. Build metadata (date + index per day) — using dt (which may originate from filename)
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
print(f"🧾 Found {len(files_df)} files across {len(daily_counts)} dates (derived).")

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

# vectorized list comprehension to preserve file order in files_df
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

# capture original order (list of filenames) so we can preserve it
original_order = list(main_table["DIC file name"]) if "DIC file name" in main_table.columns else []

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

# -------------------------------------------------------------------
# 6. Merge/update by DIC file name (unique key)
# -------------------------------------------------------------------

update_cols = [
    "File date", "Date index",
    "Integrated current raw [uAs]", "Negative slope corrected current [uAs]",
    "Difference negative removal", "Integration time", "timestamp (UTC)"
]

# Drop duplicates by filename (keep last)
files_df = files_df.drop_duplicates(subset="DIC file name", keep="last")
main_table = main_table.drop_duplicates(subset="DIC file name", keep="last")

if main_table.empty:
    # fresh start: use files_df as main_table
    main_table = files_df.copy()
else:
    # set index on filename for safe merging
    main_table = main_table.set_index("DIC file name")
    files_df = files_df.set_index("DIC file name")

    # Update only missing values in main_table from files_df
    for col in update_cols:
        if col in main_table.columns and col in files_df.columns:
            main_table.loc[:, col] = main_table[col].where(main_table[col].notna(), files_df[col])

    # For existing filenames we intentionally do NOT reorder main_table rows here.
    # We WILL refresh file-date metadata if the filename *explicitly* contains a date
    # (i.e., parse_date_from_filename returned a concrete date when we built files_info).
    # So update metadata for files where parsed date existed.
    # We can identify those because files_df["File date"] was computed from parsed dt.
    # Replace metadata for entries that exist in both but where we *trust* the filename-derived value.
    # To be safe, only override File date/Date index/timestamp when the file date in files_df is different.
    overlap = main_table.index.intersection(files_df.index)
    for fn in overlap:
        f_new = files_df.loc[fn]
        # if file date differs and original ordering should be preserved, update metadata but keep row position
        if "File date" in f_new.index:
            if pd.isna(main_table.at[fn, "File date"]) or main_table.at[fn, "File date"] != f_new["File date"]:
                main_table.at[fn, "File date"] = f_new["File date"]
                main_table.at[fn, "Date index"] = f_new.get("Date index", main_table.at[fn, "Date index"])
                main_table.at[fn, "timestamp (UTC)"] = f_new.get("timestamp (UTC)", main_table.at[fn, "timestamp (UTC)"])

    # Add completely new files (filenames not in main_table)
    new_files_idx = files_df.index.difference(main_table.index)
    if not new_files_idx.empty:
        new_rows = files_df.loc[new_files_idx]
        print(f"🆕 Adding {len(new_rows)} new files to logbook.")
        # append new rows after existing rows (this preserves existing order)
        main_table = pd.concat([main_table, new_rows])

    # restore index as column
    main_table.reset_index(inplace=True)
    files_df.reset_index(inplace=True)

    # Reorder main_table to put original rows in the same order as before (if available)
    if original_order:
        # Build final order: existing original_order (only those still present), then any remaining files (new)
        existing_in_order = [fn for fn in original_order if fn in main_table["DIC file name"].values]
        remaining = [fn for fn in main_table["DIC file name"].values if fn not in existing_in_order]
        final_order = existing_in_order + remaining
        # Reindex main_table according to final_order
        main_table["__order_key__"] = pd.Categorical(main_table["DIC file name"], categories=final_order, ordered=True)
        main_table.sort_values("__order_key__", inplace=True)
        main_table.drop(columns="__order_key__", inplace=True)
        main_table.reset_index(drop=True, inplace=True)

# -------------------------------------------------------------------
# 7. Ensure default Volume
# -------------------------------------------------------------------

main_table["Volume (mL)"] = main_table["Volume (mL)"].fillna(25.0)

# -------------------------------------------------------------------
# 8. Compute Raw and Corrected DIC
# -------------------------------------------------------------------

vol_l = main_table["Volume (mL)"] / 1000
main_table["Raw DIC (umol/L)"] = (
    main_table["Integrated current raw [uAs]"] / (vol_l * FARADAY)
).round(4)
main_table["Negative removed DIC (umol/L)"] = (
    main_table["Negative slope corrected current [uAs]"] / (vol_l * FARADAY)
).round(4)

# -------------------------------------------------------------------
# 9. Compute daily reference averages (by File date + Sample type)
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
# 10. Save and summary
# -------------------------------------------------------------------

main_table.to_csv(csv_path, index=False)
print(f"✅ Live logbook updated ({len(main_table)} entries) and saved to {csv_path}")

# helpful summary
min_date = main_table["File date"].min()
max_date = main_table["File date"].max()
print(f"Date range in logbook: {min_date} → {max_date}")
