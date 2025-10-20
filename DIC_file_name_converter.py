import os
import pandas as pd
import re
from datetime import datetime

FARADAY = 96485.3321  # C/mol

# Paths
folder_path = "data/vindta/r2co2/Bobometer"
main_csv = "DIC_logbook.csv"
out_csv = "main_table_updated.csv"

# -------------------------------------------------------------------
# 1. Rename new-format files (e.g., co2data - 2025-10-09T140222.854.txt)
# -------------------------------------------------------------------

def rename_new_files(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".txt")]

    # Match files of form: co2data - 2025-10-09T140222.854.txt
    timestamp_pattern = re.compile(r"co2data\s*-\s*(\d{4}-\d{2}-\d{2}T\d{6}\.\d{3})\.txt")

    new_files = []
    for f in files:
        match = timestamp_pattern.match(f)
        if match:
            ts = match.group(1)
            try:
                dt = datetime.strptime(ts, "%Y-%m-%dT%H%M%S.%f")
                new_files.append((f, dt))
            except ValueError:
                continue

    # Sort by timestamp
    new_files.sort(key=lambda x: x[1])

    # Group files by date
    grouped = {}
    for original_name, dt in new_files:
        date_str = dt.strftime("%d-%m")
        grouped.setdefault(date_str, []).append((original_name, dt))

    # Rename with date-based pattern
    for date_str, files_for_day in grouped.items():
        for i, (orig_name, dt) in enumerate(files_for_day, start=1):
            new_name = f"co2data {date_str} ({i}).txt"
            src = os.path.join(folder, orig_name)
            dst = os.path.join(folder, new_name)

            if not os.path.exists(dst):
                os.rename(src, dst)
                print(f"Renamed: {orig_name} → {new_name}")
            else:
                print(f"Skipped (already exists): {dst}")

rename_new_files(folder_path)
#%%
# -------------------------------------------------------------------
# 2. Build ordered file list (first the 100 numeric ones, then dated)
# -------------------------------------------------------------------

all_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]

# regexes
numeric_re = re.compile(r"^co2data\s*\((\d+)\)\.txt$", re.IGNORECASE)
dated_re = re.compile(r"^co2data\s*(\d{2}-\d{2})\s*\((\d+)\)\.txt$", re.IGNORECASE)

def extract_number(fname):
    m = numeric_re.match(fname)
    return int(m.group(1)) if m else None

def extract_date_index(fname):
    m = dated_re.match(fname)
    if m:
        month_day = m.group(1)         # '10-09'
        idx = int(m.group(2))          # (1)
        # to sort by actual date we could attach year if needed; for now we trust chronological renaming
        return (month_day, idx)
    return (None, None)

# Separate lists
numeric_files = sorted([f for f in all_files if numeric_re.match(f)], key=lambda f: extract_number(f))
dated_files   = sorted([f for f in all_files if dated_re.match(f)], key=lambda f: extract_date_index(f))

# Build filename -> file_number mapping
filename_to_number = {}

# numeric files keep their numbers
for f in numeric_files:
    filename_to_number[f] = extract_number(f)

max_num = max(filename_to_number.values()) if filename_to_number else 0
next_num = max_num + 1

# assign sequential numbers to dated files in chronological order (the order we sorted above)
for f in dated_files:
    filename_to_number[f] = next_num
    next_num += 1

# final ordered tfiles: numeric first, then dated
tfiles = numeric_files + dated_files

print(f"Found {len(numeric_files)} numeric files (max={max_num}).")
print(f"Found {len(dated_files)} dated files, numbering will continue from {max_num+1}.")
print("Example mapping (first 10):")
for i, f in enumerate(tfiles[-10:], 1):
    print(f"  {i:2d}. {f} -> file_number {filename_to_number[f]}")

# -------------------------------------------------------------------
# 3 (fixed). Load table and process data using the mapping
# -------------------------------------------------------------------
#%%
main_table = pd.read_csv(main_csv)

file_data = {}  # {file_number (int): integrated_current}

for f in tfiles:
    file_path = os.path.join(folder_path, f)
    file_num = filename_to_number.get(f)
    if file_num is None:
        print(f"Skipping (no assigned file number): {f}")
        continue

    try:
        df = pd.read_csv(file_path)

        # robustly find the 'Cell[uAs]' column (handles leading/trailing spaces or variations)
        uAs_col = None
        for col in df.columns:
            ncol = col.lower().replace(" ", "")
            if "cell" in ncol and "uas" in ncol:
                uAs_col = col
                break

        if uAs_col is not None and not df[uAs_col].dropna().empty:
            integrated_current = df[uAs_col].dropna().iloc[-1]
            file_data[file_num] = integrated_current
        else:
            # no valid value found
            file_data[file_num] = None
            print(f"Note: no Cell[uAs] value in {f}")

    except Exception as e:
        print(f"Error reading {f}: {e}")
        file_data[file_num] = None

# Map into main_table.
# IMPORTANT: DIC file numbers in the CSV may be floats (e.g., 101.0). Convert to int safely before lookup.
def lookup_integrated(val):
    if pd.isna(val):
        return None
    try:
        key = int(val)
    except Exception:
        # if it's not convertible, try direct lookup
        return file_data.get(val, None)
    return file_data.get(key, None)

main_table["Integrated current raw [uAs]"] = main_table["DIC file number"].apply(lookup_integrated)

# Report mapping coverage
mapped_count = main_table["Integrated current raw [uAs]"].notna().sum()
print(f"Mapped integrated-current to {mapped_count} rows out of {len(main_table)} in the main table.")

# Calculate Raw DIC as before
def calc_dic(row):
    if pd.notna(row["Integrated current raw [uAs]"]) and pd.notna(row["Volume (mL)"]):
        return float(row["Integrated current raw [uAs]"] / ((row["Volume (mL)"]/1000) * FARADAY))
    else:
        return None

main_table["Raw DIC (umol/L)"] = main_table.apply(calc_dic, axis=1)
main_table.to_csv(out_csv, index=False)
print(f"✅ Updated table saved to {out_csv}")
