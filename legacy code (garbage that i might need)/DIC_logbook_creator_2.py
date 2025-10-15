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
        # New-style filename with embedded timestamp
        try:
            dt = datetime.strptime(m.group(1), "%Y-%m-%dT%H%M%S.%f")
        except ValueError:
            dt = datetime.fromtimestamp(os.path.getmtime(fpath))
    else:
        # Legacy filename → use file's modification time
        dt = datetime.fromtimestamp(os.path.getmtime(fpath))

    files_info.append((fname, dt))

# Sort chronologically
files_info.sort(key=lambda x: x[1])

# -------------------------------------------------------------------
# 3. Build metadata (date + index per day)
# -------------------------------------------------------------------

records = []
daily_counts = {}

for i, (fname, dt) in enumerate(files_info, start=1):
    date_str = dt.strftime("%Y-%m-%d")
    daily_counts.setdefault(date_str, 0)
    daily_counts[date_str] += 1
    date_idx = daily_counts[date_str]
    timestamp =dt.strftime("%H:%M:%S")
    records.append({
        "DIC file number": i,
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


def extract_integrated_current(file_path):
    try:
        df = pd.read_csv(file_path)
        # Find 'Cell[uAs]' column flexibly
        uAs_col = None
        for col in df.columns:
            c = col.lower().replace(" ", "")
            if "cell" in c and "uas" in c:
                uAs_col = col
                break
        if uAs_col and not df[uAs_col].dropna().empty:
            return df[uAs_col].dropna().iloc[-1]
    except Exception as e:
        print(f"⚠️ Error reading {file_path}: {e}")
    return None

files_df["Integrated current raw [uAs]"] = [
    extract_integrated_current(os.path.join(folder_path, f))
    for f in files_df["DIC file name"]
]

def process_negative_slope(file_path):
    """
    Process a CO2 data file to compute:
    - Negative slope corrected current
    - Difference vs raw current
    - Positive slope
    
    Returns a dict with results.
    """
    file = pd.read_csv(file_path)
    
    if " Cell[uAs]" not in file.columns:
        raise ValueError(f"File {file_path} has no 'Cell[uAs]' column.")
    
    raw_current = file[" Cell[uAs]"].to_numpy()
    
    # Spike correction (mask peaks >= 1e7)
    file[" Cell[uAs]"] = file[" Cell[uAs]"].mask(
        file[" Cell[uAs]"] >= 1e7,
        (file[" Cell[uAs]"].shift(1) + file[" Cell[uAs]"].shift(-1)) / 2
    )
    
    # Remove negative slopes
    corrected = [0]
    integrated_current = 0
    for i in range(len(file) - 1):
        if file[" Cell[uAs]"].iloc[i+1] <= file[" Cell[uAs]"].iloc[i]:
            integrated_current = integrated_current
        else:
            integrated_current += file[" Cell[uAs]"].iloc[i+1] - file[" Cell[uAs]"].iloc[i]
        corrected.append(integrated_current)
    corrected = np.array(corrected)
    
    # Difference between corrected and raw
    difference = corrected - raw_current
    
    # Positive slope (differences > 0.0001)
    slope = pd.Series(difference).diff()
    slope = slope[slope > 0.0001].values
    
    return {
        "total_cell_integrated": raw_current,
        "total_cell_negative_removed": corrected,
        "negative_removed_difference": difference,
        "slope": slope
    }
# -------------------------------------------------------------------
# 5. Load or create live logbook
# -------------------------------------------------------------------

if os.path.exists(csv_path):
    main_table = pd.read_csv(csv_path)
else:
    main_table = pd.DataFrame()

# Ensure columns exist
for col in [
    "DIC file number",
    "DIC file name",
    "File date",
    "Date index",
    "Integrated current raw [uAs]",
    "Raw DIC (umol/L)",
]:
    if col not in main_table.columns:
        main_table[col] = None

# Merge with existing
main_table = pd.merge(
    main_table,
    files_df,
    on="DIC file number",
    how="outer",
    suffixes=("", "_new")
)

# Combine updated columns
for col in [
    "DIC file name",
    "File date",
    "Date index",
    "Integrated current raw [uAs]",
]:
    main_table[col] = main_table[f"{col}_new"].combine_first(main_table[col])
    if f"{col}_new" in main_table.columns:
        main_table.drop(columns=[f"{col}_new"], inplace=True)

# -------------------------------------------------------------------
# 6. Compute Raw DIC
# -------------------------------------------------------------------

def calc_dic(row):
    if pd.notna(row["Integrated current raw [uAs]"]) and pd.notna(row.get("Volume (mL)")):
        return float(row["Integrated current raw [uAs]"] / ((row["Volume (mL)"]/1000) * FARADAY))
    else:
        return None

main_table["Raw DIC (umol/L)"] = main_table.apply(calc_dic, axis=1)

# -------------------------------------------------------------------
# 7. Save (overwrite live)
# -------------------------------------------------------------------

main_table.sort_values("DIC file number", inplace=True)
main_table.to_csv(csv_path, index=False)
print(f"✅ Live logbook updated ({len(main_table)} entries) and saved to {csv_path}")
