# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 11:14:19 2025

@author: nicor
"""

import os
import pandas as pd
import re
from datetime import datetime
import numpy as np 

FARADAY = 96485.3321  # C/mol



#TODO 
#THIS ERROR OCCURS WHEN PART OF THE LOG IS FILLED IN Aa
#     raise ValueError("cannot reindex on an axis with duplicate labels")

#ValueError: cannot reindex on an axis with duplicate labels
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
        "DIC file number": i-1,
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

# TODO unify this step, avoid loading files several times. extract everything at once, calculate differences etc at once. limit the use of lists.
def extract_data_from_DIC_file(file_path):
    """ 
    Compute and returns relevant data
    WIP
    Final raw integrated count
    Final count compensated for negative slope
    The difference of accounting for the negative slope in counts
    The integration duration in seconds
    """
    #start with reading the file
    try:
        
        file = pd.read_csv(file_path)
        # Find 'Cell[uAs]' column flexibly
        uAs_col = None
        for col in file.columns:
            c = col.lower().replace(" ", "")
            if "cell" in c and "uas" in c:
                uAs_col = col
                break
        if uAs_col and not file[uAs_col].dropna().empty:
            raw_current= file[uAs_col].dropna().iloc[-1]
        
       
       #compensating for the negative slope 
       # Spike correction (mask peaks >= 1e7)
        file[" Cell[uAs]"] = file[" Cell[uAs]"].mask(
            file[" Cell[uAs]"] >= 1e7,
            (file[" Cell[uAs]"].shift(1) + file[" Cell[uAs]"].shift(-1)) / 2
        )
        
        # Remove negative slopes
        corrected = [0]
        integrated_current_negative_removed = 0
        for i in range(len(file) - 1):
            if file[" Cell[uAs]"].iloc[i+1] <= file[" Cell[uAs]"].iloc[i]:
                integrated_current_negative_removed = integrated_current_negative_removed
            else:
                integrated_current_negative_removed += file[" Cell[uAs]"].iloc[i+1] - file[" Cell[uAs]"].iloc[i]
            
        
        #calculate difference
        difference = integrated_current_negative_removed-raw_current
            
        #calculate time duration (sample rate  = 1Hz)
        integration_time = len(file[" Cell[uAs]"])
    except Exception as e:
        print(f"⚠️ Error reading {file_path}: {e}")
    return raw_current,round(integrated_current_negative_removed,2), round(difference,2),int(integration_time)
        

files_df[["Integrated current raw [uAs]","Negative slope corrected current [uAs]","Difference negative removal", "Integration time"]] = [
    extract_data_from_DIC_file(os.path.join(folder_path, f))
    for f in files_df["DIC file name"]
]

#%%

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
    "Negative slope corrected current [uAs]",
    "Difference negative removal",
    "Integration time",
    "Raw DIC (umol/L)",
    "timestamp (UTC)"
]:
    if col not in main_table.columns:
        main_table[col] = None

# -------------------------------------------------------------------
# 5b. Merge/update columns without creating extra columns
# -------------------------------------------------------------------

update_cols = [
    "DIC file name",
    "File date",
    "Date index",
    "Integrated current raw [uAs]",
    "Negative slope corrected current [uAs]",
    "Difference negative removal",
    "Integration time",
    "timestamp (UTC)"
]

if main_table.empty:
    # First run: just copy all
    main_table = files_df.copy()
else:
    # Update existing rows or append new ones
    main_table.set_index("DIC file number", inplace=True)
    files_df.set_index("DIC file number", inplace=True)
    
    for col in update_cols:
        main_table[col] = files_df[col].combine_first(main_table[col])
    
    # Add any completely new rows not in main_table
    new_rows = files_df.index.difference(main_table.index)
    if len(new_rows) > 0:
        main_table = pd.concat([main_table, files_df.loc[new_rows]], axis=0)
    
    main_table.reset_index(inplace=True)
    files_df.reset_index(inplace=True)

# -------------------------------------------------------------------
# 6. Compute Raw DIC
# -------------------------------------------------------------------

#the return is a bit sketchy, but first limit the decimals to remove numerical noise and then make them floats again
#TODO make this less terrible, keeping them as strings creates problems for the reference 

def calc_dic_raw(row):
    if pd.notna(row["Integrated current raw [uAs]"]) and pd.notna(row.get("Volume (mL)")):
        return float("{:.4f}".format(row["Integrated current raw [uAs]"] / ((row["Volume (mL)"]/1000) * FARADAY)))
    else:
        return None
def calc_dic_negative_removed(row):
    if pd.notna(row["Negative slope corrected current [uAs]"]) and pd.notna(row.get("Volume (mL)")):
        return float("{:.4f}".format(row["Negative slope corrected current [uAs]"] / ((row["Volume (mL)"]/1000) * FARADAY)))
    else:
        return None
    
main_table["Raw DIC (umol/L)"] = main_table.apply(calc_dic_raw, axis=1)

main_table["Negative removed DIC (umol/L)"] = main_table.apply(calc_dic_negative_removed, axis=1)


# -------------------------------------------------------------------
# 6b. Compute daily reference averages from 0 acid added straight from the tank measurements
# -------------------------------------------------------------------

# Initialize the column if it doesn't exist
if "Daily reference DIC (umol/L)" not in main_table.columns:
    main_table["Daily reference DIC (umol/L)"] = np.nan

# Compute daily averages for reference samples
daily_ref = round(main_table.loc[main_table["Reference"] == 1].groupby("File date")[
    "Negative removed DIC (umol/L)"
].mean(),5)

# Assign the daily reference to all rows for that date
main_table["Daily reference DIC (umol/L)"] = main_table["File date"].map(daily_ref)

# -------------------------------------------------------------------
# 7. Save (overwrite live)
# -------------------------------------------------------------------

main_table.sort_values("DIC file number", inplace=True)
main_table.to_csv(csv_path, index=False)
print(f"✅ Live logbook updated ({len(main_table)} entries) and saved to {csv_path}")
