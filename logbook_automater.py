import re
import os
from pathlib import Path
import pandas as pd
from datetime import datetime

# ---------- CONFIG ----------
# Path to your existing Excel file
excel_file = "logbook_automated_by_python_testing.xlsx"

# Folder containing your data files
folder_path = "data/vindta/r2co2/Nico"
# Read existing Excel
df = pd.read_excel(excel_file)

# Pattern: match junk/sample, YYMMDD, bottle number
pattern = re.compile(r'(junk|sample)-(\d{6})-(\d+)', re.I)

# Only process .dat files
dat_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.dat')]

rows = []

for file in dat_files:
    match = pattern.search(file.lower())
    if match:
        sample_type = match.group(1).capitalize()
        date_str = match.group(2)
        bottle = int(match.group(3))

        # Convert date to DD-MMM
        try:
            date_obj = datetime.strptime(date_str, "%y%m%d")
            date_fmt = date_obj.strftime("%d-%b")
        except:
            date_fmt = ""

        rows.append({
            "date": date_fmt,
            "sample/junk": sample_type,
            "bottle": bottle,
            "file name alkalinity": file  # add full file name
        })

# Build DataFrame
df_files = pd.DataFrame(rows)

# Sort by date then bottle
df_files['date_sort'] = pd.to_datetime(df_files['date'], format='%d-%b', errors='coerce')
df_files.sort_values(by=['date_sort', 'bottle'], inplace=True)
df_files.reset_index(drop=True, inplace=True)
df_files.drop(columns=['date_sort'], inplace=True)

# Update the existing Excel sheet
# Make sure we don’t overwrite other columns
# Assuming row count matches
for col in ["date", "sample/junk", "bottle", "file name alkalinity"]:
    if col in df.columns:
        df[col] = df_files[col]
    else:
        df[col] = df_files[col]  # add new column if missing

# Save back
print(df)
df.to_excel(excel_file, index=False)

print(f"✅ Updated {len(df_files)} rows in {excel_file}")
